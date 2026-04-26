// render/vulkan/shaders/geodesic_kernel.hlsl
//
// HLSL port of render/metal/shaders/geodesic_kernel.metal. Per-pixel null-
// geodesic ray tracer compiled to SPIR-V by DXC for the Vulkan backend.
// Shares the physics headers under shared_shader/ with the Metal and host
// paths, so a bug fix in geodesic_math.h / kerr_hamilton.h / etc. flows
// to every backend simultaneously.
//
// Entry point: `main` compute shader, 8×8 threadgroups. Dispatched once
// per output pixel by the host.
//
// Bindings (all in descriptor set 0):
//   b0 — Uniforms (scene + camera + render knobs)
//   u1 — output HDR texture (RWTexture2D<float4>, rgba16f)

#include "../../../shared_shader/uniforms.h"
#include "../../../shared_shader/geodesic_math.h"
#include "../../../shared_shader/kerr_math.h"
#include "../../../shared_shader/kerr_hamilton.h"
#include "../../../shared_shader/disc_intersection.h"

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms>  u      : register(b0);
[[vk::binding(1, 0)]] RWTexture2D<float4>       output : register(u0);

// ---------------------------------------------------------------------------
// Blackbody + disc helpers (1:1 with the MSL version)
// ---------------------------------------------------------------------------

static float3 blackbody_srgb(float T) {
    T = clamp(T, 1000.0f, 40000.0f) * 0.01f;
    float r, g, b;
    if (T <= 66.0f) {
        r = 1.0f;
        g = clamp(0.39008157876901960784f * log(T) - 0.63184144378862745098f,
                  0.0f, 1.0f);
    } else {
        const float t1 = T - 60.0f;
        r = clamp(1.29293618606274509804f * pow(t1, -0.1332047592f), 0.0f, 1.0f);
        g = clamp(1.12989086089529411765f * pow(t1, -0.0755148492f), 0.0f, 1.0f);
    }
    if (T >= 66.0f) {
        b = 1.0f;
    } else if (T <= 19.0f) {
        b = 0.0f;
    } else {
        const float t1 = T - 10.0f;
        b = clamp(0.54320678911019607843f * log(t1) - 1.19625408914f,
                  0.0f, 1.0f);
    }
    return float3(r, g, b);
}

static float disc_temperature(float r_cross, float r_inner, float peak_T) {
    if (r_cross < r_inner) return 0.0f;
    const float ratio = r_inner / r_cross;
    const float q = pow(ratio, 0.75f);
    const float edge = 1.0f - sqrt(ratio);
    const float raw = q * max(edge, 0.0f);
    return peak_T * raw / 0.186f;
}

static float disc_band_mask(float r_cross, float phi_cross,
                            float r_inner, float turbulence,
                            float time_sec) {
    const float rho = log(max(r_cross / r_inner, 0.1f));
    const float shear = 4.0f / max(pow(r_cross / r_inner, 1.5f), 0.25f);
    // Animation: rotate the band pattern in +phi at the local Keplerian
    // angular velocity ω(r) ∝ r^(-3/2). Inner edge makes ~one revolution
    // every 8 seconds at the 0.8 prefactor; outer disc visibly slower.
    const float omega = pow(max(r_cross / r_inner, 0.1f), -1.5f);
    const float phi_anim = phi_cross - 0.8f * time_sec * omega;
    const float band1 = 0.5f + 0.5f * cos(10.0f * rho + 3.0f * phi_anim + shear);
    const float band2 = 0.5f + 0.5f * cos(22.0f * rho - 1.7f * phi_anim);
    const float combined = lerp(0.8f, band1 * band1 * band2, turbulence);
    return clamp(0.65f + 0.55f * combined, 0.0f, 1.3f);
}

static uint xorshift32(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}
static uint mix32(uint a, uint b) {
    return xorshift32(a * 0x9E3779B9u + b);
}

// Procedural starfield — sparse bright pinpoints on a near-black sky.
// See render/metal/shaders/geodesic_kernel.metal for the design rationale
// (this is the HLSL twin; same numbers, same result).
static float3 starfield_color(float3 dir) {
    const float len = length(dir);
    if (len < 1e-6f) return float3(0.0f, 0.0f, 0.005f);
    const float3 n = dir / len;
    const float theta = acos(clamp(n.z, -1.0f, 1.0f));
    const float phi   = atan2(n.y, n.x) + 3.14159265358979323846f;

    const int NX = 768;
    const int NY = 384;
    const int ix = int(phi / (2.0f * 3.14159265358979323846f) * float(NX)) % NX;
    const int iy = int(theta / 3.14159265358979323846f * float(NY)) % NY;

    const uint seed = mix32(uint(ix), uint(iy));
    const bool has_star = (seed & 0xFFu) < 4u;

    float r = 0.0015f;
    float g = 0.002f;
    float b = 0.005f;

    if (has_star) {
        const float bright_raw = float((seed >> 8) & 0xFFu) / 255.0f;
        const float brightness = pow(bright_raw, 4.0f);
        const float hue = float((seed >> 16) & 0xFFu) / 255.0f;
        const float star_r = 1.0f - 0.5f * hue;
        const float star_g = 0.8f + 0.2f * hue * (1.0f - hue) * 4.0f;
        const float star_b = 0.4f + 0.6f * hue;
        r = min(1.0f, r + brightness * star_r);
        g = min(1.0f, g + brightness * star_g);
        b = min(1.0f, b + brightness * star_b);
    }
    return float3(r, g, b) * 0.6f;
}

// ---------------------------------------------------------------------------
// Camera ray construction
// ---------------------------------------------------------------------------

struct CameraRay {
    float3 cart;
    float  ur;
    float  utheta;
    float  uphi;
    float  cam_r;
    float  cam_theta;
    float  cam_phi;
    float  sin_th;
};

static CameraRay build_camera_ray_subpixel(uint2 gid, float2 sub_offset) {
    const float fx = float(gid.x) + sub_offset.x;
    const float fy = float(gid.y) + sub_offset.y;
    const float u_ndc = (2.0f * fx / float(u.width)  - 1.0f) * u.aspect * u.tan_half_fov;
    const float v_ndc = -(2.0f * fy / float(u.height) - 1.0f) * u.tan_half_fov;

    const float3 fwd   = u.cam_fwd.xyz;
    const float3 right = u.cam_right.xyz;
    const float3 up    = u.cam_up.xyz;
    float3 dir = normalize(fwd + right * u_ndc + up * v_ndc);

    CameraRay r;
    r.cart = dir;
    const float3 p = u.cam_pos.xyz;
    r.cam_r = length(p);
    r.cam_theta = acos(clamp(p.z / max(r.cam_r, 1e-6f), -1.0f, 1.0f));
    r.cam_phi   = atan2(p.y, p.x);
    r.sin_th = sin(r.cam_theta);
    const float cos_th = cos(r.cam_theta);
    const float cos_phi = cos(r.cam_phi);
    const float sin_phi = sin(r.cam_phi);

    const float3 rhat = float3( r.sin_th * cos_phi,  r.sin_th * sin_phi,  cos_th);
    const float3 that = float3( cos_th   * cos_phi,  cos_th   * sin_phi, -r.sin_th);
    const float3 phat = float3(-sin_phi,             cos_phi,             0.0f);

    const float ur_ortho   = dot(dir, rhat);
    const float uth_ortho  = dot(dir, that);
    const float uphi_ortho = dot(dir, phat);

    r.ur     = ur_ortho;
    r.utheta = uth_ortho / r.cam_r;
    r.uphi   = uphi_ortho / (r.cam_r * max(r.sin_th, 1e-6f));
    return r;
}

// ---------------------------------------------------------------------------
// Schwarzschild ray trace
// ---------------------------------------------------------------------------

static float3 trace_schwarzschild(CameraRay cam) {
    State s;
    s.t     = 0.0f;
    s.r     = cam.cam_r;
    s.theta = cam.cam_theta;
    s.phi   = cam.cam_phi;
    s.ur     = cam.ur;
    s.utheta = cam.utheta;
    s.uphi   = cam.uphi;

    const float f_cam = 1.0f - u.rs / cam.cam_r;
    const float kin = (cam.ur * cam.ur) / f_cam
                    + cam.utheta * cam.utheta * cam.cam_r * cam.cam_r
                    + cam.uphi * cam.uphi * cam.cam_r * cam.cam_r * cam.sin_th * cam.sin_th;
    s.ut = sqrt(max(kin / f_cam, 0.0f));

    float prev_theta = s.theta;
    float prev_r     = s.r;
    float prev_uphi  = s.uphi;

    for (uint step = 0; step < u.max_steps; ++step) {
        if (!isfinite(s.r) || s.r < u.horizon_cut) return float3(0.0f, 0.0f, 0.0f);
        if (s.r > u.escape_r) {
            if ((u.flags & SING_FLAG_STARFIELD_ON) != 0) {
                const float sx = sin(s.theta) * cos(s.phi);
                const float sy = sin(s.theta) * sin(s.phi);
                const float sz = cos(s.theta);
                return starfield_color(float3(sx, sy, sz));
            }
            return float3(0.0f, 0.0f, 0.0f);
        }

        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, s.theta);
        if (((u.flags & SING_FLAG_DISC_ON) != 0) && xing.crossed) {
            const float r_cross    = lerp_scalar(prev_r,    s.r,    xing.frac);
            const float uphi_cross = lerp_scalar(prev_uphi, s.uphi, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                const float T_local = disc_temperature(r_cross, u.disc_r_inner,
                                                       u.disc_peak_T);
                if (T_local > 0.0f) {
                    const float f_emit = 1.0f - u.rs / r_cross;
                    const float g_grav = ((u.flags & SING_FLAG_REDSHIFT_ON) != 0)
                        ? sqrt(max(f_emit / f_cam, 0.0f)) : 1.0f;
                    float g_dop = 1.0f;
                    if ((u.flags & SING_FLAG_DOPPLER_ON) != 0) {
                        const float omega = sqrt(u.mass_M
                                                 / (r_cross * r_cross * r_cross));
                        const float v = r_cross * omega / sqrt(max(f_emit, 1e-6f));
                        const float cos_psi = clamp(-uphi_cross * r_cross, -1.0f, 1.0f);
                        g_dop = sqrt(max(1.0f - v * v, 0.0f))
                              / max(1.0f - v * cos_psi, 1e-4f);
                    }
                    const float g = g_grav * g_dop;
                    const float3 col = blackbody_srgb(g * T_local);
                    const float flux = clamp(pow(u.disc_r_inner / r_cross, 3.0f),
                                             0.05f, 1.5f);
                    const float phi_cross = s.phi;
                    const float bands = disc_band_mask(r_cross, phi_cross,
                                                       u.disc_r_inner,
                                                       u.disc_turbulence,
                                                       u.time_sec);
                    const float g2 = g * g;
                    return col * flux * bands * g2 * g2;
                }
                return float3(0.0f, 0.0f, 0.0f);
            }
        }
        prev_theta = s.theta;
        prev_r     = s.r;
        prev_uphi  = s.uphi;
        s = rk4_step(s, u.h_step, u.rs);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// Kerr ray trace via canonical-momenta Hamiltonian integrator
// ---------------------------------------------------------------------------

static float3 trace_kerr(CameraRay cam) {
    const float M = u.mass_M;
    const float a = u.spin_a;
    const float sin_th = cam.sin_th;
    const float cos_th = cos(cam.cam_theta);
    const float cam_r  = cam.cam_r;

    const float Sigma_c = cam_r * cam_r + a * a * cos_th * cos_th;
    const float Delta_c = cam_r * cam_r - 2.0f * M * cam_r + a * a;
    const float s2_c    = sin_th * sin_th;
    const float g_tt = -(1.0f - 2.0f * M * cam_r / Sigma_c);
    const float g_tp = -2.0f * M * cam_r * a * s2_c / Sigma_c;
    const float g_rr = Sigma_c / Delta_c;
    const float g_thth = Sigma_c;
    const float g_pp = s2_c * ((cam_r * cam_r + a * a) * (cam_r * cam_r + a * a)
                               - a * a * Delta_c * s2_c) / Sigma_c;
    const float A_q = g_tt;
    const float B_q = 2.0f * g_tp * cam.uphi;
    const float C_q = g_rr   * cam.ur     * cam.ur
                    + g_thth * cam.utheta * cam.utheta
                    + g_pp   * cam.uphi   * cam.uphi;
    const float disc_q = B_q * B_q - 4.0f * A_q * C_q;
    if (disc_q < 0.0f) return float3(0.0f, 0.0f, 0.0f);
    const float ut = (-B_q - sqrt(disc_q)) / (2.0f * A_q);

    KerrConserved c;
    c.a   = a;
    c.M   = M;
    c.E   = -(g_tt * ut + g_tp * cam.uphi);
    c.L_z =  g_tp * ut + g_pp * cam.uphi;
    const float p_theta_init = g_thth * cam.utheta;
    c.Q = p_theta_init * p_theta_init
          + cos_th * cos_th * (c.L_z * c.L_z / max(s2_c, 1e-6f)
                               - a * a * c.E * c.E);

    KerrHamState s;
    s.t = 0.0f;
    s.r = cam_r;
    s.theta = cam.cam_theta;
    s.phi = cam.cam_phi;
    s.p_r     = g_rr   * cam.ur;
    s.p_theta = g_thth * cam.utheta;

    float prev_theta = s.theta;
    float prev_r     = s.r;

    const float horizon_cut_kerr = 1.02f * (M + sqrt(max(M * M - a * a, 0.0f)));

    for (uint step = 0; step < u.max_steps; ++step) {
        if (!isfinite(s.r) || s.r < horizon_cut_kerr) return float3(0.0f, 0.0f, 0.0f);
        if (s.r > u.escape_r) {
            if ((u.flags & SING_FLAG_STARFIELD_ON) != 0) {
                const float sx = sin(s.theta) * cos(s.phi);
                const float sy = sin(s.theta) * sin(s.phi);
                const float sz = cos(s.theta);
                return starfield_color(float3(sx, sy, sz));
            }
            return float3(0.0f, 0.0f, 0.0f);
        }

        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, s.theta);
        if (((u.flags & SING_FLAG_DISC_ON) != 0) && xing.crossed) {
            const float r_cross = lerp_scalar(prev_r, s.r, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                const float T_local = disc_temperature(r_cross, u.disc_r_inner,
                                                       u.disc_peak_T);
                if (T_local > 0.0f) {
                    const float f_emit = 1.0f - 2.0f * M / r_cross;
                    const float f_obs  = 1.0f - 2.0f * M / cam_r;
                    const float g_grav = ((u.flags & SING_FLAG_REDSHIFT_ON) != 0)
                        ? sqrt(max(f_emit, 1e-6f) / max(f_obs, 1e-6f)) : 1.0f;
                    float g_dop = 1.0f;
                    if ((u.flags & SING_FLAG_DOPPLER_ON) != 0) {
                        const float om0 = sqrt(M / (r_cross * r_cross * r_cross));
                        const float omega_K = om0 / (1.0f + a * om0);
                        const float v = r_cross * omega_K / sqrt(max(f_emit, 1e-6f));
                        const float cos_psi = clamp(c.L_z / r_cross, -1.0f, 1.0f);
                        const float v2 = v * v;
                        g_dop = sqrt(max(1.0f - v2, 0.0f)) / max(1.0f - v * cos_psi, 1e-4f);
                    }
                    const float g = g_grav * g_dop;
                    const float3 col = blackbody_srgb(g * T_local);
                    const float flux = clamp(pow(u.disc_r_inner / r_cross, 3.0f),
                                             0.05f, 1.5f);
                    const float phi_cross = s.phi;
                    const float bands = disc_band_mask(r_cross, phi_cross,
                                                       u.disc_r_inner,
                                                       u.disc_turbulence,
                                                       u.time_sec);
                    const float g2 = g * g;
                    return col * flux * bands * g2 * g2;
                }
                return float3(0.0f, 0.0f, 0.0f);
            }
        }
        prev_theta = s.theta;
        prev_r     = s.r;
        s = kerr_ham_rk4_step(s, u.h_step, c);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

[numthreads(8, 8, 1)]
void main(uint2 gid : SV_DispatchThreadID) {
    if (gid.x >= u.width || gid.y >= u.height) return;

    const uint ss = max(1u, min(u.supersample, 4u));
    const float sub_w = 1.0f / float(ss);

    float3 acc = float3(0.0f, 0.0f, 0.0f);
    for (uint sy = 0; sy < ss; ++sy) {
        for (uint sx = 0; sx < ss; ++sx) {
            const float2 off = float2((float(sx) + 0.5f) * sub_w,
                                      (float(sy) + 0.5f) * sub_w);
            const CameraRay cam = build_camera_ray_subpixel(gid, off);
            const float3 px = (u.metric_type == SING_METRIC_KERR)
                              ? trace_kerr(cam)
                              : trace_schwarzschild(cam);
            acc += px;
        }
    }
    acc *= 1.0f / float(ss * ss);
    output[gid] = float4(acc, 1.0f);
}
