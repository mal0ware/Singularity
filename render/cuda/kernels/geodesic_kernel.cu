// render/cuda/kernels/geodesic_kernel.cu
//
// Per-pixel null-geodesic ray tracer — CUDA port of
// render/metal/shaders/geodesic_kernel.metal. Same physics, same
// shared_shader/* headers, same outputs as the Metal and Vulkan kernels.
//
// Differences vs. the Metal/Vulkan path:
//   * Output is tone-mapped + sRGB-encoded RGBA8 inline. The interactive
//     Metal/Vulkan paths render to an HDR rgba16f target and apply ACES +
//     bloom in a separate blit pass; the CUDA backend is offline-only and
//     has no blit pipeline, so a cheap ACES + sRGB packs straight into the
//     RGBA8 device buffer that capture_frame() reads back. Bloom is
//     deliberately omitted — it is a separable Gaussian on a downsampled
//     HDR target and adds two extra kernel launches; not worth it for the
//     PNG-export use case.
//   * No texture<> output binding — we just write to a global uint8_t*
//     RGBA8 buffer that lives in device memory.
//   * Uniforms are passed by value as a kernel argument (struct is < 256 B,
//     fits comfortably in the per-launch parameter region).

#include <cuda_runtime.h>

#include <cstdint>

#include "../../../shared_shader/disc_intersection.h"
#include "../../../shared_shader/geodesic_math.h"
#include "../../../shared_shader/kerr_hamilton.h"
#include "../../../shared_shader/kerr_math.h"
#include "../../../shared_shader/uniforms.h"

namespace singularity {

namespace {

// ---------------------------------------------------------------------------
// Minimal float3 helpers. CUDA's built-in float3 has no arithmetic operators
// and no length/dot/normalize — we want the kernel body to read like the MSL
// / HLSL ports, so a handful of __device__ helpers shim that gap. Kept in an
// anonymous namespace so they don't leak into other CUDA TUs and so nvcc is
// free to inline them aggressively.
// ---------------------------------------------------------------------------

__device__ inline float3 vmake(float x, float y, float z) {
    return make_float3(x, y, z);
}
__device__ inline float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 vmul(float3 a, float k) {
    return make_float3(a.x * k, a.y * k, a.z * k);
}
__device__ inline float3 vdiv(float3 a, float k) {
    const float ik = 1.0f / k;
    return vmul(a, ik);
}
__device__ inline float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float vlength(float3 a) {
    return sqrtf(vdot(a, a));
}
__device__ inline float3 vnormalize(float3 a) {
    return vdiv(a, fmaxf(vlength(a), 1e-20f));
}

__device__ inline float vclamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}
__device__ inline float vmix(float a, float b, float t) {
    return a + t * (b - a);
}

// ---------------------------------------------------------------------------
// Helpers — 1:1 with the MSL / HLSL versions
// ---------------------------------------------------------------------------

__device__ inline float3 blackbody_srgb(float T) {
    T = vclamp(T, 1000.0f, 40000.0f) * 0.01f;
    float r, g, b;
    if (T <= 66.0f) {
        r = 1.0f;
        g = vclamp(0.39008157876901960784f * logf(T) - 0.63184144378862745098f, 0.0f, 1.0f);
    } else {
        const float t1 = T - 60.0f;
        r = vclamp(1.29293618606274509804f * powf(t1, -0.1332047592f), 0.0f, 1.0f);
        g = vclamp(1.12989086089529411765f * powf(t1, -0.0755148492f), 0.0f, 1.0f);
    }
    if (T >= 66.0f) {
        b = 1.0f;
    } else if (T <= 19.0f) {
        b = 0.0f;
    } else {
        const float t1 = T - 10.0f;
        b = vclamp(0.54320678911019607843f * logf(t1) - 1.19625408914f, 0.0f, 1.0f);
    }
    return vmake(r, g, b);
}

__device__ inline float disc_temperature(float r_cross, float r_inner, float peak_T) {
    if (r_cross < r_inner)
        return 0.0f;
    const float ratio = r_inner / r_cross;
    const float q = powf(ratio, 0.75f);
    const float edge = 1.0f - sqrtf(ratio);
    const float raw = q * fmaxf(edge, 0.0f);
    return peak_T * raw / 0.186f;
}

__device__ inline float
disc_band_mask(float r_cross, float phi_cross, float r_inner, float turbulence, float time_sec) {
    const float rho = logf(fmaxf(r_cross / r_inner, 0.1f));
    const float shear = 4.0f / fmaxf(powf(r_cross / r_inner, 1.5f), 0.25f);
    // Animation: rotate the band pattern at Keplerian ω(r) ∝ r^(-3/2).
    // 0.8 prefactor → ~8 s per revolution at the inner edge.
    const float omega = powf(fmaxf(r_cross / r_inner, 0.1f), -1.5f);
    const float phi_anim = phi_cross - 0.8f * time_sec * omega;
    const float band1 = 0.5f + 0.5f * cosf(10.0f * rho + 3.0f * phi_anim + shear);
    const float band2 = 0.5f + 0.5f * cosf(22.0f * rho - 1.7f * phi_anim);
    const float combined = vmix(0.8f, band1 * band1 * band2, turbulence);
    return vclamp(0.65f + 0.55f * combined, 0.0f, 1.3f);
}

__device__ inline unsigned int xorshift32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}
__device__ inline unsigned int mix32(unsigned int a, unsigned int b) {
    return xorshift32(a * 0x9E3779B9u + b);
}

// Procedural starfield — sparse bright pinpoints on a near-black sky.
// See render/metal/shaders/geodesic_kernel.metal for the design rationale.
__device__ inline float3 starfield_color(float3 dir) {
    const float len = vlength(dir);
    if (len < 1e-6f)
        return vmake(0.0f, 0.0f, 0.005f);
    const float3 n = vdiv(dir, len);
    const float theta = acosf(vclamp(n.z, -1.0f, 1.0f));
    const float phi = atan2f(n.y, n.x) + 3.14159265358979323846f;

    const int NX = 768;
    const int NY = 384;
    int ix = int(phi / (2.0f * 3.14159265358979323846f) * float(NX)) % NX;
    int iy = int(theta / 3.14159265358979323846f * float(NY)) % NY;
    if (ix < 0)
        ix += NX;
    if (iy < 0)
        iy += NY;

    const unsigned int seed = mix32((unsigned int)ix, (unsigned int)iy);
    const bool has_star = (seed & 0xFFu) < 4u;

    float r = 0.0015f;
    float g = 0.002f;
    float b = 0.005f;

    if (has_star) {
        const float bright_raw = float((seed >> 8) & 0xFFu) / 255.0f;
        const float brightness = powf(bright_raw, 4.0f);
        const float hue = float((seed >> 16) & 0xFFu) / 255.0f;
        const float star_r = 1.0f - 0.5f * hue;
        const float star_g = 0.8f + 0.2f * hue * (1.0f - hue) * 4.0f;
        const float star_b = 0.4f + 0.6f * hue;
        r = fminf(1.0f, r + brightness * star_r);
        g = fminf(1.0f, g + brightness * star_g);
        b = fminf(1.0f, b + brightness * star_b);
    }
    return vmul(vmake(r, g, b), 0.6f);
}

// ---------------------------------------------------------------------------
// Camera ray construction (port of build_camera_ray_subpixel)
// ---------------------------------------------------------------------------

struct CameraRay {
    float3 cart;
    float ur;
    float utheta;
    float uphi;
    float cam_r;
    float cam_theta;
    float cam_phi;
    float sin_th;
};

__device__ inline CameraRay
build_camera_ray_subpixel(unsigned int gx, unsigned int gy, float ox, float oy, const Uniforms& u) {
    const float fx = float(gx) + ox;
    const float fy = float(gy) + oy;
    const float u_ndc = (2.0f * fx / float(u.width) - 1.0f) * u.aspect * u.tan_half_fov;
    const float v_ndc = -(2.0f * fy / float(u.height) - 1.0f) * u.tan_half_fov;

    const float3 fwd = vmake(u.cam_fwd.x, u.cam_fwd.y, u.cam_fwd.z);
    const float3 right = vmake(u.cam_right.x, u.cam_right.y, u.cam_right.z);
    const float3 up = vmake(u.cam_up.x, u.cam_up.y, u.cam_up.z);
    float3 dir = vnormalize(vadd(fwd, vadd(vmul(right, u_ndc), vmul(up, v_ndc))));

    CameraRay r;
    r.cart = dir;
    const float3 p = vmake(u.cam_pos.x, u.cam_pos.y, u.cam_pos.z);
    r.cam_r = vlength(p);
    r.cam_theta = acosf(vclamp(p.z / fmaxf(r.cam_r, 1e-6f), -1.0f, 1.0f));
    r.cam_phi = atan2f(p.y, p.x);
    r.sin_th = sinf(r.cam_theta);
    const float cos_th = cosf(r.cam_theta);
    const float cos_phi = cosf(r.cam_phi);
    const float sin_phi = sinf(r.cam_phi);

    const float3 rhat = vmake(r.sin_th * cos_phi, r.sin_th * sin_phi, cos_th);
    const float3 that = vmake(cos_th * cos_phi, cos_th * sin_phi, -r.sin_th);
    const float3 phat = vmake(-sin_phi, cos_phi, 0.0f);

    const float ur_ortho = vdot(dir, rhat);
    const float uth_ortho = vdot(dir, that);
    const float uphi_ortho = vdot(dir, phat);

    r.ur = ur_ortho;
    r.utheta = uth_ortho / r.cam_r;
    r.uphi = uphi_ortho / (r.cam_r * fmaxf(r.sin_th, 1e-6f));
    return r;
}

// ---------------------------------------------------------------------------
// Schwarzschild ray integration — 1:1 with trace_schwarzschild in the MSL kernel
// ---------------------------------------------------------------------------

__device__ inline float3 trace_schwarzschild(CameraRay cam, const Uniforms& u) {
    State s;
    s.t = 0.0f;
    s.r = cam.cam_r;
    s.theta = cam.cam_theta;
    s.phi = cam.cam_phi;
    s.ur = cam.ur;
    s.utheta = cam.utheta;
    s.uphi = cam.uphi;

    const float f_cam = 1.0f - u.rs / cam.cam_r;
    const float kin = (cam.ur * cam.ur) / f_cam + cam.utheta * cam.utheta * cam.cam_r * cam.cam_r
                      + cam.uphi * cam.uphi * cam.cam_r * cam.cam_r * cam.sin_th * cam.sin_th;
    s.ut = sqrtf(fmaxf(kin / f_cam, 0.0f));

    float prev_theta = s.theta;
    float prev_r = s.r;
    float prev_uphi = s.uphi;

    for (unsigned int step = 0; step < u.max_steps; ++step) {
        if (!isfinite(s.r) || s.r < u.horizon_cut)
            return vmake(0.0f, 0.0f, 0.0f);
        if (s.r > u.escape_r) {
            if (u.flags & SING_FLAG_STARFIELD_ON) {
                const float sx = sinf(s.theta) * cosf(s.phi);
                const float sy = sinf(s.theta) * sinf(s.phi);
                const float sz = cosf(s.theta);
                return starfield_color(vmake(sx, sy, sz));
            }
            return vmake(0.0f, 0.0f, 0.0f);
        }

        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, s.theta);
        if ((u.flags & SING_FLAG_DISC_ON) && xing.crossed) {
            const float r_cross = lerp_scalar(prev_r, s.r, xing.frac);
            const float uphi_cross = lerp_scalar(prev_uphi, s.uphi, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                const float T_local = disc_temperature(r_cross, u.disc_r_inner, u.disc_peak_T);
                if (T_local > 0.0f) {
                    const float f_emit = 1.0f - u.rs / r_cross;
                    const float g_grav = (u.flags & SING_FLAG_REDSHIFT_ON)
                                             ? sqrtf(fmaxf(f_emit / f_cam, 0.0f))
                                             : 1.0f;
                    float g_dop = 1.0f;
                    if (u.flags & SING_FLAG_DOPPLER_ON) {
                        const float omega = sqrtf(u.mass_M / (r_cross * r_cross * r_cross));
                        const float v = r_cross * omega / sqrtf(fmaxf(f_emit, 1e-6f));
                        const float cos_psi = vclamp(-uphi_cross * r_cross, -1.0f, 1.0f);
                        g_dop = sqrtf(fmaxf(1.0f - v * v, 0.0f)) / fmaxf(1.0f - v * cos_psi, 1e-4f);
                    }
                    const float g = g_grav * g_dop;
                    const float3 col = blackbody_srgb(g * T_local);
                    const float flux = vclamp(powf(u.disc_r_inner / r_cross, 3.0f), 0.05f, 1.5f);
                    const float phi_cross = lerp_scalar(s.phi, s.phi, xing.frac);
                    const float bands = disc_band_mask(
                        r_cross, phi_cross, u.disc_r_inner, u.disc_turbulence, u.time_sec);
                    const float g2 = g * g;
                    return vmul(col, flux * bands * g2 * g2);
                }
                return vmake(0.0f, 0.0f, 0.0f);
            }
        }
        prev_theta = s.theta;
        prev_r = s.r;
        prev_uphi = s.uphi;
        s = rk4_step(s, u.h_step, u.rs);
    }
    return vmake(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// Kerr ray integration — 1:1 with trace_kerr in the MSL kernel
// ---------------------------------------------------------------------------

__device__ inline float3 trace_kerr(CameraRay cam, const Uniforms& u) {
    const float M = u.mass_M;
    const float a = u.spin_a;
    const float sin_th = cam.sin_th;
    const float cos_th = cosf(cam.cam_theta);
    const float cam_r = cam.cam_r;

    const float Sigma_c = cam_r * cam_r + a * a * cos_th * cos_th;
    const float Delta_c = cam_r * cam_r - 2.0f * M * cam_r + a * a;
    const float s2_c = sin_th * sin_th;
    const float g_tt = -(1.0f - 2.0f * M * cam_r / Sigma_c);
    const float g_tp = -2.0f * M * cam_r * a * s2_c / Sigma_c;
    const float g_rr = Sigma_c / Delta_c;
    const float g_thth = Sigma_c;
    const float g_pp =
        s2_c * ((cam_r * cam_r + a * a) * (cam_r * cam_r + a * a) - a * a * Delta_c * s2_c)
        / Sigma_c;
    const float A_q = g_tt;
    const float B_q = 2.0f * g_tp * cam.uphi;
    const float C_q =
        g_rr * cam.ur * cam.ur + g_thth * cam.utheta * cam.utheta + g_pp * cam.uphi * cam.uphi;
    const float disc_q = B_q * B_q - 4.0f * A_q * C_q;
    if (disc_q < 0.0f)
        return vmake(0.0f, 0.0f, 0.0f);
    const float ut = (-B_q - sqrtf(disc_q)) / (2.0f * A_q);

    KerrConserved c;
    c.a = a;
    c.M = M;
    c.E = -(g_tt * ut + g_tp * cam.uphi);
    c.L_z = g_tp * ut + g_pp * cam.uphi;
    const float p_theta_init = g_thth * cam.utheta;
    c.Q = p_theta_init * p_theta_init
          + cos_th * cos_th * (c.L_z * c.L_z / fmaxf(s2_c, 1e-6f) - a * a * c.E * c.E);

    KerrHamState s;
    s.t = 0.0f;
    s.r = cam_r;
    s.theta = cam.cam_theta;
    s.phi = cam.cam_phi;
    s.p_r = g_rr * cam.ur;
    s.p_theta = g_thth * cam.utheta;

    float prev_theta = s.theta;
    float prev_r = s.r;

    const float horizon_cut_kerr = 1.02f * (M + sqrtf(fmaxf(M * M - a * a, 0.0f)));

    for (unsigned int step = 0; step < u.max_steps; ++step) {
        if (!isfinite(s.r) || s.r < horizon_cut_kerr)
            return vmake(0.0f, 0.0f, 0.0f);
        if (s.r > u.escape_r) {
            if (u.flags & SING_FLAG_STARFIELD_ON) {
                const float sx = sinf(s.theta) * cosf(s.phi);
                const float sy = sinf(s.theta) * sinf(s.phi);
                const float sz = cosf(s.theta);
                return starfield_color(vmake(sx, sy, sz));
            }
            return vmake(0.0f, 0.0f, 0.0f);
        }

        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, s.theta);
        if ((u.flags & SING_FLAG_DISC_ON) && xing.crossed) {
            const float r_cross = lerp_scalar(prev_r, s.r, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                const float T_local = disc_temperature(r_cross, u.disc_r_inner, u.disc_peak_T);
                if (T_local > 0.0f) {
                    const float f_emit = 1.0f - 2.0f * M / r_cross;
                    const float f_obs = 1.0f - 2.0f * M / cam_r;
                    const float g_grav = (u.flags & SING_FLAG_REDSHIFT_ON)
                                             ? sqrtf(fmaxf(f_emit, 1e-6f) / fmaxf(f_obs, 1e-6f))
                                             : 1.0f;

                    float g_dop = 1.0f;
                    if (u.flags & SING_FLAG_DOPPLER_ON) {
                        const float om0 = sqrtf(M / (r_cross * r_cross * r_cross));
                        const float omega_K = om0 / (1.0f + a * om0);
                        const float v = r_cross * omega_K / sqrtf(fmaxf(f_emit, 1e-6f));
                        const float cos_psi = vclamp(c.L_z / r_cross, -1.0f, 1.0f);
                        const float v2 = v * v;
                        g_dop = sqrtf(fmaxf(1.0f - v2, 0.0f)) / fmaxf(1.0f - v * cos_psi, 1e-4f);
                    }

                    const float g = g_grav * g_dop;
                    const float3 col = blackbody_srgb(g * T_local);
                    const float flux = vclamp(powf(u.disc_r_inner / r_cross, 3.0f), 0.05f, 1.5f);
                    const float phi_cross = s.phi;
                    const float bands = disc_band_mask(
                        r_cross, phi_cross, u.disc_r_inner, u.disc_turbulence, u.time_sec);
                    const float g2 = g * g;
                    return vmul(col, flux * bands * g2 * g2);
                }
                return vmake(0.0f, 0.0f, 0.0f);
            }
        }
        prev_theta = s.theta;
        prev_r = s.r;
        s = kerr_ham_rk4_step(s, u.h_step, c);
    }
    return vmake(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// HDR → display: ACES tone-map approximation + sRGB encode + RGBA8 pack.
// The Metal/Vulkan paths do this in a separate fragment-shader blit; the
// CUDA backend has no blit pass, so we collapse it inline. The ACES form
// here is the Krzysztof Narkowicz polynomial fit (one of many compact
// approximations); error vs. the full Hill curve is < 1% perceptually.
// ---------------------------------------------------------------------------

__device__ inline float3 aces(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return vmake(vclamp((x.x * (a * x.x + b)) / (x.x * (c * x.x + d) + e), 0.0f, 1.0f),
                 vclamp((x.y * (a * x.y + b)) / (x.y * (c * x.y + d) + e), 0.0f, 1.0f),
                 vclamp((x.z * (a * x.z + b)) / (x.z * (c * x.z + d) + e), 0.0f, 1.0f));
}

__device__ inline float linear_to_srgb(float c) {
    c = vclamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__device__ inline std::uint8_t to_u8(float c) {
    return (std::uint8_t)vclamp(linear_to_srgb(c) * 255.0f + 0.5f, 0.0f, 255.0f);
}

// Halton low-discrepancy sequence — radical inverse of `i` in `base`. The
// (2, 3) pair gives the canonical 2D Halton sequence used here for subpixel
// jitter; both bases are coprime, the sequence stratifies the unit square
// uniformly at any sample count, and 256 samples already give visible-noise
// reduction that no N×N regular grid can match without aliasing artefacts.
//
// Offline-only: Metal/Vulkan kernels keep their N×N grid (max 16 SPP) so
// their per-frame budget stays bounded. The CUDA backend trades latency
// for sample count — a 4K render at 256 SPP is the headline use case.
__device__ inline float halton(unsigned int i, unsigned int base) {
    float f = 1.0f;
    float r = 0.0f;
    while (i > 0u) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public kernel entry. Launched from CudaBackend::render_frame with a 16x16
// block grid; one thread per pixel. The Uniforms struct is small (< 256 B)
// so passing by value is cheaper than allocating a __constant__ symbol per
// frame and copying into it.
// ---------------------------------------------------------------------------

__global__ void singularity_geodesic_kernel(std::uint8_t* out, Uniforms u) {
    const unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= u.width || gy >= u.height)
        return;

    // u.supersample is reinterpreted on the CUDA path as direct SPP count
    // (1..1024) rather than the N×N grid size the realtime backends use.
    // Capped at 1024 to keep wall-clock bounded — at 4K that's still under
    // a minute on Ampere; a 256 SPP render is the documented headline use.
    const unsigned int spp = max(1u, min(u.supersample, 1024u));

    // Tone-map each sample then average the LDR results, rather than
    // averaging HDR and tone-mapping once at the end. Linear-HDR averaging
    // is closer to physically-correct exposure metering, but it dilutes
    // the bright disc (HDR peaks ~50-100 from the g⁴ factor) with
    // adjacent starfield/shadow samples (~0.01) and the compressed
    // average loses the cinematic punch. Per-sample tone-mapping keeps
    // the realtime-backend look while still smoothing edge aliasing —
    // standard pattern in Mitsuba / Blender Cycles for filmic output.
    float3 ldr = vmake(0.0f, 0.0f, 0.0f);
    if (spp == 1u) {
        // SPP=1 lands at pixel center so the cross-backend equivalence
        // test (CUDA-vs-Vulkan-golden) keeps comparing the same sub-sample
        // location both backends would have used.
        const CameraRay cam = build_camera_ray_subpixel(gx, gy, 0.5f, 0.5f, u);
        const float3 hdr =
            (u.metric_type == SING_METRIC_KERR) ? trace_kerr(cam, u) : trace_schwarzschild(cam, u);
        ldr = aces(vmul(hdr, u.exposure));
    } else {
        for (unsigned int s = 0; s < spp; ++s) {
            // Skip Halton(0) — both bases give 0 there, which would put a
            // sample exactly on the pixel's top-left corner. Start at 1.
            const float ox = halton(s + 1u, 2u);
            const float oy = halton(s + 1u, 3u);
            const CameraRay cam = build_camera_ray_subpixel(gx, gy, ox, oy, u);
            const float3 hdr = (u.metric_type == SING_METRIC_KERR) ? trace_kerr(cam, u)
                                                                   : trace_schwarzschild(cam, u);
            ldr = vadd(ldr, aces(vmul(hdr, u.exposure)));
        }
        ldr = vmul(ldr, 1.0f / float(spp));
    }

    std::uint8_t* p = out + 4u * (gy * u.width + gx);
    p[0] = to_u8(ldr.x);
    p[1] = to_u8(ldr.y);
    p[2] = to_u8(ldr.z);
    p[3] = 255;
}

}  // namespace singularity
