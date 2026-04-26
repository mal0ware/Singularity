// render/webgpu/shaders/geodesic_kernel.wgsl
//
// WGSL port of render/vulkan/shaders/geodesic_kernel.hlsl. WGSL has no
// preprocessor, so the math from shared_shader/{geodesic_math, kerr_math,
// kerr_hamilton, disc_intersection, uniforms}.h is inlined here. Drift
// with the HLSL/MSL copies is the risk we accept — see docs/PHASE7_WEBGPU.md
// §"The shared-physics-headers problem" for the mitigation plan.
//
// Entry: @compute @workgroup_size(8, 8, 1), one invocation per output
// texel. Bindings match render/vulkan/vulkan_backend.cpp build_descriptor_
// layouts() so the cross-platform binding table in docs/PHASE7_WEBGPU.md
// stays 1:1.

// ---------------------------------------------------------------------------
// Uniforms (mirrors shared_shader/uniforms.h; every vec3 backed by vec4)
// ---------------------------------------------------------------------------

struct Uniforms {
    cam_pos: vec4f,
    cam_fwd: vec4f,
    cam_right: vec4f,
    cam_up: vec4f,

    mass_M: f32,
    spin_a: f32,
    rs: f32,
    tan_half_fov: f32,

    disc_r_inner: f32,
    disc_r_outer: f32,
    disc_peak_T: f32,
    aspect: f32,

    h_step: f32,
    escape_r: f32,
    horizon_cut: f32,
    time_sec: f32,

    width: u32,
    height: u32,
    metric_type: u32,
    flags: u32,

    max_steps: u32,
    frame_index: u32,
    supersample: u32,
    pad_b: u32,

    exposure: f32,
    bloom_threshold: f32,
    bloom_strength: f32,
    disc_turbulence: f32,
};

const SING_FLAG_DOPPLER_ON: u32 = 1u;
const SING_FLAG_REDSHIFT_ON: u32 = 2u;
const SING_FLAG_DISC_ON: u32 = 4u;
const SING_FLAG_STARFIELD_ON: u32 = 8u;

const SING_METRIC_SCHWARZSCHILD: u32 = 0u;
const SING_METRIC_KERR: u32 = 1u;

const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

// ---------------------------------------------------------------------------
// Schwarzschild geodesic state + RK4 (from geodesic_math.h)
// ---------------------------------------------------------------------------

struct State {
    t: f32, r: f32, theta: f32, phi: f32,
    ut: f32, ur: f32, utheta: f32, uphi: f32,
};

fn state_add(a: State, b: State) -> State {
    return State(a.t + b.t, a.r + b.r, a.theta + b.theta, a.phi + b.phi,
                 a.ut + b.ut, a.ur + b.ur, a.utheta + b.utheta, a.uphi + b.uphi);
}

fn state_scale(s: State, k: f32) -> State {
    return State(s.t * k, s.r * k, s.theta * k, s.phi * k,
                 s.ut * k, s.ur * k, s.utheta * k, s.uphi * k);
}

fn geodesic_rhs_schwarzschild(s: State, rs_val: f32) -> State {
    let f = 1.0 - rs_val / s.r;
    let r2 = s.r * s.r;
    let sin_t = sin(s.theta);
    let cos_t = cos(s.theta);

    var d: State;
    d.t = s.ut;
    d.r = s.ur;
    d.theta = s.utheta;
    d.phi = s.uphi;
    d.ut = -(rs_val / (r2 * f)) * s.ut * s.ur;
    d.ur = -(rs_val * f / (2.0 * r2)) * s.ut * s.ut
           + (rs_val / (2.0 * r2 * f)) * s.ur * s.ur
           + s.r * f * (s.utheta * s.utheta + sin_t * sin_t * s.uphi * s.uphi);
    d.utheta = -2.0 * s.ur * s.utheta / s.r + sin_t * cos_t * s.uphi * s.uphi;
    d.uphi = -2.0 * s.ur * s.uphi / s.r - 2.0 * (cos_t / sin_t) * s.utheta * s.uphi;
    return d;
}

fn rk4_step(y: State, h: f32, rs_val: f32) -> State {
    let k1 = geodesic_rhs_schwarzschild(y, rs_val);
    let k2 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k1, 0.5 * h)), rs_val);
    let k3 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k2, 0.5 * h)), rs_val);
    let k4 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k3, h)), rs_val);
    let sum = state_add(state_add(k1, state_scale(k2, 2.0)),
                        state_add(state_scale(k3, 2.0), k4));
    return state_add(y, state_scale(sum, h / 6.0));
}

// ---------------------------------------------------------------------------
// Kerr Hamiltonian integrator (from kerr_math.h + kerr_hamilton.h)
// ---------------------------------------------------------------------------

struct KerrConserved {
    E: f32,
    L_z: f32,
    Q: f32,
    a: f32,
    M: f32,
};

struct KerrHamState {
    t: f32, r: f32, theta: f32, phi: f32,
    p_r: f32, p_theta: f32,
};

fn kerr_ham_state_add(a: KerrHamState, b: KerrHamState) -> KerrHamState {
    return KerrHamState(a.t + b.t, a.r + b.r, a.theta + b.theta, a.phi + b.phi,
                        a.p_r + b.p_r, a.p_theta + b.p_theta);
}

fn kerr_ham_state_scale(s: KerrHamState, k: f32) -> KerrHamState {
    return KerrHamState(s.t * k, s.r * k, s.theta * k, s.phi * k,
                        s.p_r * k, s.p_theta * k);
}

fn kerr_ham_two_H(r: f32, theta: f32, p_r: f32, p_theta: f32, c: KerrConserved) -> f32 {
    let st = sin(theta);
    let s2 = st * st;
    var safe_s2 = s2;
    if (s2 < 1e-10) { safe_s2 = 1e-10; }
    let ct = cos(theta);
    let c2 = ct * ct;
    let r2 = r * r;
    let a2 = c.a * c.a;
    let Sigma = r2 + a2 * c2;
    let Delta = r2 - 2.0 * c.M * r + a2;
    let A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
    let SD = Sigma * Delta;

    let t_tt = -A * c.E * c.E / SD;
    let t_tp = 4.0 * c.M * r * c.a * c.E * c.L_z / SD;
    let t_pp = (Delta - a2 * safe_s2) * c.L_z * c.L_z / (SD * safe_s2);
    let t_rr = Delta * p_r * p_r / Sigma;
    let t_th = p_theta * p_theta / Sigma;
    return t_tt + t_tp + t_pp + t_rr + t_th;
}

fn kerr_ham_rhs(s: KerrHamState, c: KerrConserved) -> KerrHamState {
    let r = s.r;
    let theta = s.theta;
    let st = sin(theta);
    let s2 = st * st;
    var safe_s2 = s2;
    if (s2 < 1e-10) { safe_s2 = 1e-10; }
    let ct = cos(theta);
    let c2 = ct * ct;
    let r2 = r * r;
    let a2 = c.a * c.a;
    let Sigma = r2 + a2 * c2;
    let Delta = r2 - 2.0 * c.M * r + a2;
    let A = (r2 + a2) * (r2 + a2) - a2 * Delta * s2;
    let SD = Sigma * Delta;

    var d: KerrHamState;
    d.t = (A * c.E - 2.0 * c.M * r * c.a * c.L_z) / SD;
    d.phi = (2.0 * c.M * r * c.a * c.E * s2 + (Delta - a2 * s2) * c.L_z) / (SD * safe_s2);
    d.r = Delta * s.p_r / Sigma;
    d.theta = s.p_theta / Sigma;

    let h_diff = 1.0e-3;
    let dH_dr = (kerr_ham_two_H(r + h_diff, theta, s.p_r, s.p_theta, c)
                 - kerr_ham_two_H(r - h_diff, theta, s.p_r, s.p_theta, c))
                * (0.5 / h_diff);
    let dH_dtheta = (kerr_ham_two_H(r, theta + h_diff, s.p_r, s.p_theta, c)
                     - kerr_ham_two_H(r, theta - h_diff, s.p_r, s.p_theta, c))
                    * (0.5 / h_diff);
    d.p_r = -0.5 * dH_dr;
    d.p_theta = -0.5 * dH_dtheta;
    return d;
}

fn kerr_ham_rk4_step(y: KerrHamState, h: f32, c: KerrConserved) -> KerrHamState {
    let k1 = kerr_ham_rhs(y, c);
    let k2 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k1, 0.5 * h)), c);
    let k3 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k2, 0.5 * h)), c);
    let k4 = kerr_ham_rhs(kerr_ham_state_add(y, kerr_ham_state_scale(k3, h)), c);
    let sum = kerr_ham_state_add(
        kerr_ham_state_add(k1, kerr_ham_state_scale(k2, 2.0)),
        kerr_ham_state_add(kerr_ham_state_scale(k3, 2.0), k4));
    return kerr_ham_state_add(y, kerr_ham_state_scale(sum, h / 6.0));
}

// ---------------------------------------------------------------------------
// Disc intersection (from disc_intersection.h)
// ---------------------------------------------------------------------------

struct DiscCrossing {
    crossed: bool,
    frac: f32,
};

fn detect_equatorial_crossing(theta_prev: f32, theta_curr: f32) -> DiscCrossing {
    let mid = 1.5707963267948966;
    let dp = theta_prev - mid;
    let dc = theta_curr - mid;
    var result: DiscCrossing;
    result.crossed = (dp * dc) < 0.0;
    let denom = theta_curr - theta_prev;
    if (result.crossed) {
        result.frac = (mid - theta_prev) / denom;
    } else {
        result.frac = 0.0;
    }
    return result;
}

fn lerp_scalar(prev: f32, curr: f32, frac: f32) -> f32 {
    return prev + frac * (curr - prev);
}

fn in_disc_annulus(r_cross: f32, r_inner: f32, r_outer: f32) -> bool {
    return (r_cross >= r_inner) && (r_cross <= r_outer);
}

// ---------------------------------------------------------------------------
// Blackbody / disc / starfield helpers
// ---------------------------------------------------------------------------

fn blackbody_srgb(T_in: f32) -> vec3f {
    var T = clamp(T_in, 1000.0, 40000.0) * 0.01;
    var r: f32;
    var g: f32;
    var b: f32;
    if (T <= 66.0) {
        r = 1.0;
        g = clamp(0.39008157876901960784 * log(T) - 0.63184144378862745098, 0.0, 1.0);
    } else {
        let t1 = T - 60.0;
        r = clamp(1.29293618606274509804 * pow(t1, -0.1332047592), 0.0, 1.0);
        g = clamp(1.12989086089529411765 * pow(t1, -0.0755148492), 0.0, 1.0);
    }
    if (T >= 66.0) {
        b = 1.0;
    } else if (T <= 19.0) {
        b = 0.0;
    } else {
        let t1 = T - 10.0;
        b = clamp(0.54320678911019607843 * log(t1) - 1.19625408914, 0.0, 1.0);
    }
    return vec3f(r, g, b);
}

fn disc_temperature(r_cross: f32, r_inner: f32, peak_T: f32) -> f32 {
    if (r_cross < r_inner) { return 0.0; }
    let ratio = r_inner / r_cross;
    let q = pow(ratio, 0.75);
    let edge = 1.0 - sqrt(ratio);
    let raw = q * max(edge, 0.0);
    return peak_T * raw / 0.186;
}

fn disc_band_mask(r_cross: f32, phi_cross: f32, r_inner: f32, turbulence: f32, time_sec: f32) -> f32 {
    let rho = log(max(r_cross / r_inner, 0.1));
    let shear = 4.0 / max(pow(r_cross / r_inner, 1.5), 0.25);
    // Animation: rotate the band pattern at Keplerian ω(r) ∝ r^(-3/2).
    // 0.8 prefactor → ~8 s per revolution at the inner edge.
    let omega = pow(max(r_cross / r_inner, 0.1), -1.5);
    let phi_anim = phi_cross - 0.8 * time_sec * omega;
    let band1 = 0.5 + 0.5 * cos(10.0 * rho + 3.0 * phi_anim + shear);
    let band2 = 0.5 + 0.5 * cos(22.0 * rho - 1.7 * phi_anim);
    let combined = mix(0.8, band1 * band1 * band2, turbulence);
    return clamp(0.65 + 0.55 * combined, 0.0, 1.3);
}

fn xorshift32(x_in: u32) -> u32 {
    var x = x_in;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    return x;
}

fn mix32(a: u32, b: u32) -> u32 {
    return xorshift32(a * 0x9E3779B9u + b);
}

// Procedural starfield — sparse bright pinpoints on a near-black sky.
// See render/metal/shaders/geodesic_kernel.metal for the design rationale.
fn starfield_color(dir: vec3f) -> vec3f {
    let len_v = length(dir);
    if (len_v < 1e-6) { return vec3f(0.0, 0.0, 0.005); }
    let n = dir / len_v;
    let theta = acos(clamp(n.z, -1.0, 1.0));
    let phi = atan2(n.y, n.x) + PI;

    let NX: i32 = 768;
    let NY: i32 = 384;
    let ix_raw = i32(phi / (2.0 * PI) * f32(NX));
    let iy_raw = i32(theta / PI * f32(NY));
    let ix = ((ix_raw % NX) + NX) % NX;
    let iy = ((iy_raw % NY) + NY) % NY;

    let seed = mix32(u32(ix), u32(iy));
    let has_star = (seed & 0xFFu) < 4u;

    var r = 0.0015;
    var g = 0.002;
    var b = 0.005;

    if (has_star) {
        let bright_raw = f32((seed >> 8u) & 0xFFu) / 255.0;
        let brightness = pow(bright_raw, 4.0);
        let hue = f32((seed >> 16u) & 0xFFu) / 255.0;
        let star_r = 1.0 - 0.5 * hue;
        let star_g = 0.8 + 0.2 * hue * (1.0 - hue) * 4.0;
        let star_b = 0.4 + 0.6 * hue;
        r = min(1.0, r + brightness * star_r);
        g = min(1.0, g + brightness * star_g);
        b = min(1.0, b + brightness * star_b);
    }
    return vec3f(r, g, b) * 0.6;
}

// ---------------------------------------------------------------------------
// Camera ray
// ---------------------------------------------------------------------------

struct CameraRay {
    cart: vec3f,
    ur: f32,
    utheta: f32,
    uphi: f32,
    cam_r: f32,
    cam_theta: f32,
    cam_phi: f32,
    sin_th: f32,
};

fn build_camera_ray_subpixel(gid: vec2u, sub_offset: vec2f) -> CameraRay {
    let fx = f32(gid.x) + sub_offset.x;
    let fy = f32(gid.y) + sub_offset.y;
    let u_ndc = (2.0 * fx / f32(u.width) - 1.0) * u.aspect * u.tan_half_fov;
    let v_ndc = -(2.0 * fy / f32(u.height) - 1.0) * u.tan_half_fov;

    let fwd = u.cam_fwd.xyz;
    let right = u.cam_right.xyz;
    let up = u.cam_up.xyz;
    let dir = normalize(fwd + right * u_ndc + up * v_ndc);

    var r: CameraRay;
    r.cart = dir;
    let p = u.cam_pos.xyz;
    r.cam_r = length(p);
    r.cam_theta = acos(clamp(p.z / max(r.cam_r, 1e-6), -1.0, 1.0));
    r.cam_phi = atan2(p.y, p.x);
    r.sin_th = sin(r.cam_theta);
    let cos_th = cos(r.cam_theta);
    let cos_phi = cos(r.cam_phi);
    let sin_phi = sin(r.cam_phi);

    let rhat = vec3f(r.sin_th * cos_phi, r.sin_th * sin_phi, cos_th);
    let that = vec3f(cos_th * cos_phi, cos_th * sin_phi, -r.sin_th);
    let phat = vec3f(-sin_phi, cos_phi, 0.0);

    let ur_ortho = dot(dir, rhat);
    let uth_ortho = dot(dir, that);
    let uphi_ortho = dot(dir, phat);

    r.ur = ur_ortho;
    r.utheta = uth_ortho / r.cam_r;
    r.uphi = uphi_ortho / (r.cam_r * max(r.sin_th, 1e-6));
    return r;
}

// ---------------------------------------------------------------------------
// Schwarzschild trace
// ---------------------------------------------------------------------------

fn trace_schwarzschild(cam: CameraRay) -> vec3f {
    var s: State;
    s.t = 0.0;
    s.r = cam.cam_r;
    s.theta = cam.cam_theta;
    s.phi = cam.cam_phi;
    s.ur = cam.ur;
    s.utheta = cam.utheta;
    s.uphi = cam.uphi;

    let f_cam = 1.0 - u.rs / cam.cam_r;
    let kin = (cam.ur * cam.ur) / f_cam
              + cam.utheta * cam.utheta * cam.cam_r * cam.cam_r
              + cam.uphi * cam.uphi * cam.cam_r * cam.cam_r * cam.sin_th * cam.sin_th;
    s.ut = sqrt(max(kin / f_cam, 0.0));

    var prev_theta = s.theta;
    var prev_r = s.r;
    var prev_uphi = s.uphi;

    for (var step: u32 = 0u; step < u.max_steps; step = step + 1u) {
        // WGSL has no isfinite; NaN-propagation protects against the
        // r < horizon_cut test so a singular step bails the ray.
        if (s.r < u.horizon_cut) { return vec3f(0.0); }
        if (s.r > u.escape_r) {
            if ((u.flags & SING_FLAG_STARFIELD_ON) != 0u) {
                let sx = sin(s.theta) * cos(s.phi);
                let sy = sin(s.theta) * sin(s.phi);
                let sz = cos(s.theta);
                return starfield_color(vec3f(sx, sy, sz));
            }
            return vec3f(0.0);
        }

        let xing = detect_equatorial_crossing(prev_theta, s.theta);
        if (((u.flags & SING_FLAG_DISC_ON) != 0u) && xing.crossed) {
            let r_cross = lerp_scalar(prev_r, s.r, xing.frac);
            let uphi_cross = lerp_scalar(prev_uphi, s.uphi, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                let T_local = disc_temperature(r_cross, u.disc_r_inner, u.disc_peak_T);
                if (T_local > 0.0) {
                    let f_emit = 1.0 - u.rs / r_cross;
                    var g_grav = 1.0;
                    if ((u.flags & SING_FLAG_REDSHIFT_ON) != 0u) {
                        g_grav = sqrt(max(f_emit / f_cam, 0.0));
                    }
                    var g_dop = 1.0;
                    if ((u.flags & SING_FLAG_DOPPLER_ON) != 0u) {
                        let omega = sqrt(u.mass_M / (r_cross * r_cross * r_cross));
                        let v = r_cross * omega / sqrt(max(f_emit, 1e-6));
                        let cos_psi = clamp(-uphi_cross * r_cross, -1.0, 1.0);
                        g_dop = sqrt(max(1.0 - v * v, 0.0))
                                / max(1.0 - v * cos_psi, 1e-4);
                    }
                    let g = g_grav * g_dop;
                    let col = blackbody_srgb(g * T_local);
                    let flux = clamp(pow(u.disc_r_inner / r_cross, 3.0), 0.05, 1.5);
                    let phi_cross = s.phi;
                    let bands = disc_band_mask(r_cross, phi_cross, u.disc_r_inner,
                                               u.disc_turbulence, u.time_sec);
                    let g2 = g * g;
                    return col * flux * bands * g2 * g2;
                }
                return vec3f(0.0);
            }
        }
        prev_theta = s.theta;
        prev_r = s.r;
        prev_uphi = s.uphi;
        s = rk4_step(s, u.h_step, u.rs);
    }
    return vec3f(0.0);
}

// ---------------------------------------------------------------------------
// Kerr trace
// ---------------------------------------------------------------------------

fn trace_kerr(cam: CameraRay) -> vec3f {
    let M = u.mass_M;
    let a = u.spin_a;
    let sin_th = cam.sin_th;
    let cos_th = cos(cam.cam_theta);
    let cam_r = cam.cam_r;

    let Sigma_c = cam_r * cam_r + a * a * cos_th * cos_th;
    let Delta_c = cam_r * cam_r - 2.0 * M * cam_r + a * a;
    let s2_c = sin_th * sin_th;
    let g_tt = -(1.0 - 2.0 * M * cam_r / Sigma_c);
    let g_tp = -2.0 * M * cam_r * a * s2_c / Sigma_c;
    let g_rr = Sigma_c / Delta_c;
    let g_thth = Sigma_c;
    let g_pp = s2_c * ((cam_r * cam_r + a * a) * (cam_r * cam_r + a * a)
                       - a * a * Delta_c * s2_c) / Sigma_c;
    let A_q = g_tt;
    let B_q = 2.0 * g_tp * cam.uphi;
    let C_q = g_rr * cam.ur * cam.ur
              + g_thth * cam.utheta * cam.utheta
              + g_pp * cam.uphi * cam.uphi;
    let disc_q = B_q * B_q - 4.0 * A_q * C_q;
    if (disc_q < 0.0) { return vec3f(0.0); }
    let ut = (-B_q - sqrt(disc_q)) / (2.0 * A_q);

    var c: KerrConserved;
    c.a = a;
    c.M = M;
    c.E = -(g_tt * ut + g_tp * cam.uphi);
    c.L_z = g_tp * ut + g_pp * cam.uphi;
    let p_theta_init = g_thth * cam.utheta;
    c.Q = p_theta_init * p_theta_init
          + cos_th * cos_th * (c.L_z * c.L_z / max(s2_c, 1e-6)
                               - a * a * c.E * c.E);

    var s: KerrHamState;
    s.t = 0.0;
    s.r = cam_r;
    s.theta = cam.cam_theta;
    s.phi = cam.cam_phi;
    s.p_r = g_rr * cam.ur;
    s.p_theta = g_thth * cam.utheta;

    var prev_theta = s.theta;
    var prev_r = s.r;

    let horizon_cut_kerr = 1.02 * (M + sqrt(max(M * M - a * a, 0.0)));

    for (var step: u32 = 0u; step < u.max_steps; step = step + 1u) {
        if (s.r < horizon_cut_kerr) { return vec3f(0.0); }
        if (s.r > u.escape_r) {
            if ((u.flags & SING_FLAG_STARFIELD_ON) != 0u) {
                let sx = sin(s.theta) * cos(s.phi);
                let sy = sin(s.theta) * sin(s.phi);
                let sz = cos(s.theta);
                return starfield_color(vec3f(sx, sy, sz));
            }
            return vec3f(0.0);
        }

        let xing = detect_equatorial_crossing(prev_theta, s.theta);
        if (((u.flags & SING_FLAG_DISC_ON) != 0u) && xing.crossed) {
            let r_cross = lerp_scalar(prev_r, s.r, xing.frac);
            if (in_disc_annulus(r_cross, u.disc_r_inner, u.disc_r_outer)) {
                let T_local = disc_temperature(r_cross, u.disc_r_inner, u.disc_peak_T);
                if (T_local > 0.0) {
                    let f_emit = 1.0 - 2.0 * M / r_cross;
                    let f_obs = 1.0 - 2.0 * M / cam_r;
                    var g_grav = 1.0;
                    if ((u.flags & SING_FLAG_REDSHIFT_ON) != 0u) {
                        g_grav = sqrt(max(f_emit, 1e-6) / max(f_obs, 1e-6));
                    }
                    var g_dop = 1.0;
                    if ((u.flags & SING_FLAG_DOPPLER_ON) != 0u) {
                        let om0 = sqrt(M / (r_cross * r_cross * r_cross));
                        let omega_K = om0 / (1.0 + a * om0);
                        let v = r_cross * omega_K / sqrt(max(f_emit, 1e-6));
                        let cos_psi = clamp(c.L_z / r_cross, -1.0, 1.0);
                        let v2 = v * v;
                        g_dop = sqrt(max(1.0 - v2, 0.0))
                                / max(1.0 - v * cos_psi, 1e-4);
                    }
                    let g = g_grav * g_dop;
                    let col = blackbody_srgb(g * T_local);
                    let flux = clamp(pow(u.disc_r_inner / r_cross, 3.0), 0.05, 1.5);
                    let phi_cross = s.phi;
                    let bands = disc_band_mask(r_cross, phi_cross, u.disc_r_inner,
                                               u.disc_turbulence, u.time_sec);
                    let g2 = g * g;
                    return col * flux * bands * g2 * g2;
                }
                return vec3f(0.0);
            }
        }
        prev_theta = s.theta;
        prev_r = s.r;
        s = kerr_ham_rk4_step(s, u.h_step, c);
    }
    return vec3f(0.0);
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u.width || gid.y >= u.height) { return; }

    let ss = max(1u, min(u.supersample, 4u));
    let sub_w = 1.0 / f32(ss);

    var acc = vec3f(0.0);
    for (var sy: u32 = 0u; sy < ss; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < ss; sx = sx + 1u) {
            let off = vec2f((f32(sx) + 0.5) * sub_w, (f32(sy) + 0.5) * sub_w);
            let cam = build_camera_ray_subpixel(gid.xy, off);
            var px: vec3f;
            if (u.metric_type == SING_METRIC_KERR) {
                px = trace_kerr(cam);
            } else {
                px = trace_schwarzschild(cam);
            }
            acc = acc + px;
        }
    }
    acc = acc * (1.0 / f32(ss * ss));
    textureStore(output, vec2i(gid.xy), vec4f(acc, 1.0));
}
