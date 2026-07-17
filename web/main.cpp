// web/main.cpp
//
// Browser entry + pointer-driven orbital camera. Settings panel is a DOM
// overlay (see web/shell.html); it drives the renderer via the C setters
// at the bottom of this file. backend.initialize() kicks off the async
// adapter->device chain; render_frame() no-ops until ready.

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#endif

#include "render/webgpu/webgpu_backend.hpp"

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct OrbitalCamera {
    float distance_M = 32.0f;
    float azimuth_rad = 0.7853981633974483f;  // π/4, matches desktop default.
    float elevation_rad = 0.18f;
    float fov_y_deg = 60.0f;
};

// Internal-resolution ladder the dynamic controller walks. Quantized so a
// scale change (which re-creates the HDR/bloom textures) happens at most a
// handful of times per session, not per frame.
constexpr float kScaleSteps[] = {0.4f, 0.5f, 0.65f, 0.8f, 1.0f};
constexpr int kScaleCount = 5;

struct WebApp {
    singularity::webgpu::WebGPUBackend backend;
    singularity::Scene scene{};
    singularity::CameraState camera{};
    OrbitalCamera orbit{};
    bool dragging = false;
    double drag_anchor_x = 0.0;
    double drag_anchor_y = 0.0;
    double drag_anchor_az = 0.0;
    double drag_anchor_el = 0.0;
    float t_seconds = 0.0f;
    // Auto-orbit until first canvas pointer interaction. Panel widgets hit
    // the overlay div, not the canvas, so they don't kill it.
    bool auto_orbit = true;

    // --- Performance state -------------------------------------------------
    int quality = 1;                  // 0 Draft / 1 Balanced / 2 Quality (panel)
    int resolution_mode = 0;          // 0 Auto / 1 100% / 2 80% / 3 50%
    int scale_idx = kScaleCount - 1;  // current rung on kScaleSteps
    double ema_ms = 16.7;             // frame-time EMA (ms)
    double last_now_ms = 0.0;
    int ctrl_cooldown = 0;  // frames until the controller may act again
    double last_wheel_ms = -1.0e9;
};

WebApp g_app;

// Integration presets. With the adaptive integrator on, h_step is the
// *base* step near the hole — the far field steps up to 40× coarser
// (see shared_shader/geodesic_math.h adaptive_h).
void apply_integration_quality(bool interacting) {
    struct Preset {
        float h;
        std::uint32_t steps;
    };
    // Draft / Balanced / Quality. Draft keeps enough budget that winding
    // photon-ring rays still terminate (exhausted rays render black, which
    // reads as fringing at the shadow edge during camera drags).
    constexpr Preset kPresets[3] = {{0.18f, 900u}, {0.12f, 1200u}, {0.07f, 2200u}};
    const int q = std::max(0, std::min(2, g_app.quality));
    const Preset p = interacting ? kPresets[0] : kPresets[q];
    g_app.scene.h_step = p.h;
    g_app.scene.max_steps = p.steps;
}

// Dynamic-resolution controller: walk kScaleSteps to hold ~60 FPS.
// Hysteresis via asymmetric thresholds + a cooldown so a single slow frame
// (GC pause, tab wake) doesn't thrash texture re-creation.
void update_resolution_controller(double now_ms) {
    if (g_app.last_now_ms > 0.0) {
        double dt = now_ms - g_app.last_now_ms;
        if (dt < 0.0)
            dt = 0.0;
        if (dt > 100.0)
            dt = 100.0;
        g_app.ema_ms = 0.9 * g_app.ema_ms + 0.1 * dt;
    }
    g_app.last_now_ms = now_ms;

    int target_idx = g_app.scale_idx;
    if (g_app.resolution_mode != 0) {
        // Fixed modes: 100% / 80% / 50% map onto ladder rungs 4 / 3 / 1.
        target_idx = (g_app.resolution_mode == 1)   ? kScaleCount - 1
                     : (g_app.resolution_mode == 2) ? 3
                                                    : 1;
    } else if (g_app.ctrl_cooldown > 0) {
        --g_app.ctrl_cooldown;
    } else if (g_app.ema_ms > 20.0 && g_app.scale_idx > 0) {
        target_idx = g_app.scale_idx - 1;
        g_app.ctrl_cooldown = 45;
    } else if (g_app.ema_ms < 11.5 && g_app.scale_idx < kScaleCount - 1) {
        target_idx = g_app.scale_idx + 1;
        g_app.ctrl_cooldown = 90;
    }

    if (target_idx != g_app.scale_idx) {
        g_app.scale_idx = target_idx;
        g_app.backend.set_internal_scale(kScaleSteps[g_app.scale_idx]);
    }
}

// Build CameraState from (distance, azimuth, elevation). Identical
// convention to app_shell.cpp's compute_basis — row 0 = right,
// row 1 = up, row 2 = -forward.
void write_camera_from_orbit() {
    const auto& o = g_app.orbit;
    const float ce = std::cos(o.elevation_rad);
    const float se = std::sin(o.elevation_rad);
    const float ca = std::cos(o.azimuth_rad);
    const float sa = std::sin(o.azimuth_rad);

    g_app.camera.position[0] = o.distance_M * ce * ca;
    g_app.camera.position[1] = o.distance_M * ce * sa;
    g_app.camera.position[2] = o.distance_M * se;

    const float inv = 1.0f / o.distance_M;
    const float fx = -g_app.camera.position[0] * inv;
    const float fy = -g_app.camera.position[1] * inv;
    const float fz = -g_app.camera.position[2] * inv;

    float rx = fy;
    float ry = -fx;
    float rz = 0.0f;
    const float rl = std::sqrt((rx * rx) + (ry * ry) + (rz * rz));
    const float rinv = (rl > 1e-6f) ? (1.0f / rl) : 1.0f;
    rx *= rinv;
    ry *= rinv;
    rz *= rinv;

    const float ux = (fy * rz) - (fz * ry);
    const float uy = (fz * rx) - (fx * rz);
    const float uz = (fx * ry) - (fy * rx);

    g_app.camera.basis[0] = rx;
    g_app.camera.basis[1] = ry;
    g_app.camera.basis[2] = rz;
    g_app.camera.basis[3] = ux;
    g_app.camera.basis[4] = uy;
    g_app.camera.basis[5] = uz;
    g_app.camera.basis[6] = -fx;
    g_app.camera.basis[7] = -fy;
    g_app.camera.basis[8] = -fz;

    g_app.camera.fov_y_radians = o.fov_y_deg * (kPi / 180.0f);
}

void tick() {
    if (g_app.auto_orbit) {
        g_app.t_seconds += 1.0f / 60.0f;
        g_app.orbit.azimuth_rad = 0.2f * g_app.t_seconds;
    }

#ifdef __EMSCRIPTEN__
    const double now_ms = emscripten_get_now();
#else
    const double now_ms = 0.0;
#endif
    // Draft integration while the pointer is actively steering (drag, or
    // within 300 ms of a wheel event); the panel's preset otherwise.
    const bool interacting = g_app.dragging || (now_ms - g_app.last_wheel_ms) < 300.0;
    apply_integration_quality(interacting);
    update_resolution_controller(now_ms);

    write_camera_from_orbit();
    g_app.backend.render_frame(g_app.scene, g_app.camera);
}

#ifdef __EMSCRIPTEN__

EM_BOOL on_mouse_down(int /*event_type*/, const EmscriptenMouseEvent* e, void* /*ud*/) {
    if (e->button != 0) {
        return EM_FALSE;
    }
    g_app.auto_orbit = false;
    g_app.dragging = true;
    g_app.drag_anchor_x = e->clientX;
    g_app.drag_anchor_y = e->clientY;
    g_app.drag_anchor_az = g_app.orbit.azimuth_rad;
    g_app.drag_anchor_el = g_app.orbit.elevation_rad;
    return EM_TRUE;
}

EM_BOOL on_mouse_up(int /*event_type*/, const EmscriptenMouseEvent* /*e*/, void* /*ud*/) {
    g_app.dragging = false;
    return EM_TRUE;
}

EM_BOOL on_mouse_move(int /*event_type*/, const EmscriptenMouseEvent* e, void* /*ud*/) {
    if (!g_app.dragging) {
        return EM_FALSE;
    }
    const double dx = e->clientX - g_app.drag_anchor_x;
    const double dy = e->clientY - g_app.drag_anchor_y;
    constexpr double kDragSens = 0.005;
    g_app.orbit.azimuth_rad = static_cast<float>(g_app.drag_anchor_az - (dx * kDragSens));
    const double elev_proposal = g_app.drag_anchor_el + (dy * kDragSens);
    const double lim = 0.49 * kPi;
    g_app.orbit.elevation_rad = static_cast<float>(std::max(-lim, std::min(lim, elev_proposal)));
    return EM_TRUE;
}

EM_BOOL on_wheel(int /*event_type*/, const EmscriptenWheelEvent* e, void* /*ud*/) {
    g_app.auto_orbit = false;
    g_app.last_wheel_ms = emscripten_get_now();
    // deltaY positive on scroll-down → zoom out. exp() keeps motion
    // proportional across different input devices (mouse wheel vs
    // trackpad pinch).
    const float k = std::exp(static_cast<float>(e->deltaY) * 0.001f);
    g_app.orbit.distance_M *= k;
    if (g_app.orbit.distance_M < 5.0f) {
        g_app.orbit.distance_M = 5.0f;
    }
    if (g_app.orbit.distance_M > 400.0f) {
        g_app.orbit.distance_M = 400.0f;
    }
    return EM_TRUE;
}

void register_input_callbacks() {
    // "#canvas" matches the selector in shell.html.
    emscripten_set_mousedown_callback("#canvas", nullptr, EM_TRUE, on_mouse_down);
    emscripten_set_mouseup_callback("#canvas", nullptr, EM_TRUE, on_mouse_up);
    emscripten_set_mousemove_callback("#canvas", nullptr, EM_TRUE, on_mouse_move);
    emscripten_set_wheel_callback("#canvas", nullptr, EM_TRUE, on_wheel);
}

#endif  // __EMSCRIPTEN__

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    singularity::WindowHandle win;
    singularity::RenderConfig cfg{960, 540, true};

    if (!g_app.backend.initialize(win, cfg)) {
        std::fprintf(stderr, "singularity/web: initialize() kick-off failed\n");
        return 1;
    }

    g_app.scene.metric = singularity::Scene::MetricType::Kerr;
    g_app.scene.spin_a_over_M = 0.9f;
    g_app.scene.disc_inner_M = 3.5f;
    g_app.scene.disc_outer_M = 30.0f;
    g_app.scene.exposure = 1.8f;
    g_app.scene.bloom_strength = 0.55f;
    g_app.scene.bloom_threshold = 0.8f;
    g_app.scene.disc_turbulence = 0.5f;
    g_app.scene.disc_peak_T_K = 18000.0f;
    g_app.scene.adaptive_step_on = true;

#ifdef __EMSCRIPTEN__
    register_input_callbacks();
    emscripten_set_main_loop(tick, 0, /*simulate_infinite_loop=*/0);
#endif
    return 0;
}

// C bridge for the DOM overlay panel. JS calls these via Module.ccall;
// underscore-prefixed names are listed in EXPORTED_FUNCTIONS so LTO
// can't strip them.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define SINGULARITY_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SINGULARITY_EXPORT
#endif

extern "C" {

SINGULARITY_EXPORT void singularity_set_metric(int m) {
    g_app.scene.metric = static_cast<singularity::Scene::MetricType>(m);
}
SINGULARITY_EXPORT void singularity_set_spin(float v) {
    g_app.scene.spin_a_over_M = v;
}
SINGULARITY_EXPORT void singularity_set_disc_visible(int v) {
    g_app.scene.disc_texture_on = (v != 0);
}
SINGULARITY_EXPORT void singularity_set_disc_inner(float v) {
    g_app.scene.disc_inner_M = v;
}
SINGULARITY_EXPORT void singularity_set_disc_outer(float v) {
    g_app.scene.disc_outer_M = v;
}
SINGULARITY_EXPORT void singularity_set_disc_peak_t(float v) {
    g_app.scene.disc_peak_T_K = v;
}
SINGULARITY_EXPORT void singularity_set_disc_turbulence(float v) {
    g_app.scene.disc_turbulence = v;
}
SINGULARITY_EXPORT void singularity_set_doppler(int v) {
    g_app.scene.disc_doppler_on = (v != 0);
}
SINGULARITY_EXPORT void singularity_set_redshift(int v) {
    g_app.scene.disc_redshift_on = (v != 0);
}
SINGULARITY_EXPORT void singularity_set_exposure(float v) {
    g_app.scene.exposure = v;
}
SINGULARITY_EXPORT void singularity_set_bloom_threshold(float v) {
    g_app.scene.bloom_threshold = v;
}
SINGULARITY_EXPORT void singularity_set_bloom_strength(float v) {
    g_app.scene.bloom_strength = v;
}
SINGULARITY_EXPORT void singularity_set_fov(float deg) {
    g_app.orbit.fov_y_deg = deg;
}
SINGULARITY_EXPORT void singularity_reset() {
    g_app.scene = singularity::Scene{};
    g_app.scene.metric = singularity::Scene::MetricType::Kerr;
    g_app.scene.spin_a_over_M = 0.9f;
    g_app.scene.disc_inner_M = 3.5f;
    g_app.scene.disc_outer_M = 30.0f;
    g_app.scene.exposure = 1.8f;
    g_app.scene.bloom_strength = 0.55f;
    g_app.scene.bloom_threshold = 0.8f;
    g_app.scene.disc_turbulence = 0.5f;
    g_app.scene.disc_peak_T_K = 18000.0f;
    g_app.scene.adaptive_step_on = true;
    g_app.orbit = OrbitalCamera{};
    g_app.auto_orbit = true;
    g_app.t_seconds = 0.0f;
    g_app.quality = 1;
    g_app.resolution_mode = 0;
}

// --- Performance controls ---------------------------------------------------

SINGULARITY_EXPORT void singularity_set_quality(int q) {
    g_app.quality = q;
}
SINGULARITY_EXPORT void singularity_set_resolution_mode(int m) {
    g_app.resolution_mode = m;
}
SINGULARITY_EXPORT void singularity_set_adaptive(int v) {
    g_app.scene.adaptive_step_on = (v != 0);
}
SINGULARITY_EXPORT float singularity_get_fps() {
    return (g_app.ema_ms > 0.1) ? float(1000.0 / g_app.ema_ms) : 0.0f;
}
SINGULARITY_EXPORT float singularity_get_scale_pct() {
    return 100.0f * kScaleSteps[g_app.scale_idx];
}

// --- Annotation-overlay geometry --------------------------------------------
// The camera always looks at the origin, so the shadow is centred on the
// canvas. Its apparent radius follows from the critical impact parameter
// b_crit = √27 M (Schwarzschild; the Kerr shadow is asymmetric but the
// same scale): screen_px = tan(asin(b_crit / d)) / tan(fov/2) · H/2.
SINGULARITY_EXPORT float singularity_get_shadow_px_radius(float canvas_h) {
    const float M = 1.0f;
    const float b_crit = 5.196152f * M;  // √27
    const float d = g_app.orbit.distance_M;
    const float px_cap = 4.0f * canvas_h;  // "shadow more than fills the screen"
    // Inside (or at) the critical impact parameter the shadow spans the whole
    // sky — report the cap, not zero, so the overlay treats it as huge rather
    // than absent. The zoom clamp (5M) sits just below b_crit, so this is
    // reachable.
    if (d <= b_crit * 1.02f)
        return px_cap;
    const float alpha = std::asin(b_crit / d);
    const float tan_half_fov = std::tan(0.5f * g_app.orbit.fov_y_deg * (kPi / 180.0f));
    const float px = std::tan(alpha) / tan_half_fov * 0.5f * canvas_h;
    return std::min(px, px_cap);
}

// Which side of the screen the disc's *approaching* (Doppler-brightened)
// material is on: -1 = left, +1 = right.
//
// Under the demo's invariants this is a constant: the disc always orbits
// prograde (the spin slider is non-negative), the orbital camera has no
// roll and always looks at the origin, and horizontal screen position is
// unaffected by elevation sign — so the beamed side never changes screen
// sides. The constant is calibrated against the kernel's actual output
// (left/right half-luminance with the Doppler toggle A/B'd on the live
// render): the approaching side renders on screen-RIGHT. Revisit if
// retrograde spin or camera roll ever ship.
SINGULARITY_EXPORT int singularity_get_doppler_side() {
    return 1;
}

}  // extern "C"
