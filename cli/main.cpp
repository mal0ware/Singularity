// cli/main.cpp
//
// Singularity CLI — headless driver for the verification harness and
// the Phase 1 2D toy mode. Links only against singularity::core (no GPU,
// no window system, no SDL), so this is what runs on CI before any
// backend is compiled.
//
// Usage:
//   singularity_cli --help
//   singularity_cli --mode 2d-toy [--output PATH] [--dump-trails PATH]
//
// See docs/TODO.md Phase 1 for the exit criterion.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "disc_intersection.h"
#include "kerr_hamilton.h"
#include "kerr_math.h"
#include "physics/disc.hpp"
#include "physics/kerr.hpp"
#include "physics/redshift.hpp"
#include "physics/schwarzschild.hpp"
#include "scene/scene_config.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

void print_help() {
    std::puts(
        "singularity_cli — headless physics / verification driver\n"
        "\n"
        "Usage:\n"
        "  singularity_cli --help\n"
        "  singularity_cli --mode 2d-toy        [options]\n"
        "  singularity_cli --mode kerr-geometry [options]\n"
        "\n"
        "Modes:\n"
        "  2d-toy           Integrate 100 parallel photon rays past a\n"
        "                   Schwarzschild BH in the equatorial plane and\n"
        "                   render the trails to a PNG. Phase 1.\n"
        "  kerr-2d-toy      Equatorial Kerr counterpart — visualises frame\n"
        "                   dragging at configurable spin.\n"
        "  kerr-geometry    Dump the analytic Kerr scalars (horizons, ISCO,\n"
        "                   photon spheres, ergosphere extents) as JSON for\n"
        "                   the Phase 6 verification harness.\n"
        "  photon-orbit     Integrate a single null circular orbit (default\n"
        "                   r = 1.5 r_s, Schwarzschild photon sphere) and\n"
        "                   dump the (t, r, θ, φ) trail as CSV. PHYSICS.md\n"
        "                   §5.2 closure test.\n"
        "  disc-preview     Top-down PNG of the accretion-disc temperature\n"
        "                   profile coloured through the Phase-3 blackbody\n"
        "                   sRGB LUT, with Doppler tint but no lensing.\n"
        "  cpu-render       Per-pixel Schwarzschild ray trace — disc, horizon,\n"
        "                   skybox gradient. Slow (seconds per frame) but\n"
        "                   shares the exact physics of the GPU kernels.\n"
        "  kerr-cpu-render  Kerr counterpart to cpu-render using the\n"
        "                   Hamiltonian integrator — shows frame dragging\n"
        "                   and asymmetric Doppler on the disc.\n"
        "  benchmark        Time the Schwarzschild RK4 or Kerr Hamiltonian\n"
        "                   integrator over a deterministic ray fan and emit\n"
        "                   a single JSON line of perf metrics — for CI\n"
        "                   regression tracking. Select with --metric.\n"
        "\n"
        "Options:\n"
        "  --output PATH        PNG / JSON output path (mode-dependent default)\n"
        "  --dump-trails PATH   (2d-toy) Also emit CSV of ray trails\n"
        "  --b-range MIN MAX    (2d-toy) Impact parameter span in units of M\n"
        "                       (default -10 +10). Integration bounds auto-scale.\n"
        "  --n-rays N           (2d-toy) Number of rays to fire (default 100)\n"
        "  --max-steps N        (2d-toy) Integrator step budget per ray (default 5000)\n"
        "  --spin VALUE         (kerr-*)  a/M in [0, 1] (default 0)\n"
        "  --mass VALUE         M in geometrized units (default 1)\n"
        "  --r-init VALUE       (photon-orbit) initial radius in M (default 3)\n"
        "  --orbits VALUE       (photon-orbit) orbital periods to integrate (default 1)\n"
        "  --h-step VALUE       (photon-orbit) RK4 step in M (default 0.01)\n"
        "  --r-inner VALUE      (disc-preview / cpu-render) disc inner edge in M (default ISCO)\n"
        "  --r-outer VALUE      (disc-preview / cpu-render) disc outer edge in M (default 20)\n"
        "  --resolution WxH     (cpu-render) output resolution (default 256x256)\n"
        "  --camera-distance M  (cpu-render) camera orbital radius in M (default 30)\n"
        "  --camera-elevation R (cpu-render) camera elevation from equator in rad (default 0.15)\n"
        "  --camera-fov D       (cpu-render) camera vertical FOV in degrees (default 55)\n"
        "  --supersample N      (cpu-render) N×N supersampling (default 1, max 8)\n"
        "  --metric NAME        (benchmark) 'schw' (default) or 'kerr'\n"
        "  --scene PATH         Load scene config (key = value text). Keys:\n"
        "                       mass spin disc_r_inner disc_r_outer\n"
        "                       disc_doppler_on disc_redshift_on\n"
        "                       camera_fov_deg camera_distance camera_elevation\n"
        "                       h_step max_steps\n");
}

// ---------------------------------------------------------------------------
// 2D toy mode
// ---------------------------------------------------------------------------
//
// Photons travel in straight lines in flat space, so the visual signature of
// the Schwarzschild metric is unambiguous here: parallel rays enter from the
// right and are bent toward the black hole, with rays inside the critical
// impact parameter b_crit = (3√3 / 2) r_s captured. This is the sanity check
// that every later phase is built on top of.

struct Image {
    int width;
    int height;
    std::vector<uint8_t> rgba;
};

void put_pixel(Image& img, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img.width || y >= img.height)
        return;
    const size_t i = (size_t(y) * size_t(img.width) + size_t(x)) * 4;
    img.rgba[i + 0] = r;
    img.rgba[i + 1] = g;
    img.rgba[i + 2] = b;
    img.rgba[i + 3] = 255;
}

// Bresenham segment so adjacent integrator steps don't leave gaps when the
// ray moves more than one pixel per step at small r.
void draw_line(Image& img, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        put_pixel(img, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1)
            break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

struct TrailSample {
    int ray_idx;
    int step;
    float b;
    float x;
    float y;
};

struct TwoDToyConfig {
    float b_min = -10.0f;
    float b_max = +10.0f;
    int n_rays = 100;
    int max_steps = 5000;
};

int run_2d_toy(const std::string& output_path,
               const std::string& dump_trails_path,
               const TwoDToyConfig& cfg) {
    // World: view the XY equatorial plane of Schwarzschild in geometrized
    // units (G = c = M = 1, so rs = 2). Rays enter from the right. The
    // rendered viewport stays fixed at 25 M — that's the scale the photon
    // sphere and b_crit are interesting at — while the integration bounds
    // auto-scale with the widest requested impact parameter so weak-field
    // tests at |b| ≫ rs have room to run to their true asymptotic region.
    constexpr int W = 1024;
    constexpr int H = 1024;
    constexpr float view_half = 25.0f;  // image covers [-25 M, +25 M]
    constexpr float M = 1.0f;
    constexpr float rs = 2.0f * M;
    const int n_rays = cfg.n_rays;
    const float b_min = cfg.b_min * M;
    const float b_max = cfg.b_max * M;
    const float b_abs = std::max(std::fabs(b_min), std::fabs(b_max));
    // Integration extent: start 2× outside the widest |b|, but never closer
    // in than 50 M (so the default view still exercises near-horizon curving)
    // and bail out once the ray is 4× |b| past origin.
    const float x_start = std::max(50.0f * M, 2.0f * b_abs);
    const float x_bail = -std::max(75.0f * M, 4.0f * b_abs);
    const float y_bail = std::max(75.0f * M, 4.0f * b_abs);
    const int max_steps = cfg.max_steps;
    constexpr float h_step = 0.1f;  // affine-parameter step
    constexpr float horizon_cut = 1.01f * rs;

    Image img{W, H, std::vector<uint8_t>(size_t(W) * H * 4, 15)};  // bg ~ #0f0f0f

    auto world_to_px = [&](float wx, float wy, int& px, int& py) {
        // Clamp before int conversion: for out-of-range wx/wy, int() is UB in
        // C++. put_pixel / draw_line then discards anything outside [0,W),
        // so clamping to a small pad around the viewport is both safe and
        // correct.
        const float fx = std::clamp(wx / view_half * 0.5f + 0.5f, -1.0f, 2.0f);
        const float fy = std::clamp(wy / view_half * 0.5f + 0.5f, -1.0f, 2.0f);
        px = int(fx * W);
        py = H - 1 - int(fy * H);
    };

    // Event horizon disc (solid black).
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            const float wx = (float(px) / W * 2.0f - 1.0f) * view_half;
            const float wy = -(float(py) / H * 2.0f - 1.0f) * view_half;
            if (std::sqrt(wx * wx + wy * wy) <= rs) {
                put_pixel(img, px, py, 0, 0, 0);
            }
        }
    }

    std::vector<TrailSample> trail;
    if (!dump_trails_path.empty()) {
        trail.reserve(size_t(n_rays) * 64);
    }

    for (int i = 0; i < n_rays; ++i) {
        const float b = b_min + (b_max - b_min) * (float(i) + 0.5f) / float(n_rays);
        const float x0 = x_start;
        const float y0 = b;
        const float r0 = std::sqrt(x0 * x0 + y0 * y0);
        const float phi0 = std::atan2(y0, x0);

        // Photon fired along -x̂ with unit spatial speed (affine parameter
        // chosen so the spatial part of dx^i/dλ has Euclidean length 1 at the
        // starting point). Converting Cartesian (dx/dλ, dy/dλ) = (-1, 0) to
        // polar in the equatorial plane: dr/dλ = -cos φ, dφ/dλ = sin φ / r.
        const float ur = -std::cos(phi0);
        const float uphi = std::sin(phi0) / r0;
        const float utheta = 0.0f;

        // Null condition g_μν u^μ u^ν = 0 fixes u^t (PHYSICS.md §4):
        // -(1 - rs/r)(u^t)^2 + (u^r)^2/(1 - rs/r) + r^2 sin²θ (u^φ)^2 = 0.
        const float f = 1.0f - rs / r0;
        const float ut_sq = ((ur * ur) / f + r0 * r0 * uphi * uphi) / f;
        const float ut = std::sqrt(ut_sq);

        State s{};
        s.t = 0.0f;
        s.r = r0;
        s.theta = 0.5f * kPi;  // equatorial plane
        s.phi = phi0;
        s.ut = ut;
        s.ur = ur;
        s.utheta = utheta;
        s.uphi = uphi;

        // Colour: blue (negative b) through to red (positive b).
        const float frac = float(i) / float(n_rays - 1);
        const uint8_t cr = uint8_t(60 + frac * 195);
        const uint8_t cg = uint8_t(80 + std::sin(frac * kPi) * 90);
        const uint8_t cb = uint8_t(240 - frac * 200);

        int prev_px = -1, prev_py = -1;
        bool captured = false;

        for (int step = 0; step < max_steps; ++step) {
            // Stop before rasterising if the integrator has gone bad: NaN/inf
            // can appear once ray dives through the horizon's coordinate
            // singularity. Bail cleanly rather than feeding garbage to the
            // Bresenham rasteriser.
            if (!std::isfinite(s.r) || !std::isfinite(s.phi)) {
                captured = true;
                break;
            }
            if (s.r < horizon_cut) {
                captured = true;
                break;
            }

            const float wx = s.r * std::cos(s.phi);
            const float wy = s.r * std::sin(s.phi);

            int px, py;
            world_to_px(wx, wy, px, py);
            if (prev_px >= 0) {
                draw_line(img, prev_px, prev_py, px, py, cr, cg, cb);
            } else {
                put_pixel(img, px, py, cr, cg, cb);
            }
            prev_px = px;
            prev_py = py;

            if (!dump_trails_path.empty() && (step % 20) == 0) {
                trail.push_back({i, step, b, wx, wy});
            }

            if (wx < x_bail)
                break;
            if (std::fabs(wy) > y_bail)
                break;

            s = rk4_step(s, h_step, rs);
        }

        // Always record the final state so the verification harness can
        // compute an asymptotic deflection angle, regardless of subsampling.
        if (!dump_trails_path.empty()) {
            const float wx = s.r * std::cos(s.phi);
            const float wy = s.r * std::sin(s.phi);
            trail.push_back({i, captured ? -1 : -2, b, wx, wy});
        }
    }

    if (!stbi_write_png(output_path.c_str(), W, H, 4, img.rgba.data(), W * 4)) {
        std::fprintf(stderr, "singularity_cli: failed to write '%s'\n", output_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d, %d rays)\n", output_path.c_str(), W, H, n_rays);

    if (!dump_trails_path.empty()) {
        FILE* f = std::fopen(dump_trails_path.c_str(), "wb");
        if (!f) {
            std::fprintf(
                stderr, "singularity_cli: failed to open '%s'\n", dump_trails_path.c_str());
            return 1;
        }
        std::fprintf(f, "ray_idx,step,b,x,y\n");
        for (const auto& s : trail) {
            std::fprintf(f, "%d,%d,%.9g,%.9g,%.9g\n", s.ray_idx, s.step, s.b, s.x, s.y);
        }
        std::fclose(f);
        std::printf("wrote %s (%zu samples)\n", dump_trails_path.c_str(), trail.size());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Kerr 2D toy mode
// ---------------------------------------------------------------------------
//
// Parallel-ray visualisation for Kerr in the equatorial plane — the analogue
// of the Schwarzschild 2d-toy. Rays enter from the right at a range of impact
// parameters, are integrated through the spinning spacetime, and rendered as
// coloured trails on a PNG with the event horizon painted black. Frame
// dragging manifests visually as an asymmetry between rays on the prograde
// and retrograde sides of the BH: rays moving with the spin (positive b in
// our convention) wrap more tightly than rays against the spin (negative b).
//
// Initialisation: at large r and equatorial θ = π/2 the Kerr metric is a
// small perturbation of Schwarzschild, so we borrow the Schwarzschild-style
// initial direction (dr/dλ, dφ/dλ) from the ray's Cartesian entry and then
// compute the conserved (E, L_z, Q) from it via the covariant null condition.
// Q = 0 exactly because the ray is equatorial.

int run_kerr_2d_toy(const std::string& output_path,
                    const std::string& dump_trails_path,
                    const TwoDToyConfig& cfg,
                    float M,
                    float a) {
    using singularity::physics::kerr_outer_horizon;

    if (M <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: --mass must be positive\n");
        return 2;
    }
    if (a < 0.0f || a > M) {
        std::fprintf(stderr, "singularity_cli: --spin %.6f out of range [0, M=%.6f]\n", a, M);
        return 2;
    }

    constexpr int W = 1024;
    constexpr int H = 1024;
    constexpr float view_half = 25.0f;
    const int n_rays = cfg.n_rays;
    const float b_min = cfg.b_min * M;
    const float b_max = cfg.b_max * M;
    const float b_abs = std::max(std::fabs(b_min), std::fabs(b_max));
    const float x_start = std::max(50.0f * M, 2.0f * b_abs);
    const float x_bail = -std::max(75.0f * M, 4.0f * b_abs);
    const float y_bail = std::max(75.0f * M, 4.0f * b_abs);
    const int max_steps = cfg.max_steps;
    constexpr float h_step = 0.1f;
    const float r_plus = kerr_outer_horizon(M, a);
    const float horizon_cut = 1.02f * r_plus;

    Image img{W, H, std::vector<uint8_t>(size_t(W) * H * 4, 15)};

    auto world_to_px = [&](float wx, float wy, int& px, int& py) {
        const float fx = std::clamp(wx / view_half * 0.5f + 0.5f, -1.0f, 2.0f);
        const float fy = std::clamp(wy / view_half * 0.5f + 0.5f, -1.0f, 2.0f);
        px = int(fx * W);
        py = H - 1 - int(fy * H);
    };

    // Paint the event horizon — for Kerr in BL the outer horizon is a sphere
    // of coordinate radius r_+. The ergosphere could be overlaid next but
    // we keep this render clean for now.
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            const float wx = (float(px) / W * 2.0f - 1.0f) * view_half;
            const float wy = -(float(py) / H * 2.0f - 1.0f) * view_half;
            if (std::sqrt(wx * wx + wy * wy) <= r_plus) {
                put_pixel(img, px, py, 0, 0, 0);
            }
        }
    }

    std::vector<TrailSample> trail;
    if (!dump_trails_path.empty()) {
        trail.reserve(size_t(n_rays) * 64);
    }

    for (int i = 0; i < n_rays; ++i) {
        const float b = b_min + (b_max - b_min) * (float(i) + 0.5f) / float(n_rays);
        const float x0 = x_start;
        const float y0 = b;
        const float r0 = std::sqrt(x0 * x0 + y0 * y0);
        const float phi0 = std::atan2(y0, x0);

        // Initial coordinate velocity — same as the Schwarzschild toy at
        // large r, where the BL spatial part is approximately spherical.
        const float ur = -std::cos(phi0);
        const float uphi = std::sin(phi0) / r0;

        // Covariant null-condition solve for u^t, and (E, L_z) from u^μ.
        // Equatorial Kerr:
        //   g_tt  = −(1 − 2M/r)
        //   g_tφ  = −2 M a / r
        //   g_rr  = r² / Δ
        //   g_φφ  = ((r²+a²)² − a² Δ) / r²
        const float Delta = r0 * r0 - 2.0f * M * r0 + a * a;
        const float g_tt = -(1.0f - 2.0f * M / r0);
        const float g_tp = -2.0f * M * a / r0;
        const float g_rr = r0 * r0 / Delta;
        const float g_pp = ((r0 * r0 + a * a) * (r0 * r0 + a * a) - a * a * Delta) / (r0 * r0);

        // Null condition is quadratic in u^t with u^r, u^φ fixed:
        //   g_tt (u^t)² + 2 g_tφ u^t u^φ + g_rr (u^r)² + g_φφ (u^φ)² = 0
        // => u^t = [−g_tφ u^φ ± √((g_tφ u^φ)² − g_tt·(g_rr u^r² + g_φφ u^φ²))] / g_tt
        // We take the root with positive u^t (future-directed photon).
        const float A = g_tt;
        const float B = 2.0f * g_tp * uphi;
        const float C = g_rr * ur * ur + g_pp * uphi * uphi;
        const float disc = B * B - 4.0f * A * C;
        if (disc < 0.0f) {
            std::fprintf(stderr, "singularity_cli: kerr-2d-toy bad init (ray %d)\n", i);
            continue;
        }
        // Future-directed: u^t > 0. A = g_tt < 0, so the − branch of the
        // quadratic formula with A < 0 gives u^t > 0.
        const float ut = (-B - std::sqrt(disc)) / (2.0f * A);

        // Conserved quantities from the 4-velocity:
        //   E   = −(g_tt u^t + g_tφ u^φ)
        //   L_z =  g_tφ u^t + g_φφ u^φ
        //   Q   = 0  (equatorial + u^θ = 0)
        KerrConserved c{};
        c.E = -(g_tt * ut + g_tp * uphi);
        c.L_z = g_tp * ut + g_pp * uphi;
        c.Q = 0.0f;
        c.a = a;
        c.M = M;

        KerrHamState s{};
        s.t = 0.0f;
        s.r = r0;
        s.theta = 0.5f * kPi;
        s.phi = phi0;
        kerr_ham_momenta_from_velocities(&s, ur, /*u_θ=*/0.0f, c);

        // Blue (negative b — retrograde side) → red (positive b — prograde).
        const float frac = float(i) / float(n_rays - 1);
        const uint8_t cr = uint8_t(60 + frac * 195);
        const uint8_t cg = uint8_t(80 + std::sin(frac * kPi) * 90);
        const uint8_t cb = uint8_t(240 - frac * 200);

        int prev_px = -1, prev_py = -1;
        bool captured = false;

        for (int step = 0; step < max_steps; ++step) {
            if (!std::isfinite(s.r) || !std::isfinite(s.phi)) {
                captured = true;
                break;
            }
            if (s.r < horizon_cut) {
                captured = true;
                break;
            }

            const float wx = s.r * std::cos(s.phi);
            const float wy = s.r * std::sin(s.phi);

            int px, py;
            world_to_px(wx, wy, px, py);
            if (prev_px >= 0) {
                draw_line(img, prev_px, prev_py, px, py, cr, cg, cb);
            } else {
                put_pixel(img, px, py, cr, cg, cb);
            }
            prev_px = px;
            prev_py = py;

            if (!dump_trails_path.empty() && (step % 20) == 0) {
                trail.push_back({i, step, b, wx, wy});
            }

            if (wx < x_bail)
                break;
            if (std::fabs(wy) > y_bail)
                break;

            s = kerr_ham_rk4_step(s, h_step, c);
        }

        if (!dump_trails_path.empty()) {
            const float wx = s.r * std::cos(s.phi);
            const float wy = s.r * std::sin(s.phi);
            trail.push_back({i, captured ? -1 : -2, b, wx, wy});
        }
    }

    if (!stbi_write_png(output_path.c_str(), W, H, 4, img.rgba.data(), W * 4)) {
        std::fprintf(stderr, "singularity_cli: failed to write '%s'\n", output_path.c_str());
        return 1;
    }
    std::printf(
        "wrote %s (%dx%d, %d rays, a/M = %.4g)\n", output_path.c_str(), W, H, n_rays, a / M);

    if (!dump_trails_path.empty()) {
        FILE* f = std::fopen(dump_trails_path.c_str(), "wb");
        if (!f) {
            std::fprintf(
                stderr, "singularity_cli: failed to open '%s'\n", dump_trails_path.c_str());
            return 1;
        }
        std::fprintf(f, "ray_idx,step,b,x,y\n");
        for (const auto& s : trail) {
            std::fprintf(f, "%d,%d,%.9g,%.9g,%.9g\n", s.ray_idx, s.step, s.b, s.x, s.y);
        }
        std::fclose(f);
        std::printf("wrote %s (%zu samples)\n", dump_trails_path.c_str(), trail.size());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Procedural starfield — shared by the CPU ray tracers
// ---------------------------------------------------------------------------
//
// Cheap but convincing background: bucket the escape-ray direction into
// angular cells, hash the cell index to decide whether a star sits inside
// and (if so) pick its brightness and colour from a deterministic RNG.
// No file-backed skybox texture required — keeps the CLI dependency-free
// and the seed reproducible. Not astronomically accurate but dense enough
// that the eye reads it as "space".

namespace {

inline uint32_t _xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline uint32_t _mix32(uint32_t a, uint32_t b) {
    uint32_t x = a * 0x9E3779B9u + b;
    return _xorshift32(x);
}

// Returns (R, G, B) for the background at direction (dir_x, dir_y, dir_z).
struct _BgColor {
    uint8_t r, g, b;
};

inline _BgColor starfield_color(float dir_x, float dir_y, float dir_z) {
    // Normalise and convert to equirectangular (θ, φ) cells.
    const float len = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
    if (len < 1e-6f)
        return {0, 0, 1};
    const float nx = dir_x / len, ny = dir_y / len, nz = dir_z / len;
    const float theta = std::acos(std::clamp(nz, -1.0f, 1.0f));
    const float phi = std::atan2(ny, nx) + kPi;  // in [0, 2π)

    // Bucket into a uniform grid of ~768 × 384 cells. One potential star per
    // cell, populated by a Bernoulli-like hash.
    constexpr int NX = 768;
    constexpr int NY = 384;
    const int ix = int(phi / (2.0f * kPi) * NX) % NX;
    const int iy = int(theta / kPi * NY) % NY;

    const uint32_t seed = _mix32(uint32_t(ix), uint32_t(iy));
    const bool has_star = (seed & 0xFFu) < 4u;  // ~1.5% of cells

    // Near-black base. The previous Milky-Way-style latitude tint read as
    // off-white gray once bloom + ACES lifted the highlights; pure black is
    // closer to what real deep space looks like, and lets the bright stars
    // visibly *pop* instead of vanishing into the haze.
    (void)nz;                            // lat_frac no longer used
    float r = 0.4f, g = 0.5f, b = 1.3f;  // /255 → near-pure black with blue tint

    if (has_star) {
        // Star brightness is a steep power-law cut of the next hash byte — a
        // few bright, many invisible. Steeper exponent than before (4 vs 6)
        // so the visible peaks land harder.
        const float bright_raw = float((seed >> 8) & 0xFFu) / 255.0f;
        const float brightness = 255.0f * std::pow(bright_raw, 4.0f);
        const float hue = float((seed >> 16) & 0xFFu) / 255.0f;
        // Low hue → red-orange (K/M stars), high hue → blue-white (O/B).
        const float star_r = 1.0f - 0.5f * hue;
        const float star_g = 0.8f + 0.2f * hue * (1.0f - hue) * 4.0f;
        const float star_b = 0.4f + 0.6f * hue;
        r = std::min(255.0f, r + brightness * star_r);
        g = std::min(255.0f, g + brightness * star_g);
        b = std::min(255.0f, b + brightness * star_b);
    }
    return {uint8_t(r), uint8_t(g), uint8_t(b)};
}

}  // namespace

// ---------------------------------------------------------------------------
// CPU ray-traced Schwarzschild renderer — Phase 2 preview
// ---------------------------------------------------------------------------
//
// Per-pixel null-geodesic integration through Schwarzschild spacetime on
// the CPU. Slow — typical 256x256 frame takes a few seconds — but correct
// in the same sense the GPU kernels will be: every ray traces the full
// metric, so gravitational lensing, the photon ring, and the relativistic
// disc appearance all emerge from the same ``shared_shader/geodesic_math.h``
// that feeds Metal and Vulkan. Exists so the physics stack can be driven
// end-to-end from CPU-only hardware (WSL2, CI) before the Phase 2 / 4 GPU
// backends come online.
//
// Each ray starts at the camera and is traced *backwards* in affine
// parameter until it meets one of:
//   - the horizon (r < r_+)               → colour pixel black
//   - the equatorial accretion disc (r∈[r_in, r_out] at θ=π/2)
//                                         → sample T(r) via Novikov-Thorne,
//                                           apply redshift + Doppler, colour
//                                           through the Tanner-Helland LUT
//   - large r (escape to infinity)        → colour by direction (gradient)

int run_cpu_render(const std::string& output_path,
                   int width,
                   int height,
                   int supersample,
                   float M,
                   float r_inner_M,
                   float r_outer_M,
                   float camera_distance_M,
                   float camera_elevation_rad,
                   float camera_fov_deg) {
    using singularity::physics::blackbody_srgb_tanner_helland;
    using singularity::physics::disc_temperature_nt;
    using singularity::physics::isco_timelike;
    using singularity::physics::sRGBColor;

    if (M <= 0.0f || width < 4 || height < 4) {
        std::fprintf(stderr, "singularity_cli: bad cpu-render args\n");
        return 2;
    }

    const float rs = 2.0f * M;
    const float r_inner = (r_inner_M > 0.0f) ? (r_inner_M * M) : isco_timelike(M);
    const float r_outer = r_outer_M * M;
    if (r_outer <= r_inner) {
        std::fprintf(stderr, "singularity_cli: r_outer must exceed r_inner\n");
        return 2;
    }

    // Camera — orbital about the origin, looking at it.
    const float cam_r = camera_distance_M * M;
    const float cam_th = 0.5f * kPi - camera_elevation_rad;  // polar
    const float cam_phi = 0.0f;
    const float cos_th = std::cos(cam_th);
    const float sin_th = std::sin(cam_th);
    const float cos_phi = std::cos(cam_phi);
    const float sin_phi = std::sin(cam_phi);

    // Camera-frame basis vectors (orthonormal triad in the Cartesian
    // approximation at large r). We look toward origin, so "forward" points
    // inward.
    const float cx = cam_r * sin_th * cos_phi;
    const float cy = cam_r * sin_th * sin_phi;
    const float cz = cam_r * cos_th;
    const float fwd_x = -cx / cam_r;
    const float fwd_y = -cy / cam_r;
    const float fwd_z = -cz / cam_r;
    // World-up = +z; right = cross(up, forward) normalised.
    float rt_x = fwd_y * 1.0f - fwd_z * 0.0f;  // (up × fwd)_x; up = (0,0,1)
    float rt_y = -fwd_x * 1.0f + fwd_z * 0.0f;
    float rt_z = 0.0f;
    {
        const float rt_len = std::sqrt(rt_x * rt_x + rt_y * rt_y + rt_z * rt_z);
        rt_x /= rt_len;
        rt_y /= rt_len;
        rt_z /= rt_len;
    }
    // Actual up = fwd × right (ensures right-handed basis).
    const float up_x = fwd_y * rt_z - fwd_z * rt_y;
    const float up_y = fwd_z * rt_x - fwd_x * rt_z;
    const float up_z = fwd_x * rt_y - fwd_y * rt_x;

    const float aspect = float(width) / float(height);
    const float tan_half = std::tan(0.5f * camera_fov_deg * kPi / 180.0f);
    const float horizon_cut = 1.02f * rs;
    const float escape_r = 200.0f * M;
    constexpr int max_steps = 5000;
    constexpr float h_step = 0.2f;

    std::vector<uint8_t> pixels(size_t(width) * height * 4, 0);

    // Orthonormal BL basis at the camera point (equatorial simplifies since
    // our default elevation is 0, but we handle the general case).
    // Radial unit (Cartesian): r̂ = (sinθ cosφ, sinθ sinφ, cosθ)
    // Polar unit:  θ̂ = (cosθ cosφ, cosθ sinφ, -sinθ)
    // Azimuth unit: φ̂ = (-sinφ, cosφ, 0)
    const float rhat_x = sin_th * cos_phi;
    const float rhat_y = sin_th * sin_phi;
    const float rhat_z = cos_th;
    const float that_x = cos_th * cos_phi;
    const float that_y = cos_th * sin_phi;
    const float that_z = -sin_th;
    const float phat_x = -sin_phi;
    const float phat_y = cos_phi;
    const float phat_z = 0.0f;

    // Supersampling grid — N×N sub-pixels averaged into each output pixel.
    // N=1 is the baseline; N=2 doubles the physical resolution internally
    // and quarters the aliasing at 4× cost.
    const int ss = std::max(1, supersample);
    const float sub_w = 1.0f / float(ss);

    // Per-pixel work is embarrassingly parallel — split rows across
    // hardware threads via a work-stealing atomic counter.
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next_row{0};
    auto render_row = [&](int py) {
        for (int px = 0; px < width; ++px) {
            float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    const float fx = float(px) + (sx + 0.5f) * sub_w;
                    const float fy = float(py) + (sy + 0.5f) * sub_w;
                    // NDC pixel → camera-space direction.
                    const float u = (2.0f * fx / float(width) - 1.0f) * aspect * tan_half;
                    const float v = -(2.0f * fy / float(height) - 1.0f) * tan_half;
                    float dir_x = fwd_x + rt_x * u + up_x * v;
                    float dir_y = fwd_y + rt_y * u + up_y * v;
                    float dir_z = fwd_z + rt_z * u + up_z * v;
                    const float dir_len = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
                    dir_x /= dir_len;
                    dir_y /= dir_len;
                    dir_z /= dir_len;

                    // Project onto BL orthonormal basis at the camera point.
                    const float ur_ortho = dir_x * rhat_x + dir_y * rhat_y + dir_z * rhat_z;
                    const float uth_ortho = dir_x * that_x + dir_y * that_y + dir_z * that_z;
                    const float uphi_ortho = dir_x * phat_x + dir_y * phat_y + dir_z * phat_z;

                    // Coordinate 4-velocity components: scale by the inverse of the
                    // BL basis norms (r for θ, r sinθ for φ).
                    State s{};
                    s.t = 0.0f;
                    s.r = cam_r;
                    s.theta = cam_th;
                    s.phi = cam_phi;
                    s.ur = ur_ortho;
                    s.utheta = uth_ortho / cam_r;
                    s.uphi = uphi_ortho / (cam_r * sin_th);
                    // Null condition: g_μν u^μ u^ν = 0 fixes |u^t|. Future-directed
                    // photon, so u^t > 0.
                    const float f = 1.0f - rs / cam_r;
                    const float kin =
                        (ur_ortho * ur_ortho) / f + uth_ortho * uth_ortho + uphi_ortho * uphi_ortho;
                    s.ut = std::sqrt(kin / f);

                    uint8_t cr = 0, cg = 0, cb = 0;
                    float prev_theta = s.theta;
                    float prev_r = s.r;
                    float prev_uphi = s.uphi;

                    for (int step = 0; step < max_steps; ++step) {
                        if (!std::isfinite(s.r) || s.r < horizon_cut) {
                            break;  // BH — pixel stays black from initialisation.
                        }
                        if (s.r > escape_r) {
                            // Sky — sample the starfield at the ray's asymptotic
                            // direction. Convert the current BL state to a unit
                            // Cartesian direction so the starfield stays rotation-
                            // consistent across rays.
                            const float sx = std::sin(s.theta) * std::cos(s.phi);
                            const float sy = std::sin(s.theta) * std::sin(s.phi);
                            const float sz = std::cos(s.theta);
                            const auto bg = starfield_color(sx, sy, sz);
                            cr = bg.r;
                            cg = bg.g;
                            cb = bg.b;
                            break;
                        }

                        // Disc intersection — θ crosses π/2 between two steps.
                        // Shared helper: see shared_shader/disc_intersection.h.
                        const float new_theta = s.theta;
                        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, new_theta);
                        if (xing.crossed) {
                            const float r_cross = lerp_scalar(prev_r, s.r, xing.frac);
                            const float uphi_cross = lerp_scalar(prev_uphi, s.uphi, xing.frac);
                            if (in_disc_annulus(r_cross, r_inner, r_outer)) {
                                const float T =
                                    disc_temperature_nt(r_cross, r_inner, 1.0e4f, r_inner);
                                if (T > 0.0f) {
                                    // Gravitational redshift, emitter at r_cross to
                                    // observer at cam_r (static approximation).
                                    const float f_emit = 1.0f - rs / r_cross;
                                    const float f_obs = 1.0f - rs / cam_r;
                                    const float g_grav = std::sqrt(f_emit / f_obs);
                                    // Simple Doppler via Keplerian orbital speed and
                                    // the ray's azimuthal direction at the crossing.
                                    const float v_orb =
                                        std::sqrt(M / (r_cross * r_cross * r_cross));
                                    const float speed = r_cross * v_orb / std::sqrt(f_emit);
                                    const float cos_psi =
                                        std::clamp(-uphi_cross * r_cross, -1.0f, 1.0f);
                                    const float g_dop =
                                        std::sqrt(1.0f - speed * speed) / (1.0f - speed * cos_psi);
                                    const float g = g_grav * g_dop;
                                    const sRGBColor col = blackbody_srgb_tanner_helland(g * T);
                                    const float flux =
                                        std::clamp(std::pow(r_inner / r_cross, 3.0f), 0.05f, 1.0f);
                                    cr = uint8_t(std::clamp(col.r * flux, 0.0f, 1.0f) * 255.0f);
                                    cg = uint8_t(std::clamp(col.g * flux, 0.0f, 1.0f) * 255.0f);
                                    cb = uint8_t(std::clamp(col.b * flux, 0.0f, 1.0f) * 255.0f);
                                }
                                break;
                            }
                        }
                        prev_theta = new_theta;
                        prev_r = s.r;
                        prev_uphi = s.uphi;

                        s = rk4_step(s, h_step, rs);
                    }

                    acc_r += float(cr);
                    acc_g += float(cg);
                    acc_b += float(cb);
                }
            }
            const float inv = 1.0f / float(ss * ss);
            const size_t i = (size_t(py) * size_t(width) + size_t(px)) * 4;
            pixels[i + 0] = uint8_t(std::clamp(acc_r * inv, 0.0f, 255.0f));
            pixels[i + 1] = uint8_t(std::clamp(acc_g * inv, 0.0f, 255.0f));
            pixels[i + 2] = uint8_t(std::clamp(acc_b * inv, 0.0f, 255.0f));
            pixels[i + 3] = 255;
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            int py;
            while ((py = next_row.fetch_add(1)) < height) {
                render_row(py);
            }
        });
    }
    for (auto& w : workers)
        w.join();

    if (!stbi_write_png(output_path.c_str(), width, height, 4, pixels.data(), width * 4)) {
        std::fprintf(stderr, "singularity_cli: failed to write '%s'\n", output_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d CPU ray-trace, ss=%d, r_inner=%.3gM, r_outer=%.3gM, "
                "cam_dist=%.3gM, elev=%.3g rad, fov=%.3g°)\n",
                output_path.c_str(),
                width,
                height,
                ss,
                r_inner / M,
                r_outer / M,
                camera_distance_M,
                camera_elevation_rad,
                camera_fov_deg);
    return 0;
}

// ---------------------------------------------------------------------------
// CPU ray-traced Kerr renderer — Phase 6 preview
// ---------------------------------------------------------------------------
//
// Kerr counterpart to ``cpu-render``. Uses the Hamiltonian-form integrator
// in ``shared_shader/kerr_hamilton.h``, so it smoothly handles turning
// points and produces the full asymmetric disc signature (frame-dragging
// distortion, prograde Doppler brightening, retrograde dimming).

int run_kerr_cpu_render(const std::string& output_path,
                        int width,
                        int height,
                        int supersample,
                        float M,
                        float a,
                        float r_inner_M,
                        float r_outer_M,
                        float camera_distance_M,
                        float camera_elevation_rad,
                        float camera_fov_deg) {
    using singularity::physics::blackbody_srgb_tanner_helland;
    using singularity::physics::disc_temperature_nt;
    using singularity::physics::kerr_isco_prograde;
    using singularity::physics::kerr_outer_horizon;
    using singularity::physics::sRGBColor;

    if (M <= 0.0f || a < 0.0f || a > M || width < 4 || height < 4) {
        std::fprintf(stderr, "singularity_cli: bad kerr-cpu-render args\n");
        return 2;
    }

    const float r_plus = kerr_outer_horizon(M, a);
    const float r_inner = (r_inner_M > 0.0f) ? (r_inner_M * M) : kerr_isco_prograde(M, a);
    const float r_outer = r_outer_M * M;
    if (r_outer <= r_inner) {
        std::fprintf(stderr, "singularity_cli: r_outer must exceed r_inner\n");
        return 2;
    }

    const float cam_r = camera_distance_M * M;
    const float cam_th = 0.5f * kPi - camera_elevation_rad;
    const float cam_phi = 0.0f;
    const float cos_th = std::cos(cam_th);
    const float sin_th = std::sin(cam_th);
    const float cos_phi = std::cos(cam_phi);
    const float sin_phi = std::sin(cam_phi);

    const float cx = cam_r * sin_th * cos_phi;
    const float cy = cam_r * sin_th * sin_phi;
    const float cz = cam_r * cos_th;
    const float fwd_x = -cx / cam_r;
    const float fwd_y = -cy / cam_r;
    const float fwd_z = -cz / cam_r;
    float rt_x = fwd_y;
    float rt_y = -fwd_x;
    float rt_z = 0.0f;
    {
        const float len = std::sqrt(rt_x * rt_x + rt_y * rt_y + rt_z * rt_z);
        rt_x /= len;
        rt_y /= len;
        rt_z /= len;
    }
    const float up_x = fwd_y * rt_z - fwd_z * rt_y;
    const float up_y = fwd_z * rt_x - fwd_x * rt_z;
    const float up_z = fwd_x * rt_y - fwd_y * rt_x;

    const float aspect = float(width) / float(height);
    const float tan_half = std::tan(0.5f * camera_fov_deg * kPi / 180.0f);
    const float horizon_cut = 1.02f * r_plus;
    const float escape_r = 200.0f * M;
    constexpr int max_steps = 6000;
    constexpr float h_step = 0.2f;

    const float rhat_x = sin_th * cos_phi;
    const float rhat_y = sin_th * sin_phi;
    const float rhat_z = cos_th;
    const float that_x = cos_th * cos_phi;
    const float that_y = cos_th * sin_phi;
    const float that_z = -sin_th;
    const float phat_x = -sin_phi;
    const float phat_y = cos_phi;
    const float phat_z = 0.0f;

    std::vector<uint8_t> pixels(size_t(width) * height * 4, 0);

    const int ss = std::max(1, supersample);
    const float sub_w = 1.0f / float(ss);
    const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next_row{0};
    auto render_row = [&](int py) {
        for (int px = 0; px < width; ++px) {
            float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    const float fx = float(px) + (sx + 0.5f) * sub_w;
                    const float fy = float(py) + (sy + 0.5f) * sub_w;
                    const float u = (2.0f * fx / float(width) - 1.0f) * aspect * tan_half;
                    const float v = -(2.0f * fy / float(height) - 1.0f) * tan_half;
                    float dir_x = fwd_x + rt_x * u + up_x * v;
                    float dir_y = fwd_y + rt_y * u + up_y * v;
                    float dir_z = fwd_z + rt_z * u + up_z * v;
                    const float dir_len = std::sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z);
                    dir_x /= dir_len;
                    dir_y /= dir_len;
                    dir_z /= dir_len;

                    const float ur_ortho = dir_x * rhat_x + dir_y * rhat_y + dir_z * rhat_z;
                    const float uth_ortho = dir_x * that_x + dir_y * that_y + dir_z * that_z;
                    const float uphi_ortho = dir_x * phat_x + dir_y * phat_y + dir_z * phat_z;
                    const float ur_coord = ur_ortho;
                    const float uth_coord = uth_ortho / cam_r;
                    const float uphi_coord = uphi_ortho / (cam_r * sin_th);

                    // Kerr metric components at the camera — solve the null condition
                    // quadratic g_tt(u^t)² + 2g_tφ u^t u^φ + g_rr u^r² + g_θθ u^θ² +
                    // g_φφ u^φ² = 0 for u^t > 0 (future-directed).
                    const float Sigma_cam = cam_r * cam_r + a * a * cos_th * cos_th;
                    const float Delta_cam = cam_r * cam_r - 2.0f * M * cam_r + a * a;
                    const float s2_cam = sin_th * sin_th;
                    const float g_tt = -(1.0f - 2.0f * M * cam_r / Sigma_cam);
                    const float g_tp = -2.0f * M * cam_r * a * s2_cam / Sigma_cam;
                    const float g_rr = Sigma_cam / Delta_cam;
                    const float g_thth = Sigma_cam;
                    const float g_pp = s2_cam
                                       * ((cam_r * cam_r + a * a) * (cam_r * cam_r + a * a)
                                          - a * a * Delta_cam * s2_cam)
                                       / Sigma_cam;
                    const float A_q = g_tt;
                    const float B_q = 2.0f * g_tp * uphi_coord;
                    const float C_q = g_rr * ur_coord * ur_coord + g_thth * uth_coord * uth_coord
                                      + g_pp * uphi_coord * uphi_coord;
                    const float disc_q = B_q * B_q - 4.0f * A_q * C_q;
                    if (disc_q < 0.0f) {
                        // Degenerate initialisation — leave pixel black.
                        continue;
                    }
                    const float ut = (-B_q - std::sqrt(disc_q)) / (2.0f * A_q);

                    KerrConserved c{};
                    c.a = a;
                    c.M = M;
                    c.E = -(g_tt * ut + g_tp * uphi_coord);
                    c.L_z = g_tp * ut + g_pp * uphi_coord;
                    // Carter constant for a photon (m² = 0):
                    //   Q = p_θ² + cos²θ [L_z² / sin²θ − a² E²]
                    // with p_θ = g_θθ u^θ = Σ u^θ. PHYSICS.md §7.3.
                    const float p_theta_init = Sigma_cam * uth_coord;
                    c.Q = p_theta_init * p_theta_init
                          + cos_th * cos_th * (c.L_z * c.L_z / s2_cam - a * a * c.E * c.E);

                    KerrHamState s{};
                    s.t = 0.0f;
                    s.r = cam_r;
                    s.theta = cam_th;
                    s.phi = cam_phi;
                    s.p_r = g_rr * ur_coord;
                    s.p_theta = g_thth * uth_coord;

                    uint8_t cr = 0, cg = 0, cb = 0;
                    float prev_theta = s.theta;
                    float prev_r = s.r;

                    for (int step = 0; step < max_steps; ++step) {
                        if (!std::isfinite(s.r) || s.r < horizon_cut) {
                            break;
                        }
                        if (s.r > escape_r) {
                            const float sx = std::sin(s.theta) * std::cos(s.phi);
                            const float sy = std::sin(s.theta) * std::sin(s.phi);
                            const float sz = std::cos(s.theta);
                            const auto bg = starfield_color(sx, sy, sz);
                            cr = bg.r;
                            cg = bg.g;
                            cb = bg.b;
                            break;
                        }

                        const float new_theta = s.theta;
                        const DiscCrossing xing = detect_equatorial_crossing(prev_theta, new_theta);
                        if (xing.crossed) {
                            const float r_cross = lerp_scalar(prev_r, s.r, xing.frac);
                            if (in_disc_annulus(r_cross, r_inner, r_outer)) {
                                const float T =
                                    disc_temperature_nt(r_cross, r_inner, 1.0e4f, r_inner);
                                if (T > 0.0f) {
                                    const float f_emit_approx = 1.0f - 2.0f * M / r_cross;
                                    const float f_obs_approx = 1.0f - 2.0f * M / cam_r;
                                    const float g_grav = std::sqrt(std::max(f_emit_approx, 1e-6f)
                                                                   / std::max(f_obs_approx, 1e-6f));
                                    const float omega_K =
                                        std::sqrt(M / (r_cross * r_cross * r_cross))
                                        / (1.0f + a * std::sqrt(M / (r_cross * r_cross * r_cross)));
                                    const float speed = r_cross * omega_K
                                                        / std::sqrt(std::max(f_emit_approx, 1e-6f));
                                    // Azimuthal component of ray tangent at crossing.
                                    // Use p_φ = L_z, u^φ = ∂H/∂p_φ ≈ L_z/(r² sin²θ)
                                    // at near-equatorial crossing. Simplified.
                                    const float cos_psi = std::clamp(
                                        c.L_z / (r_cross * r_cross) * r_cross, -1.0f, 1.0f);
                                    const float v2 = speed * speed;
                                    const float g_dop = std::sqrt(std::max(1.0f - v2, 1e-6f))
                                                        / (1.0f - speed * cos_psi);
                                    const float g = g_grav * g_dop;
                                    const sRGBColor col = blackbody_srgb_tanner_helland(g * T);
                                    const float flux =
                                        std::clamp(std::pow(r_inner / r_cross, 3.0f), 0.05f, 1.0f);
                                    cr = uint8_t(std::clamp(col.r * flux, 0.0f, 1.0f) * 255.0f);
                                    cg = uint8_t(std::clamp(col.g * flux, 0.0f, 1.0f) * 255.0f);
                                    cb = uint8_t(std::clamp(col.b * flux, 0.0f, 1.0f) * 255.0f);
                                }
                                break;
                            }
                        }
                        prev_theta = new_theta;
                        prev_r = s.r;

                        s = kerr_ham_rk4_step(s, h_step, c);
                    }

                    acc_r += float(cr);
                    acc_g += float(cg);
                    acc_b += float(cb);
                }
            }
            const float inv = 1.0f / float(ss * ss);
            const size_t idx = (size_t(py) * size_t(width) + size_t(px)) * 4;
            pixels[idx + 0] = uint8_t(std::clamp(acc_r * inv, 0.0f, 255.0f));
            pixels[idx + 1] = uint8_t(std::clamp(acc_g * inv, 0.0f, 255.0f));
            pixels[idx + 2] = uint8_t(std::clamp(acc_b * inv, 0.0f, 255.0f));
            pixels[idx + 3] = 255;
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        workers.emplace_back([&]() {
            int py;
            while ((py = next_row.fetch_add(1)) < height) {
                render_row(py);
            }
        });
    }
    for (auto& w : workers)
        w.join();

    if (!stbi_write_png(output_path.c_str(), width, height, 4, pixels.data(), width * 4)) {
        std::fprintf(stderr, "singularity_cli: failed to write '%s'\n", output_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d Kerr CPU ray-trace, a/M=%.3g, ss=%d, "
                "r_inner=%.3gM, r_outer=%.3gM, cam_dist=%.3gM, elev=%.3g, fov=%.3g°)\n",
                output_path.c_str(),
                width,
                height,
                a / M,
                ss,
                r_inner / M,
                r_outer / M,
                camera_distance_M,
                camera_elevation_rad,
                camera_fov_deg);
    return 0;
}

// ---------------------------------------------------------------------------
// Disc preview mode
// ---------------------------------------------------------------------------
//
// Top-down "flat" render of the accretion disc's temperature profile
// coloured via the Tanner-Helland blackbody sRGB approximation, with a
// horizontal-Doppler tint added so the prograde side reads blue-shifted and
// the retrograde side red-shifted. No gravitational lensing is applied —
// this is a Phase-3 building-block preview that validates the disc physics
// (T(r), redshift + Doppler, blackbody colour) before the Phase-2 Metal /
// Phase-4 Vulkan kernels wire it into a true ray-traced frame. Event
// horizon and ergosphere appear as outlines; the BH sits black and the disc
// is rendered from r_ISCO out to r_outer.

int run_disc_preview(
    const std::string& output_path, float M, float a, float r_inner_M, float r_outer_M) {
    using singularity::physics::blackbody_srgb_tanner_helland;
    using singularity::physics::disc_temperature_nt;
    using singularity::physics::intensity_scaling;
    using singularity::physics::kerr_ergosphere_outer;
    using singularity::physics::kerr_isco_prograde;
    using singularity::physics::kerr_outer_horizon;
    using singularity::physics::schwarzschild_combined_shift;
    using singularity::physics::schwarzschild_keplerian_omega;
    using singularity::physics::sRGBColor;

    if (M <= 0.0f || a < 0.0f || a > M) {
        std::fprintf(stderr, "singularity_cli: bad --mass / --spin combination\n");
        return 2;
    }

    // Prefer caller-supplied r_inner; default to ISCO at the requested spin.
    const float r_isco = kerr_isco_prograde(M, a);
    const float r_inner = (r_inner_M > 0.0f) ? (r_inner_M * M) : r_isco;
    const float r_outer = r_outer_M * M;
    if (r_outer <= r_inner) {
        std::fprintf(stderr, "singularity_cli: r_outer must exceed r_inner\n");
        return 2;
    }
    const float r_plus = kerr_outer_horizon(M, a);
    const float r_ergo_eq = kerr_ergosphere_outer(M, a, 0.5f * kPi);
    const float rs = 2.0f * M;

    // Temperature anchor: T_ref = 1 at r = r_inner; the image uses the
    // *ratio* T/T_ref, which is unit-agnostic and saturates the sRGB LUT's
    // brightness via intensity_scaling(g).
    const float t_ref = 1.0e4f;  // arbitrary — only flux *ratios* show.

    constexpr int W = 1024;
    constexpr int H = 1024;
    const float view_half = r_outer * 1.2f;

    Image img{W, H, std::vector<uint8_t>(size_t(W) * H * 4, 0)};

    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            const float wx = (float(px) / W * 2.0f - 1.0f) * view_half;
            const float wy = -(float(py) / H * 2.0f - 1.0f) * view_half;
            const float r = std::sqrt(wx * wx + wy * wy);

            uint8_t cr = 0, cg = 0, cb = 0, ca = 255;

            if (r < r_plus) {
                // Inside horizon — pure black. Already initialised to 0.
            } else if (r < r_inner) {
                // Gap between horizon and disc inner edge. Paint the
                // ergosphere boundary in a very subtle near-black for
                // orientation (Kerr case only).
                if (a > 0.01f && std::fabs(r - r_ergo_eq) < 0.02f * M) {
                    cr = cg = cb = 40;
                }
            } else if (r <= r_outer) {
                // Disc emission region. Temperature, Doppler, redshift, LUT.
                const float T = disc_temperature_nt(r, r_inner, t_ref, r_inner);
                // Disc-preview is top-down, so the horizontal-Doppler
                // asymmetry is introduced via cos_psi = wy / r (the
                // component of the orbital velocity toward the camera at
                // the top of the image).
                const float cos_psi = wy / (r + 1e-6f);
                const float g = schwarzschild_combined_shift(rs, r, 1e6f * M, cos_psi);
                const float scale = intensity_scaling(g);
                // Convert the temperature (in kelvin) through a shift factor:
                // observed T scales as g · T_emit.
                const sRGBColor col = blackbody_srgb_tanner_helland(g * T);
                // Brightness proportional to log of flux ratio to keep the
                // dynamic range reasonable; saturate to 1 at inner edge.
                const float flux_ratio = scale * std::pow(r_inner / r, 3.0f);
                const float bright = std::clamp(flux_ratio, 0.05f, 1.0f);
                cr = uint8_t(std::clamp(col.r * bright, 0.0f, 1.0f) * 255.0f);
                cg = uint8_t(std::clamp(col.g * bright, 0.0f, 1.0f) * 255.0f);
                cb = uint8_t(std::clamp(col.b * bright, 0.0f, 1.0f) * 255.0f);
            }
            put_pixel(img, px, py, cr, cg, cb);
            (void)ca;
        }
    }

    if (!stbi_write_png(output_path.c_str(), W, H, 4, img.rgba.data(), W * 4)) {
        std::fprintf(stderr, "singularity_cli: failed to write '%s'\n", output_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d disc preview, a/M=%.3g, "
                "r_inner=%.3gM, r_outer=%.3gM)\n",
                output_path.c_str(),
                W,
                H,
                a / M,
                r_inner / M,
                r_outer / M);
    return 0;
}

// ---------------------------------------------------------------------------
// Photon-sphere orbit mode
// ---------------------------------------------------------------------------
//
// Integrates a single null geodesic initialised on a circular photon orbit
// and dumps its (t, r, θ, φ) trail as CSV. The default configuration
// (r_init = 1.5 r_s, Schwarzschild) is the PHYSICS.md §5.2 photon sphere:
// an unstable equilibrium whose closure within tolerance after one orbital
// period is the sharpest available test of Christoffel-symbol correctness.
// PHYSICS.md §11 test matrix, tolerance 0.5%.

int run_photon_orbit(
    const std::string& output_path, float M, float a, float r_init, float orbits, float h_step) {
    using singularity::physics::kerr_outer_horizon;

    if (M <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: --mass must be positive\n");
        return 2;
    }
    if (a < 0.0f || a > M) {
        std::fprintf(stderr, "singularity_cli: --spin must be in [0, M]\n");
        return 2;
    }
    const float rs = 2.0f * M;
    const float r_plus = (a > 0.0f) ? kerr_outer_horizon(M, a) : rs;
    if (r_init <= r_plus * 1.001f) {
        std::fprintf(
            stderr, "singularity_cli: --r-init %.4g must be > r_+ = %.4g\n", r_init, r_plus);
        return 2;
    }
    if (orbits <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: --orbits must be positive\n");
        return 2;
    }
    if (h_step <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: --h-step must be positive\n");
        return 2;
    }

    // Schwarzschild branch keeps the simpler 8-component State stepper.
    if (a <= 0.0f) {
        const float f = 1.0f - rs / r_init;
        State s{};
        s.t = 0.0f;
        s.r = r_init;
        s.theta = 0.5f * kPi;
        s.phi = 0.0f;
        s.ut = 1.0f;
        s.ur = 0.0f;
        s.utheta = 0.0f;
        s.uphi = std::sqrt(f) / r_init;

        const float period = 2.0f * kPi / s.uphi;
        const int total_steps = int(orbits * period / h_step);
        const int sub_every = std::max(1, int(period / (h_step * 200.0f)));

        FILE* fh = std::fopen(output_path.c_str(), "wb");
        if (!fh) {
            std::fprintf(stderr, "singularity_cli: failed to open '%s'\n", output_path.c_str());
            return 1;
        }
        std::fprintf(fh, "step,t,r,theta,phi\n");
        std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", 0, s.t, s.r, s.theta, s.phi);

        for (int step = 1; step <= total_steps; ++step) {
            s = rk4_step(s, h_step, rs);
            if (!std::isfinite(s.r) || s.r < rs * 1.001f) {
                std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", -1, s.t, s.r, s.theta, s.phi);
                break;
            }
            if ((step % sub_every) == 0 || step == total_steps) {
                std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", step, s.t, s.r, s.theta, s.phi);
            }
        }
        std::fclose(fh);
        std::printf("wrote %s (Schwarzschild photon orbit at r = %.4g M, %.2f orbits)\n",
                    output_path.c_str(),
                    r_init / M,
                    orbits);
        return 0;
    }

    // Kerr branch — use the Hamiltonian-form integrator. Equatorial prograde
    // circular photon orbit at r_init requires L_z/E = b_pro(a, M, r_init)
    // = (r √Δ − 2 M a) / (r − 2 M) per Chandrasekhar §63.
    const float Delta = r_init * r_init - 2.0f * M * r_init + a * a;
    if (Delta <= 0.0f) {
        std::fprintf(
            stderr, "singularity_cli: Δ(%.4g, a=%.4g) ≤ 0 — r_init inside horizon\n", r_init, a);
        return 2;
    }
    const float b_pro = (r_init * std::sqrt(Delta) - 2.0f * M * a) / (r_init - 2.0f * M);

    KerrConserved c{};
    c.E = 1.0f;
    c.L_z = b_pro;  // b_pro = L_z / E with E = 1
    c.Q = 0.0f;
    c.a = a;
    c.M = M;

    KerrHamState s{};
    s.t = 0.0f;
    s.r = r_init;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    s.p_r = 0.0f;
    s.p_theta = 0.0f;

    // Orbital period in affine parameter — use the Hamiltonian dφ/dλ at the
    // starting point since this is a circular orbit (value constant).
    const KerrHamState d0 = kerr_ham_rhs(s, c);
    if (!std::isfinite(d0.phi) || d0.phi <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: bad Kerr init (dφ/dλ = %.3e)\n", d0.phi);
        return 2;
    }
    const float period = 2.0f * kPi / d0.phi;
    const int total_steps = int(orbits * period / h_step);
    const int sub_every = std::max(1, int(period / (h_step * 200.0f)));

    FILE* fh = std::fopen(output_path.c_str(), "wb");
    if (!fh) {
        std::fprintf(stderr, "singularity_cli: failed to open '%s'\n", output_path.c_str());
        return 1;
    }
    std::fprintf(fh, "step,t,r,theta,phi\n");
    std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", 0, s.t, s.r, s.theta, s.phi);

    for (int step = 1; step <= total_steps; ++step) {
        s = kerr_ham_rk4_step(s, h_step, c);
        if (!std::isfinite(s.r) || s.r < r_plus * 1.001f) {
            std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", -1, s.t, s.r, s.theta, s.phi);
            break;
        }
        if ((step % sub_every) == 0 || step == total_steps) {
            std::fprintf(fh, "%d,%.9g,%.9g,%.9g,%.9g\n", step, s.t, s.r, s.theta, s.phi);
        }
    }
    std::fclose(fh);
    std::printf("wrote %s (Kerr prograde photon orbit at r = %.4g M, a/M = %.3g, %.2f orbits)\n",
                output_path.c_str(),
                r_init / M,
                a / M,
                orbits);
    return 0;
}

// ---------------------------------------------------------------------------
// Kerr-geometry mode
// ---------------------------------------------------------------------------
//
// Dumps every closed-form scalar in core/include/physics/kerr.hpp as a flat
// JSON object. Drives verification/test_kerr_geometry.py, which reads this
// back and cross-checks against independently-computed double-precision
// reference values. No integrator, no rendering — just serialised analytic
// truth.

int run_kerr_geometry(const std::string& output_path, float M, float a) {
    using namespace singularity::physics;

    if (M <= 0.0f) {
        std::fprintf(stderr, "singularity_cli: --mass must be positive\n");
        return 2;
    }
    if (a < 0.0f || a > M) {
        std::fprintf(stderr, "singularity_cli: --spin %.6f out of range [0, M=%.6f]\n", a, M);
        return 2;
    }

    const float r_plus = kerr_outer_horizon(M, a);
    const float r_minus = kerr_inner_horizon(M, a);
    const float r_ergo_pole = kerr_ergosphere_outer(M, a, 0.0f);
    const float r_ergo_eq = kerr_ergosphere_outer(M, a, 0.5f * kPi);
    const float r_ph_pro = kerr_photon_sphere_prograde(M, a);
    const float r_ph_ret = kerr_photon_sphere_retrograde(M, a);
    const float r_isco_pro = kerr_isco_prograde(M, a);
    const float r_isco_ret = kerr_isco_retrograde(M, a);

    FILE* f = std::fopen(output_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "singularity_cli: failed to open '%s'\n", output_path.c_str());
        return 1;
    }
    // Hand-written JSON keeps the CLI dependency-free: nlohmann/json isn't
    // vendored and pulling in a whole library for eight numbers would be
    // unprincipled. %.9g preserves every bit of float32 round-trip.
    std::fprintf(f,
                 "{\n"
                 "  \"mass\": %.9g,\n"
                 "  \"spin\": %.9g,\n"
                 "  \"spin_over_mass\": %.9g,\n"
                 "  \"outer_horizon\": %.9g,\n"
                 "  \"inner_horizon\": %.9g,\n"
                 "  \"ergosphere_polar\": %.9g,\n"
                 "  \"ergosphere_equatorial\": %.9g,\n"
                 "  \"photon_sphere_prograde\": %.9g,\n"
                 "  \"photon_sphere_retrograde\": %.9g,\n"
                 "  \"isco_prograde\": %.9g,\n"
                 "  \"isco_retrograde\": %.9g\n"
                 "}\n",
                 M,
                 a,
                 a / M,
                 r_plus,
                 r_minus,
                 r_ergo_pole,
                 r_ergo_eq,
                 r_ph_pro,
                 r_ph_ret,
                 r_isco_pro,
                 r_isco_ret);
    std::fclose(f);
    std::printf("wrote %s (Kerr geometry at a/M = %.6g)\n", output_path.c_str(), a / M);
    return 0;
}

// ---------------------------------------------------------------------------
// Benchmark mode — deterministic integrator timing for CI regression tracking
// ---------------------------------------------------------------------------
//
// Emits a single JSON line on stdout; no image / file output. The step count
// is not fixed at n_rays * max_steps because rays that hit the horizon or
// diverge to inf terminate early — `total_steps` reports the actual count so
// downstream tooling can compute the true per-step cost.

int run_benchmark(const TwoDToyConfig& cfg, const std::string& metric, float spin) {
    using singularity::physics::kerr_outer_horizon;

    if (metric != "schw" && metric != "kerr") {
        std::fprintf(stderr, "singularity_cli: --metric must be 'schw' or 'kerr'\n");
        return 2;
    }
    // Normalize M = 1 so benchmark timings are reproducible regardless of
    // --mass; --spin is interpreted as a/M and must be in [0, 1) for Kerr
    // (a = M is extremal and carries coordinate subtleties we dodge here).
    constexpr float M = 1.0f;
    constexpr float rs = 2.0f * M;
    constexpr float h_step = 0.1f;
    const float a = (metric == "kerr") ? spin * M : 0.0f;
    if (metric == "kerr" && (a < 0.0f || a >= 1.0f * M)) {
        std::fprintf(stderr, "singularity_cli: --spin must be in [0, 1) for kerr benchmark\n");
        return 2;
    }
    const float b_min = cfg.b_min * M;
    const float b_max = cfg.b_max * M;
    const float x_start = std::max(50.0f * M, 2.0f * std::max(std::fabs(b_min), std::fabs(b_max)));

    long long total_steps = 0;
    const auto t0 = std::chrono::steady_clock::now();

    if (metric == "schw") {
        constexpr float horizon_cut = 1.01f * rs;
        for (int i = 0; i < cfg.n_rays; ++i) {
            const float b = b_min + (b_max - b_min) * (float(i) + 0.5f) / float(cfg.n_rays);
            const float r0 = std::sqrt(x_start * x_start + b * b);
            const float phi0 = std::atan2(b, x_start);
            const float ur = -std::cos(phi0);
            const float uphi = std::sin(phi0) / r0;
            const float f = 1.0f - rs / r0;
            const float ut = std::sqrt(((ur * ur) / f + r0 * r0 * uphi * uphi) / f);

            State s{};
            s.t = 0.0f;
            s.r = r0;
            s.theta = 0.5f * kPi;
            s.phi = phi0;
            s.ut = ut;
            s.ur = ur;
            s.utheta = 0.0f;
            s.uphi = uphi;

            for (int step = 0; step < cfg.max_steps; ++step) {
                if (!std::isfinite(s.r) || !std::isfinite(s.phi))
                    break;
                if (s.r < horizon_cut)
                    break;
                s = rk4_step(s, h_step, rs);
                ++total_steps;
            }
        }
    } else {
        const float r_plus = kerr_outer_horizon(M, a);
        const float horizon_cut = 1.02f * r_plus;
        for (int i = 0; i < cfg.n_rays; ++i) {
            const float b = b_min + (b_max - b_min) * (float(i) + 0.5f) / float(cfg.n_rays);
            const float r0 = std::sqrt(x_start * x_start + b * b);
            const float phi0 = std::atan2(b, x_start);
            const float ur = -std::cos(phi0);
            const float uphi = std::sin(phi0) / r0;

            // Equatorial Kerr metric components at (r0, θ = π/2), null-
            // condition quadratic in u^t; future-directed root gives us a
            // valid photon 4-velocity that we convert to (E, L_z, Q = 0).
            const float Delta = r0 * r0 - 2.0f * M * r0 + a * a;
            const float g_tt = -(1.0f - 2.0f * M / r0);
            const float g_tp = -2.0f * M * a / r0;
            const float g_rr = r0 * r0 / Delta;
            const float g_pp = ((r0 * r0 + a * a) * (r0 * r0 + a * a) - a * a * Delta) / (r0 * r0);
            const float A = g_tt;
            const float B = 2.0f * g_tp * uphi;
            const float C = g_rr * ur * ur + g_pp * uphi * uphi;
            const float disc = B * B - 4.0f * A * C;
            if (disc < 0.0f)
                continue;
            const float ut = (-B - std::sqrt(disc)) / (2.0f * A);

            KerrConserved c{};
            c.E = -(g_tt * ut + g_tp * uphi);
            c.L_z = g_tp * ut + g_pp * uphi;
            c.Q = 0.0f;
            c.a = a;
            c.M = M;

            KerrHamState s{};
            s.t = 0.0f;
            s.r = r0;
            s.theta = 0.5f * kPi;
            s.phi = phi0;
            kerr_ham_momenta_from_velocities(&s, ur, /*u_theta=*/0.0f, c);

            for (int step = 0; step < cfg.max_steps; ++step) {
                if (!std::isfinite(s.r) || !std::isfinite(s.phi))
                    break;
                if (s.r < horizon_cut)
                    break;
                s = kerr_ham_rk4_step(s, h_step, c);
                ++total_steps;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double steps_per_s = (ms > 0.0) ? double(total_steps) / (ms * 1e-3) : 0.0;
    const double mean_step_ns = (total_steps > 0) ? ms * 1e6 / double(total_steps) : 0.0;

    std::printf("{\"mode\":\"benchmark\","
                "\"metric\":\"%s\","
                "\"spin\":%.6f,"
                "\"n_rays\":%d,"
                "\"max_steps\":%d,"
                "\"total_steps\":%lld,"
                "\"total_ms\":%.3f,"
                "\"steps_per_s\":%.1f,"
                "\"mean_step_ns\":%.1f}\n",
                metric.c_str(),
                a,
                cfg.n_rays,
                cfg.max_steps,
                total_steps,
                ms,
                steps_per_s,
                mean_step_ns);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode;
    std::string output_path;
    std::string dump_trails_path;
    float spin = 0.0f;
    float mass = 1.0f;
    float photon_r_init = -1.0f;  // -1 = auto (Schwarzschild 3M or Kerr r_ph_pro)
    float photon_orbits = 1.0f;
    float photon_h_step = 0.01f;
    float disc_r_inner = -1.0f;  // -1 = default to ISCO
    float disc_r_outer = 20.0f;
    int cpu_render_w = 256;
    int cpu_render_h = 256;
    int cpu_supersample = 1;
    float cpu_camera_distance = 30.0f;   // M
    float cpu_camera_elevation = 0.15f;  // rad
    float cpu_camera_fov = 55.0f;        // deg
    std::string bench_metric = "schw";
    TwoDToyConfig toy_cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--help" || a == "-h") {
            print_help();
            return 0;
        }
        if (a == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--dump-trails" && i + 1 < argc) {
            dump_trails_path = argv[++i];
        } else if (a == "--spin" && i + 1 < argc) {
            spin = std::strtof(argv[++i], nullptr);
        } else if (a == "--mass" && i + 1 < argc) {
            mass = std::strtof(argv[++i], nullptr);
        } else if (a == "--b-range" && i + 2 < argc) {
            toy_cfg.b_min = std::strtof(argv[++i], nullptr);
            toy_cfg.b_max = std::strtof(argv[++i], nullptr);
            if (toy_cfg.b_min >= toy_cfg.b_max) {
                std::fprintf(stderr, "singularity_cli: --b-range MIN MAX requires MIN < MAX\n");
                return 2;
            }
        } else if (a == "--n-rays" && i + 1 < argc) {
            toy_cfg.n_rays = int(std::strtol(argv[++i], nullptr, 10));
            if (toy_cfg.n_rays < 2) {
                std::fprintf(stderr, "singularity_cli: --n-rays must be >= 2\n");
                return 2;
            }
        } else if (a == "--max-steps" && i + 1 < argc) {
            toy_cfg.max_steps = int(std::strtol(argv[++i], nullptr, 10));
            if (toy_cfg.max_steps < 100) {
                std::fprintf(stderr, "singularity_cli: --max-steps must be >= 100\n");
                return 2;
            }
        } else if (a == "--r-init" && i + 1 < argc) {
            photon_r_init = std::strtof(argv[++i], nullptr);
        } else if (a == "--orbits" && i + 1 < argc) {
            photon_orbits = std::strtof(argv[++i], nullptr);
        } else if (a == "--h-step" && i + 1 < argc) {
            photon_h_step = std::strtof(argv[++i], nullptr);
        } else if (a == "--r-inner" && i + 1 < argc) {
            disc_r_inner = std::strtof(argv[++i], nullptr);
        } else if (a == "--r-outer" && i + 1 < argc) {
            disc_r_outer = std::strtof(argv[++i], nullptr);
        } else if (a == "--resolution" && i + 1 < argc) {
            // WxH; parse both.
            const char* arg = argv[++i];
            const char* x = std::strchr(arg, 'x');
            if (!x) {
                std::fprintf(stderr, "singularity_cli: --resolution expects WxH, got %s\n", arg);
                return 2;
            }
            cpu_render_w = int(std::strtol(arg, nullptr, 10));
            cpu_render_h = int(std::strtol(x + 1, nullptr, 10));
            if (cpu_render_w < 4 || cpu_render_h < 4) {
                std::fprintf(stderr, "singularity_cli: --resolution too small\n");
                return 2;
            }
        } else if (a == "--camera-distance" && i + 1 < argc) {
            cpu_camera_distance = std::strtof(argv[++i], nullptr);
        } else if (a == "--camera-elevation" && i + 1 < argc) {
            cpu_camera_elevation = std::strtof(argv[++i], nullptr);
        } else if (a == "--camera-fov" && i + 1 < argc) {
            cpu_camera_fov = std::strtof(argv[++i], nullptr);
        } else if (a == "--metric" && i + 1 < argc) {
            bench_metric = argv[++i];
        } else if (a == "--supersample" && i + 1 < argc) {
            cpu_supersample = int(std::strtol(argv[++i], nullptr, 10));
            if (cpu_supersample < 1 || cpu_supersample > 8) {
                std::fprintf(stderr, "singularity_cli: --supersample must be 1..8\n");
                return 2;
            }
        } else if (a == "--scene" && i + 1 < argc) {
            // Load a scene-config text file and apply its values as the new
            // defaults. Subsequent --mass / --spin / etc. flags on the same
            // command line will still override what the scene set, which is
            // the standard "config file then per-invocation overrides"
            // ergonomic.
            singularity::scene::SceneConfig scene;
            const auto res = singularity::scene::load_scene_config_from_file(argv[++i], scene);
            using S = singularity::scene::SceneLoadStatus;
            if (res.status != S::Ok) {
                std::fprintf(stderr,
                             "singularity_cli: --scene load failed (%s, line %d): %s\n",
                             res.status == S::FileNotFound ? "file-not-found"
                             : res.status == S::UnknownKey ? "unknown-key"
                                                           : "parse-error",
                             res.line_number,
                             res.error_line.c_str());
                return 2;
            }
            mass = scene.mass;
            spin = scene.spin;
            disc_r_inner = scene.disc_r_inner;
            disc_r_outer = scene.disc_r_outer;
            toy_cfg.max_steps = scene.max_steps;
            photon_h_step = scene.h_step;
            cpu_camera_distance = scene.camera_distance;
            cpu_camera_elevation = scene.camera_elevation;
            cpu_camera_fov = scene.camera_fov_deg;
        } else {
            std::fprintf(
                stderr, "singularity_cli: unknown argument: %.*s\n", int(a.size()), a.data());
            print_help();
            return 2;
        }
    }

    if (mode.empty()) {
        print_help();
        return 0;
    }

    if (mode == "2d-toy") {
        if (output_path.empty())
            output_path = "phase1_rays.png";
        return run_2d_toy(output_path, dump_trails_path, toy_cfg);
    }

    if (mode == "kerr-2d-toy") {
        if (output_path.empty())
            output_path = "kerr_rays.png";
        return run_kerr_2d_toy(output_path, dump_trails_path, toy_cfg, mass, spin);
    }

    if (mode == "kerr-geometry") {
        if (output_path.empty())
            output_path = "kerr_geometry.json";
        return run_kerr_geometry(output_path, mass, spin);
    }

    if (mode == "photon-orbit") {
        if (output_path.empty())
            output_path = "photon_orbit.csv";
        // Auto-select the photon sphere when the caller didn't specify
        // --r-init: Schwarzschild 3 M at spin = 0, prograde Kerr photon
        // sphere otherwise.
        float r_init = photon_r_init;
        if (r_init < 0.0f) {
            r_init = (spin <= 0.0f) ? 3.0f * mass
                                    : singularity::physics::kerr_photon_sphere_prograde(mass, spin);
        }
        return run_photon_orbit(output_path, mass, spin, r_init, photon_orbits, photon_h_step);
    }

    if (mode == "disc-preview") {
        if (output_path.empty())
            output_path = "disc_preview.png";
        return run_disc_preview(output_path, mass, spin, disc_r_inner, disc_r_outer);
    }

    if (mode == "cpu-render") {
        if (output_path.empty())
            output_path = "cpu_render.png";
        return run_cpu_render(output_path,
                              cpu_render_w,
                              cpu_render_h,
                              cpu_supersample,
                              mass,
                              disc_r_inner,
                              disc_r_outer,
                              cpu_camera_distance,
                              cpu_camera_elevation,
                              cpu_camera_fov);
    }

    if (mode == "kerr-cpu-render") {
        if (output_path.empty())
            output_path = "kerr_cpu_render.png";
        return run_kerr_cpu_render(output_path,
                                   cpu_render_w,
                                   cpu_render_h,
                                   cpu_supersample,
                                   mass,
                                   spin,
                                   disc_r_inner,
                                   disc_r_outer,
                                   cpu_camera_distance,
                                   cpu_camera_elevation,
                                   cpu_camera_fov);
    }

    if (mode == "benchmark") {
        return run_benchmark(toy_cfg, bench_metric, spin);
    }

    std::fprintf(stderr, "singularity_cli: unknown mode '%s'\n", mode.c_str());
    return 2;
}
