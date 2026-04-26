// app/app_shell.hpp
//
// Cross-platform (Mac + Windows) desktop shell. Owns the SDL3 window, the
// render backend, and the per-frame camera/scene state. Deliberately a
// plain struct + free functions so it's trivially testable and the platform
// main can be tiny.

#pragma once

#include <memory>
#include <string>

#include "render_backend.hpp"

struct SDL_Window;

namespace singularity::app {

// Orbital camera controller — azimuth, elevation, radial distance. Matches
// the CPU `run_cpu_render` orbital basis so images from both paths read the
// same at default parameters.
struct OrbitalCamera {
    float distance_M = 30.0f;  // radial distance in M
    // Default azimuth set off the φ=0 meridian so the BL coordinate seam
    // (where `atan2(y, x)` flips sign and numerical integration gets
    // noisy) runs through the corner of the frame rather than straight
    // down the middle.
    float azimuth_rad = 0.7853981633974483f;  // π/4 ≈ 45°
    // Nearly edge-on — the iconic Interstellar viewing angle. The disc
    // wraps prominently over the BH shadow; far side lenses up into the
    // halo arc that made Gargantua a household image.
    float elevation_rad = 0.05f;
    float fov_y_deg = 55.0f;
    bool dragging = false;
    float drag_sens = 0.004f;  // rad / px
    float zoom_sens = 0.85f;   // exponential

    // Writes the Cartesian position + orthonormal basis into `state`. The
    // convention matches render_backend.hpp: basis is row-major,
    //   row 0 = right, row 1 = up, row 2 = -forward.
    void write_state(CameraState& state) const;
};

struct AppShellConfig {
    int width = 1280;
    int height = 720;
    bool vsync = true;
    bool start_kerr = false;
    float start_spin_a_over_M = 0.0f;

    // Headless capture mode. When `capture_path` is non-empty the app does
    // not open a window; it renders one frame off-screen, writes a PNG
    // at that path, and exits with 0. Used for golden-image regression
    // against the CPU reference in `singularity_cli cpu-render`.
    std::string capture_path;
    int capture_supersample = 2;
    float capture_distance_M = 30.0f;
    float capture_elevation_rad = 0.15f;
    float capture_spin_a_over_M = 0.0f;
};

// Runs the main loop until the user closes the window. Returns the
// process exit code (0 = clean shutdown).
int run(const AppShellConfig& cfg);

// Headless render — no SDL window. Writes `cfg.capture_path`. Returns
// 0 on success.
int run_capture(const AppShellConfig& cfg);

}  // namespace singularity::app
