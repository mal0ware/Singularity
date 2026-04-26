// app/app_shell.cpp
//
// Main loop + camera controller + ImGui settings panel. Kept in a single
// translation unit because everything here is bound tightly to the
// event->render cadence — splitting into three tiny TUs adds noise
// without clarifying anything.
//
// Platform split:
//   __APPLE__     -> Metal backend + ImGui Metal backend + SDL_metal surface
//   otherwise     -> Vulkan backend + ImGui Vulkan backend + SDL Vulkan surface
//
// The two paths are kept as #ifdef islands in the same file so the shared
// glue (camera math, ImGui panel, event loop structure) is obviously
// unified. Per-file language override in app/CMakeLists.txt compiles
// this as Objective-C++ on Apple, plain C++ everywhere else — so the
// platform-specific islands can use the syntax each one needs.

#include "app_shell.hpp"

#include <SDL3/SDL.h>

#include "backends/imgui_impl_sdl3.h"
#include "imgui.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "stb_image_write.h"

#if defined(__APPLE__)
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <SDL3/SDL_metal.h>

#include "backends/imgui_impl_metal.h"
#include "render/metal/metal_backend.hpp"
#else
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "backends/imgui_impl_vulkan.h"
#include "render/vulkan/vulkan_backend.hpp"
#endif

namespace singularity::app {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Rebuild a basis vector triple from (distance, azimuth, elevation) and the
// orbit center at origin. Produces a right-handed basis with the convention
// row 0 = right, row 1 = up, row 2 = -forward.
void compute_basis(const OrbitalCamera& cam, float pos[3], float basis[9]) {
    const float ce = std::cos(cam.elevation_rad);
    const float se = std::sin(cam.elevation_rad);
    const float ca = std::cos(cam.azimuth_rad);
    const float sa = std::sin(cam.azimuth_rad);

    pos[0] = cam.distance_M * ce * ca;
    pos[1] = cam.distance_M * ce * sa;
    pos[2] = cam.distance_M * se;

    const float inv = 1.0f / cam.distance_M;
    const float fwd[3] = {-pos[0] * inv, -pos[1] * inv, -pos[2] * inv};

    float rt[3] = {fwd[1], -fwd[0], 0.0f};
    const float rt_len = std::sqrt(rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2]);
    const float inv_r = (rt_len > 1e-6f) ? 1.0f / rt_len : 1.0f;
    rt[0] *= inv_r;
    rt[1] *= inv_r;
    rt[2] *= inv_r;

    const float up[3] = {
        fwd[1] * rt[2] - fwd[2] * rt[1],
        fwd[2] * rt[0] - fwd[0] * rt[2],
        fwd[0] * rt[1] - fwd[1] * rt[0],
    };

    basis[0] = rt[0];
    basis[1] = rt[1];
    basis[2] = rt[2];
    basis[3] = up[0];
    basis[4] = up[1];
    basis[5] = up[2];
    basis[6] = -fwd[0];
    basis[7] = -fwd[1];
    basis[8] = -fwd[2];
}

}  // namespace

void OrbitalCamera::write_state(CameraState& state) const {
    compute_basis(*this, state.position, state.basis);
    state.fov_y_radians = fov_y_deg * (kPi / 180.0f);
}

namespace {

void apply_drag(OrbitalCamera& cam, float dx, float dy) {
    cam.azimuth_rad -= dx * cam.drag_sens;
    cam.elevation_rad += dy * cam.drag_sens;
    const float lim = 0.49f * kPi;
    if (cam.elevation_rad > lim)
        cam.elevation_rad = lim;
    if (cam.elevation_rad < -lim)
        cam.elevation_rad = -lim;
}

void apply_zoom(OrbitalCamera& cam, float wheel_y) {
    cam.distance_M *= std::pow(cam.zoom_sens, wheel_y);
    if (cam.distance_M < 4.0f)
        cam.distance_M = 4.0f;
    if (cam.distance_M > 500.0f)
        cam.distance_M = 500.0f;
}

// Per-user config directory for settings.conf. Standard locations:
//   Linux   $XDG_CONFIG_HOME/singularity  (else ~/.config/singularity)
//   macOS   ~/Library/Application Support/Singularity
//   Windows %APPDATA%\Singularity         (== %USERPROFILE%\AppData\Roaming)
// Returns empty path if none of the expected env vars are defined.
std::filesystem::path settings_path() {
#if defined(_WIN32)
    if (const char* ad = std::getenv("APPDATA")) {
        return std::filesystem::path(ad) / "Singularity" / "settings.conf";
    }
    return {};
#elif defined(__APPLE__)
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / "Library" / "Application Support" / "Singularity"
               / "settings.conf";
    }
    return {};
#else
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && *xdg) {
        return std::filesystem::path(xdg) / "singularity" / "settings.conf";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".config" / "singularity" / "settings.conf";
    }
    return {};
#endif
}

void settings_trim(std::string& s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// Write Scene + OrbitalCamera to a flat `key = value` file. Format matches
// core/include/scene/scene_config.hpp so the same parser could be taught to
// consume it; extra keys here just mean the app has a richer UI than the
// headless CLI. Returns false if the directory couldn't be created or the
// write failed.
bool save_settings(const std::filesystem::path& path,
                   const Scene& scene,
                   const OrbitalCamera& cam) {
    if (path.empty())
        return false;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
        return false;
    std::ofstream f(path, std::ios::trunc);
    if (!f)
        return false;
    f << "# Singularity app settings — written on app exit.\n";
    f << "metric = " << (scene.metric == Scene::MetricType::Kerr ? "kerr" : "schwarzschild")
      << "\n";
    f << "mass_solar = " << scene.mass_solar << "\n";
    f << "spin_a_over_M = " << scene.spin_a_over_M << "\n";
    f << "disc_inner_M = " << scene.disc_inner_M << "\n";
    f << "disc_outer_M = " << scene.disc_outer_M << "\n";
    f << "disc_doppler_on = " << (scene.disc_doppler_on ? 1 : 0) << "\n";
    f << "disc_redshift_on = " << (scene.disc_redshift_on ? 1 : 0) << "\n";
    f << "disc_texture_on = " << (scene.disc_texture_on ? 1 : 0) << "\n";
    f << "render_supersample = " << scene.render_supersample << "\n";
    f << "exposure = " << scene.exposure << "\n";
    f << "bloom_threshold = " << scene.bloom_threshold << "\n";
    f << "bloom_strength = " << scene.bloom_strength << "\n";
    f << "disc_turbulence = " << scene.disc_turbulence << "\n";
    f << "disc_peak_T_K = " << scene.disc_peak_T_K << "\n";
    f << "cam_distance_M = " << cam.distance_M << "\n";
    f << "cam_azimuth_rad = " << cam.azimuth_rad << "\n";
    f << "cam_elevation_rad = " << cam.elevation_rad << "\n";
    f << "cam_fov_y_deg = " << cam.fov_y_deg << "\n";
    return f.good();
}

// Mutates scene/cam in place for every recognised key; missing keys keep
// their current (defaulted) values. Unknown keys are ignored so older saves
// from future builds don't bail the load.
bool load_settings(const std::filesystem::path& path, Scene& scene, OrbitalCamera& cam) {
    if (path.empty())
        return false;
    std::ifstream f(path);
    if (!f)
        return false;
    std::string line;
    while (std::getline(f, line)) {
        if (auto h = line.find('#'); h != std::string::npos)
            line.erase(h);
        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        settings_trim(key);
        settings_trim(val);
        if (key.empty() || val.empty())
            continue;
        auto to_bool = [](const std::string& v) {
            return v == "1" || v == "true" || v == "yes" || v == "on";
        };
        auto to_float = [](const std::string& v) { return std::strtof(v.c_str(), nullptr); };
        auto to_uint = [](const std::string& v) {
            return (std::uint32_t)std::strtoul(v.c_str(), nullptr, 10);
        };
        if (key == "metric") {
            scene.metric =
                (val == "kerr") ? Scene::MetricType::Kerr : Scene::MetricType::Schwarzschild;
        } else if (key == "mass_solar")
            scene.mass_solar = to_float(val);
        else if (key == "spin_a_over_M")
            scene.spin_a_over_M = to_float(val);
        else if (key == "disc_inner_M")
            scene.disc_inner_M = to_float(val);
        else if (key == "disc_outer_M")
            scene.disc_outer_M = to_float(val);
        else if (key == "disc_doppler_on")
            scene.disc_doppler_on = to_bool(val);
        else if (key == "disc_redshift_on")
            scene.disc_redshift_on = to_bool(val);
        else if (key == "disc_texture_on")
            scene.disc_texture_on = to_bool(val);
        else if (key == "render_supersample")
            scene.render_supersample = to_uint(val);
        else if (key == "exposure")
            scene.exposure = to_float(val);
        else if (key == "bloom_threshold")
            scene.bloom_threshold = to_float(val);
        else if (key == "bloom_strength")
            scene.bloom_strength = to_float(val);
        else if (key == "disc_turbulence")
            scene.disc_turbulence = to_float(val);
        else if (key == "disc_peak_T_K")
            scene.disc_peak_T_K = to_float(val);
        else if (key == "cam_distance_M")
            cam.distance_M = to_float(val);
        else if (key == "cam_azimuth_rad")
            cam.azimuth_rad = to_float(val);
        else if (key == "cam_elevation_rad")
            cam.elevation_rad = to_float(val);
        else if (key == "cam_fov_y_deg")
            cam.fov_y_deg = to_float(val);
    }
    return true;
}

// User's Pictures folder per platform, or empty if we couldn't figure it out.
// Stays off SHGetKnownFolderPath/ShlObj.h on Windows — the %USERPROFILE%
// fallback is portable-enough for a screenshot dumping ground.
std::filesystem::path pictures_dir() {
#if defined(_WIN32)
    if (const char* up = std::getenv("USERPROFILE")) {
        return std::filesystem::path(up) / "Pictures" / "Singularity";
    }
    return {};
#else
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / "Pictures" / "Singularity";
    }
    return {};
#endif
}

// "singularity-YYYYMMDD-HHMMSS.png" filename.
std::string timestamp_png_filename() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return std::string("singularity-") + buf + ".png";
}

// Write an ImageData to disk via stb_image_write. Returns the final path on
// success, empty path on any failure (directory creation / write).
std::filesystem::path save_screenshot(const ImageData& img) {
    auto dir = pictures_dir();
    if (dir.empty())
        return {};
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
        return {};
    const auto path = dir / timestamp_png_filename();
    const int rc = stbi_write_png(path.string().c_str(),
                                  (int)img.width,
                                  (int)img.height,
                                  4,
                                  img.pixels_rgba.data(),
                                  (int)img.width * 4);
    return rc ? path : std::filesystem::path{};
}

// Scene + camera preset bundles. Selecting one swaps both at once so the
// user gets a coherent "look" rather than having to tune sliders to find
// it. Defined inline in the panel so the user-facing strings live next
// to the values they apply.
struct ScenePreset {
    const char* label;
    Scene scene;
    OrbitalCamera cam;
};

void apply_preset(int index, Scene& scene, OrbitalCamera& cam) {
    auto preserve_runtime = [&](Scene& s) {
        // Don't clobber per-frame ephemeral state when swapping presets —
        // SPP and the quality override stay where the user left them so a
        // preset switch doesn't surprise the framerate.
        s.render_supersample = scene.render_supersample;
        s.max_steps = scene.max_steps;
        s.h_step = scene.h_step;
    };
    switch (index) {
    case 0: {       // Interstellar — Kerr a=0.6, edge-on, warm orange disc.
        Scene s{};  // Defaults already match Interstellar look.
        preserve_runtime(s);
        scene = s;
        cam = OrbitalCamera{};
        cam.elevation_rad = 0.05f;
        break;
    }
    case 1: {  // Schwarzschild reference — symmetric, ISCO inner.
        Scene s{};
        s.metric = Scene::MetricType::Schwarzschild;
        s.spin_a_over_M = 0.0f;
        s.disc_inner_M = 6.0f;  // Schwarzschild ISCO.
        s.disc_outer_M = 30.0f;
        preserve_runtime(s);
        scene = s;
        cam = OrbitalCamera{};
        cam.elevation_rad = 0.15f;
        break;
    }
    case 2: {  // Near-extremal Kerr — flat-sided shadow, Doppler punch.
        Scene s{};
        s.metric = Scene::MetricType::Kerr;
        s.spin_a_over_M = 0.94f;
        s.disc_inner_M = 1.7f;  // ISCO at a/M=0.94, prograde.
        s.disc_outer_M = 25.0f;
        s.disc_peak_T_K = 5500.0f;
        preserve_runtime(s);
        scene = s;
        cam = OrbitalCamera{};
        cam.elevation_rad = 0.05f;
        break;
    }
    case 3: {  // Realistic Sgr A* — moderate spin, cooler disc.
        Scene s{};
        s.metric = Scene::MetricType::Kerr;
        s.spin_a_over_M = 0.5f;
        s.disc_inner_M = 4.23f;  // ISCO at a/M=0.5, prograde.
        s.disc_outer_M = 20.0f;
        s.disc_peak_T_K = 3500.0f;  // Sgr A* runs cool by stellar standards.
        s.disc_turbulence = 0.65f;
        preserve_runtime(s);
        scene = s;
        cam = OrbitalCamera{};
        cam.elevation_rad = 0.18f;  // Tilted ~10° — typical EHT-style view.
        cam.distance_M = 35.0f;
        break;
    }
    case 4: {       // Cinematic close-up — bigger BH on screen, more drama.
        Scene s{};  // Inherits Interstellar defaults.
        preserve_runtime(s);
        scene = s;
        cam = OrbitalCamera{};
        cam.elevation_rad = 0.05f;
        cam.distance_M = 15.0f;
        cam.fov_y_deg = 70.0f;
        break;
    }
    default:
        break;
    }
}

void apply_quality(int q, Scene& scene) {
    // 0 = Draft (snappy slider drags), 1 = Default (backend constant), 2 = Quality.
    switch (q) {
    case 0:
        scene.max_steps = 800;
        scene.h_step = 0.6f;
        break;
    case 1:
        scene.max_steps = 0;  // backend default kicks in.
        scene.h_step = 0.0f;
        break;
    case 2:
        scene.max_steps = 4000;
        scene.h_step = 0.25f;
        break;
    default:
        break;
    }
}

// Hover-tooltip helper. ImGui::Text + SameLine(0) + tiny "(?)" then the
// hover handler — keeps the panel readable without exploding into
// per-control wrapper functions.
void tip(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void imgui_settings_panel(Scene& scene,
                          OrbitalCamera& cam,
                          float fps,
                          double gpu_seconds,
                          bool& screenshot_requested,
                          const std::string& last_screenshot_path) {
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 200), ImVec2(640, 1200));
    ImGui::Begin("Singularity");

    // --- Performance readout ----------------------------------------------
    const float gpu_ms = float(gpu_seconds) * 1000.0f;
    ImGui::Text("%.0f FPS", fps);
    ImGui::SameLine();
    ImGui::TextDisabled(
        "|  cpu %.1f ms   gpu %.1f ms", 1000.0f / (fps > 0.0f ? fps : 1.0f), gpu_ms);
    if (gpu_ms > 33.0f) {
        // Soft hint at >30 ms — nudges the user toward Draft quality
        // before they think the app is broken.
        ImGui::TextColored(ImVec4(0.95f, 0.7f, 0.2f, 1.0f), "Slow frame — try Quality: Draft.");
    }
    ImGui::Separator();

    // --- Presets ----------------------------------------------------------
    static int preset_idx = 0;
    const char* preset_items[] = {
        "Interstellar (Kerr a=0.6)",
        "Schwarzschild reference",
        "Near-extremal Kerr (a=0.94)",
        "Realistic Sgr A*",
        "Cinematic close-up",
    };
    if (ImGui::Combo("Preset", &preset_idx, preset_items, IM_ARRAYSIZE(preset_items))) {
        apply_preset(preset_idx, scene, cam);
    }
    tip("Pre-tuned scene + camera bundles. Selecting one overwrites all "
        "the sliders below — except Quality and Supersample, which stay "
        "wherever you set them.");

    static int quality_idx = 1;
    const char* quality_items[] = {"Draft (fast)", "Default", "Quality (slow)"};
    if (ImGui::Combo("Quality", &quality_idx, quality_items, IM_ARRAYSIZE(quality_items))) {
        apply_quality(quality_idx, scene);
    }
    tip("Trades visible aliasing for framerate. Draft halves integrator "
        "step count and doubles the affine step — ~3× faster, slightly "
        "blurrier shadow rim. Default matches the headless capture path. "
        "Quality is for export-grade stills.");

    ImGui::Spacing();

    // --- Black hole -------------------------------------------------------
    if (ImGui::CollapsingHeader("Black hole", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* metric_items[] = {"Schwarzschild (no spin)", "Kerr (rotating)"};
        int metric_idx = scene.metric == Scene::MetricType::Kerr ? 1 : 0;
        if (ImGui::Combo("Metric", &metric_idx, metric_items, 2)) {
            scene.metric =
                metric_idx == 1 ? Scene::MetricType::Kerr : Scene::MetricType::Schwarzschild;
        }
        if (scene.metric == Scene::MetricType::Kerr) {
            ImGui::SliderFloat("Spin a/M", &scene.spin_a_over_M, 0.0f, 0.998f, "%.3f");
            tip("Dimensionless rotation. 0 = Schwarzschild, 1 = extremal "
                "Kerr (theoretical max). Real astrophysical BHs cluster "
                "around 0.5–0.95.");
        }
        ImGui::SliderFloat(
            "Mass (M)", &scene.mass_solar, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        tip("Geometric units — a multiplicative scale on the entire scene. "
            "Larger M makes everything bigger and the disc cooler at the "
            "same radius (T ∝ 1/√M).");
    }

    // --- Accretion disc ---------------------------------------------------
    if (ImGui::CollapsingHeader("Accretion disc", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Visible", &scene.disc_texture_on);
        ImGui::BeginDisabled(!scene.disc_texture_on);
        ImGui::SliderFloat("Inner radius (M)", &scene.disc_inner_M, 1.5f, 30.0f, "%.2f");
        tip("Inner edge of the disc, in units of M. Use ISCO for the spin "
            "you've picked: 6.0 (Schw), 4.23 (a=0.5), 3.83 (a=0.6), "
            "1.7 (a=0.94 prograde).");
        ImGui::SliderFloat(
            "Outer radius (M)", &scene.disc_outer_M, scene.disc_inner_M + 0.5f, 60.0f, "%.1f");
        ImGui::SliderFloat("Peak temperature (K)",
                           &scene.disc_peak_T_K,
                           2500.0f,
                           20000.0f,
                           "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        tip("Temperature at the hottest disc annulus. 4500 K ≈ orange-yellow "
            "(default; Interstellar look). 6000 K ≈ sun-like white. 18000 K "
            "≈ blue-white welder's arc.");
        ImGui::SliderFloat("Band turbulence", &scene.disc_turbulence, 0.0f, 1.0f);
        tip("Procedural spiraling-band amplitude. Real accretion shows "
            "filamentary structure that winds with Keplerian shear; this "
            "knob is the visible amount of that texture.");
        ImGui::Checkbox("Doppler beaming", &scene.disc_doppler_on);
        tip("Approaching side blue-shifted + brightened, receding side "
            "red-shifted + dimmed. Physically correct; Interstellar's VFX "
            "explicitly turned this off for narrative reasons.");
        ImGui::Checkbox("Gravitational redshift", &scene.disc_redshift_on);
        tip("Light climbing out of the BH's potential well loses energy. "
            "Inner-disc emission gets reddened; turning this off makes "
            "the inner ring look hotter than it really is.");
        ImGui::EndDisabled();
    }

    // --- Camera -----------------------------------------------------------
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat(
            "Distance (M)", &cam.distance_M, 4.0f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderAngle("Azimuth", &cam.azimuth_rad, -180.0f, 180.0f);
        ImGui::SliderAngle("Elevation", &cam.elevation_rad, -85.0f, 85.0f);
        tip("Vertical angle off the equatorial plane. Near 0° gives the "
            "edge-on Interstellar look; ±85° puts the camera on the polar "
            "axis (face-on disc).");
        ImGui::SliderFloat("FOV (deg)", &cam.fov_y_deg, 20.0f, 120.0f);
    }

    // --- Cinematics (advanced; collapsed by default) ----------------------
    if (ImGui::CollapsingHeader("Cinematics")) {
        ImGui::SliderFloat("Exposure", &scene.exposure, 0.1f, 4.0f);
        tip("Linear multiplier on HDR before tone mapping. Higher = "
            "brighter; the ACES tone map will compress the highlights.");
        ImGui::SliderFloat("Bloom threshold", &scene.bloom_threshold, 0.1f, 5.0f);
        tip("Luminance gate above which a pixel feeds the bloom-glow "
            "extract. Lower → more glow; too low blows out everything.");
        ImGui::SliderFloat("Bloom strength", &scene.bloom_strength, 0.0f, 1.5f);
        const char* ss_items[] = {"1x1 (fastest)", "2x2", "4x4 (cleanest)"};
        int ss_idx = scene.render_supersample == 4 ? 2 : (scene.render_supersample == 2 ? 1 : 0);
        if (ImGui::Combo("Supersample", &ss_idx, ss_items, 3)) {
            scene.render_supersample = ss_idx == 2 ? 4 : (ss_idx == 1 ? 2 : 1);
        }
        tip("Per-pixel sub-sample grid. 4×4 is the cleanest export setting "
            "but costs 16× the per-frame work — only flip it on for "
            "screenshots, not during slider drags.");
    }

    ImGui::Spacing();
    ImGui::Separator();

    // --- Actions ----------------------------------------------------------
    if (ImGui::Button("Screenshot")) {
        screenshot_requested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset to defaults")) {
        scene = Scene{};
        cam = OrbitalCamera{};
        preset_idx = 0;
        quality_idx = 1;
    }
    if (!last_screenshot_path.empty()) {
        ImGui::TextDisabled("→ %s", last_screenshot_path.c_str());
    }

    ImGui::End();
}

#if !defined(__APPLE__)

// Vulkan-path ImGui descriptor pool. Sized generously since ImGui allocates
// one descriptor set per font atlas + one per user-uploaded texture. 1000
// of each type is a standard recipe from the dear-imgui sample.
VkDescriptorPool create_imgui_descriptor_pool(VkDevice device) {
    const VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000},
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 1000 * (sizeof(sizes) / sizeof(sizes[0]));
    ci.poolSizeCount = (uint32_t)(sizeof(sizes) / sizeof(sizes[0]));
    ci.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device, &ci, nullptr, &pool);
    return pool;
}

#endif  // !__APPLE__

}  // namespace

// ===========================================================================
// run() — windowed main loop
// ===========================================================================

int run(const AppShellConfig& cfg) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        std::fprintf(stderr, "[singularity] SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

#if defined(__APPLE__)
    // --------------------------------------------------------------- Metal
    SDL_Window* window =
        SDL_CreateWindow("Singularity",
                         cfg.width,
                         cfg.height,
                         SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) {
        std::fprintf(stderr, "[singularity] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_MetalView mv = SDL_Metal_CreateView(window);
    if (!mv) {
        std::fprintf(stderr, "[singularity] SDL_Metal_CreateView failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    CAMetalLayer* layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(mv);

    auto backend = singularity::metal::create_metal_backend();
    auto* metal_be = static_cast<singularity::metal::MetalBackend*>(backend.get());
    WindowHandle wh{};
    wh.native_window = static_cast<void*>(window);
    wh.native_view = (__bridge void*)layer;
    RenderConfig rc{};
    rc.width = (std::uint32_t)cfg.width;
    rc.height = (std::uint32_t)cfg.height;
    rc.vsync_enabled = cfg.vsync;
    if (!backend->initialize(wh, rc)) {
        std::fprintf(stderr, "[singularity] backend initialize failed\n");
        SDL_Metal_DestroyView(mv);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForMetal(window);
    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_be->metal_device_handle();
    ImGui_ImplMetal_Init(dev);

#else
    // --------------------------------------------------------------- Vulkan
    SDL_Window* window =
        SDL_CreateWindow("Singularity",
                         cfg.width,
                         cfg.height,
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) {
        std::fprintf(stderr, "[singularity] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    auto backend = singularity::vulkan::create_vulkan_backend();
    auto* vk_be = static_cast<singularity::vulkan::VulkanBackend*>(backend.get());
    WindowHandle wh{};
    wh.native_window = static_cast<void*>(window);
    // Lambda captures `window` so the backend can ask SDL for a surface
    // after it builds the instance. Keeps SDL out of the backend target.
    wh.vulkan_create_surface = [window](void* instance_raw) -> void* {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(window, (VkInstance)instance_raw, nullptr, &surface)) {
            std::fprintf(stderr, "[singularity] SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
            return nullptr;
        }
        return (void*)surface;
    };
    RenderConfig rc{};
    rc.width = (std::uint32_t)cfg.width;
    rc.height = (std::uint32_t)cfg.height;
    rc.vsync_enabled = cfg.vsync;
    if (!backend->initialize(wh, rc)) {
        std::fprintf(stderr, "[singularity] backend initialize failed\n");
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL3_InitForVulkan(window);

    VkDevice device = (VkDevice)vk_be->vk_device_handle();
    VkDescriptorPool imgui_pool = create_imgui_descriptor_pool(device);

    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = (VkInstance)vk_be->vk_instance_handle();
    ii.PhysicalDevice = (VkPhysicalDevice)vk_be->vk_physical_device_handle();
    ii.Device = device;
    ii.QueueFamily = vk_be->vk_graphics_queue_family();
    ii.Queue = (VkQueue)vk_be->vk_graphics_queue_handle();
    ii.DescriptorPool = imgui_pool;
    ii.RenderPass = (VkRenderPass)vk_be->vk_render_pass_handle();
    ii.Subpass = 0;
    ii.MinImageCount = 2;
    ii.ImageCount = std::max<std::uint32_t>(2, vk_be->vk_swapchain_image_count());
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&ii);
#endif

    Scene scene{};
    OrbitalCamera cam{};
    // Restore previous session's settings, then let CLI flags override.
    // Unknown keys and missing files are silently ignored.
    const auto settings_file = settings_path();
    load_settings(settings_file, scene, cam);
    if (cfg.start_kerr) {
        scene.metric = Scene::MetricType::Kerr;
        scene.spin_a_over_M = cfg.start_spin_a_over_M;
    }
    CameraState camera_state{};

    std::uint64_t last_tick = SDL_GetPerformanceCounter();
    float smoothed_fps = 60.0f;

    // Screenshot state: the ImGui Screenshot button flips `requested` true;
    // the main loop captures a frame after render_frame() and stores the
    // resulting file path for display in the panel on the next tick.
    bool screenshot_requested = false;
    std::string last_screenshot_path;

    // Debounced settings save: on the frame where IsAnyItemActive transitions
    // true -> false (i.e. the user just released a slider / closed a combo),
    // flip `settings_dirty`. Main loop consumes it and writes settings.conf.
    // Effectively edge-triggered — no writes mid-drag, one write per settled
    // adjustment.
    bool settings_dirty = false;
    bool prev_item_active = false;

    // Register overlay callback — runs inside the backend's blit render pass,
    // composed over the tone-mapped black hole.
#if defined(__APPLE__)
    metal_be->set_overlay([&](void* rpd_handle, void* cmd_handle, void* enc_handle) {
        MTLRenderPassDescriptor* rpd = (__bridge MTLRenderPassDescriptor*)rpd_handle;
        id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_handle;
        id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)enc_handle;

        ImGui_ImplMetal_NewFrame(rpd);
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        imgui_settings_panel(scene,
                             cam,
                             smoothed_fps,
                             metal_be->last_gpu_seconds(),
                             screenshot_requested,
                             last_screenshot_path);
        {
            const bool cur_active = ImGui::IsAnyItemActive();
            if (prev_item_active && !cur_active) {
                settings_dirty = true;
            }
            prev_item_active = cur_active;
        }
        ImGui::Render();
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, enc);
    });
#else
    vk_be->set_overlay([&](void* /*fb*/, void* cmd_handle, void* /*rp*/) {
        VkCommandBuffer cmd = (VkCommandBuffer)cmd_handle;

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        imgui_settings_panel(scene,
                             cam,
                             smoothed_fps,
                             vk_be->last_gpu_seconds(),
                             screenshot_requested,
                             last_screenshot_path);
        {
            const bool cur_active = ImGui::IsAnyItemActive();
            if (prev_item_active && !cur_active) {
                settings_dirty = true;
            }
            prev_item_active = cur_active;
        }
        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    });
#endif

    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL3_ProcessEvent(&ev);
            switch (ev.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
                int w, h;
                SDL_GetWindowSizeInPixels(window, &w, &h);
                backend->resize((std::uint32_t)w, (std::uint32_t)h);
                break;
            }
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (!io.WantCaptureMouse && ev.button.button == SDL_BUTTON_LEFT) {
                    cam.dragging = true;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (ev.button.button == SDL_BUTTON_LEFT)
                    cam.dragging = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (cam.dragging && !io.WantCaptureMouse) {
                    apply_drag(cam, ev.motion.xrel, ev.motion.yrel);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (!io.WantCaptureMouse)
                    apply_zoom(cam, ev.wheel.y);
                break;
            case SDL_EVENT_KEY_DOWN:
                if (ev.key.key == SDLK_ESCAPE)
                    running = false;
                break;
            default:
                break;
            }
        }

        cam.write_state(camera_state);

#if defined(__APPLE__)
        // Keep the Metal layer's drawable size in sync with the window's
        // pixel size — HiDPI displays change this dynamically.
        {
            CAMetalLayer* imlayer = (__bridge CAMetalLayer*)metal_be->metal_layer_handle();
            int px_w = 0, px_h = 0;
            SDL_GetWindowSizeInPixels(window, &px_w, &px_h);
            imlayer.drawableSize = CGSizeMake(px_w, px_h);
        }
#endif

        backend->render_frame(scene, camera_state);

        // Pick up the screenshot request deferred from the ImGui panel. We
        // capture *after* render_frame so we get the image the user is
        // actually looking at, and the wait+blit cost is paid as a one-frame
        // hitch rather than stalling every frame. When the panel's live
        // supersample is below 4×, do one extra render at 4× per-pixel
        // samples so screenshots are always high-quality regardless of what
        // the user has the slider on for interactive tuning.
        if (screenshot_requested) {
            screenshot_requested = false;
            const std::uint32_t saved_ss = scene.render_supersample;
            if (saved_ss < 4) {
                scene.render_supersample = 4;
                backend->render_frame(scene, camera_state);
            }
            ImageData img = backend->capture_frame();
            scene.render_supersample = saved_ss;
            auto path = save_screenshot(img);
            last_screenshot_path = path.empty() ? std::string("save failed") : path.string();
        }

        // Debounced settings save — consumed at the tail of the frame so
        // the write lands after the render_frame/capture work has already
        // happened and doesn't compete for a drawable.
        if (settings_dirty) {
            settings_dirty = false;
            save_settings(settings_file, scene, cam);
        }

        const std::uint64_t now = SDL_GetPerformanceCounter();
        const double dt = double(now - last_tick) / double(SDL_GetPerformanceFrequency());
        last_tick = now;
        if (dt > 0.0) {
            const float inst = float(1.0 / dt);
            smoothed_fps = smoothed_fps * 0.9f + inst * 0.1f;
        }
    }

    // Persist the latest tunings before tearing down — best-effort, we
    // don't block shutdown if the write fails.
    save_settings(settings_file, scene, cam);

#if defined(__APPLE__)
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    backend->shutdown();
    SDL_Metal_DestroyView(mv);
    SDL_DestroyWindow(window);
#else
    backend->shutdown();  // must happen first — ImGui-Vulkan shutdown calls
                          // vkQueueWaitIdle internally, which is safe, but we
                          // also need the backend's device to still exist
                          // while ImGui destroys its buffers. backend owns
                          // the device; its destructor runs last.
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);
    SDL_DestroyWindow(window);
#endif
    SDL_Quit();
    return 0;
}

// ===========================================================================
// run_capture() — headless single-frame capture
// ===========================================================================

int run_capture(const AppShellConfig& cfg) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "[singularity] SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

#if defined(__APPLE__)
    SDL_Window* window = SDL_CreateWindow(
        "Singularity (capture)", cfg.width, cfg.height, SDL_WINDOW_METAL | SDL_WINDOW_HIDDEN);
    if (!window) {
        std::fprintf(stderr, "[singularity] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    SDL_MetalView mv = SDL_Metal_CreateView(window);
    CAMetalLayer* layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(mv);

    auto backend = singularity::metal::create_metal_backend();
    WindowHandle wh{};
    wh.native_window = static_cast<void*>(window);
    wh.native_view = (__bridge void*)layer;
#else
    SDL_Window* window = SDL_CreateWindow(
        "Singularity (capture)", cfg.width, cfg.height, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    if (!window) {
        std::fprintf(stderr, "[singularity] SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    auto backend = singularity::vulkan::create_vulkan_backend();
    WindowHandle wh{};
    wh.native_window = static_cast<void*>(window);
    wh.vulkan_create_surface = [window](void* instance_raw) -> void* {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(window, (VkInstance)instance_raw, nullptr, &surface)) {
            return nullptr;
        }
        return (void*)surface;
    };
#endif

    RenderConfig rc{};
    rc.width = (std::uint32_t)cfg.width;
    rc.height = (std::uint32_t)cfg.height;
    rc.vsync_enabled = false;
    if (!backend->initialize(wh, rc)) {
        std::fprintf(stderr, "[singularity] backend initialize failed\n");
#if defined(__APPLE__)
        SDL_Metal_DestroyView(mv);
#endif
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    Scene scene{};
    if (cfg.capture_spin_a_over_M > 0.0f || cfg.start_kerr) {
        scene.metric = Scene::MetricType::Kerr;
        scene.spin_a_over_M = cfg.capture_spin_a_over_M;
    }
    scene.render_supersample = (std::uint32_t)cfg.capture_supersample;

    OrbitalCamera cam{};
    cam.distance_M = cfg.capture_distance_M;
    cam.elevation_rad = cfg.capture_elevation_rad;
    CameraState camera_state{};
    cam.write_state(camera_state);

    backend->render_frame(scene, camera_state);
    ImageData img = backend->capture_frame();

    int rc_write = stbi_write_png(cfg.capture_path.c_str(),
                                  (int)img.width,
                                  (int)img.height,
                                  4,
                                  img.pixels_rgba.data(),
                                  (int)img.width * 4);

    backend->shutdown();
#if defined(__APPLE__)
    SDL_Metal_DestroyView(mv);
#endif
    SDL_DestroyWindow(window);
    SDL_Quit();

    if (!rc_write) {
        std::fprintf(
            stderr, "[singularity] stbi_write_png failed for %s\n", cfg.capture_path.c_str());
        return 1;
    }
    std::printf("wrote %s (%ux%u, ss=%d, metric=%s, spin=%g, "
                "dist=%.2fM, elev=%.2f rad)\n",
                cfg.capture_path.c_str(),
                img.width,
                img.height,
                cfg.capture_supersample,
                scene.metric == Scene::MetricType::Kerr ? "kerr" : "schw",
                scene.spin_a_over_M,
                cfg.capture_distance_M,
                cfg.capture_elevation_rad);
    return 0;
}

}  // namespace singularity::app
