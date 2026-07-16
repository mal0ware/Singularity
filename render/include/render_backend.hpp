// render/include/singularity/render_backend.hpp
//
// The abstract RenderBackend interface. Every GPU API implementation (Metal,
// Vulkan, CUDA offline) lives behind this one header. Kept deliberately small
// — over-abstracting is the failure mode. See docs/ARCHITECTURE.md §3 for
// rationale on what this interface deliberately omits.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace singularity {

// Platform-native window + surface handles passed into `initialize()`.
// The backend interprets these per its API:
//   - Metal:  native_window = NSWindow*,  native_view = CAMetalLayer*
//   - Vulkan: native_window = SDL_Window* (app-side), vulkan_create_surface
//             invoked by the backend after it builds its VkInstance so the
//             backend never links SDL itself. Cast the arg to VkInstance and
//             the return to VkSurfaceKHR.
//   - CUDA:   all fields ignored (offline renderer has no window)
struct WindowHandle {
    void* native_window{nullptr};
    void* native_view{nullptr};
    std::function<void*(void* /*VkInstance*/)> vulkan_create_surface{};
};

struct RenderConfig {
    std::uint32_t width{1280};
    std::uint32_t height{720};
    bool vsync_enabled{true};
};

// Scene state — small enough (<100 bytes) to upload as a uniform every frame.
// Kept trivially-copyable so the same struct can be memcpy'd into a GPU buffer.
//
// Defaults are tuned for the **Interstellar shape**: Kerr a/M=0.6 (Kip
// Thorne's Gargantua), disc inner edge at ISCO for that spin, peak disc
// temperature 4500 K (warm orange-yellow rather than the hot blue-white
// you'd get at 18000 K). Doppler + redshift are *on* — that's the
// physically-honest choice and produces a deliberately asymmetric
// disc, which the realtime backends preserve.
struct Scene {
    enum class MetricType : std::uint32_t {
        Schwarzschild = 0,
        Kerr = 1,
    };

    MetricType metric{MetricType::Kerr};
    float mass_solar{1.0f};
    float spin_a_over_M{0.6f};  // 0.6 — Kip Thorne's Gargantua spin.
    float disc_inner_M{3.83f};  // ISCO at a/M=0.6, prograde (analytic).
    float disc_outer_M{25.0f};  // Tighter than the 30M default for visual punch.
    bool disc_doppler_on{true};
    bool disc_redshift_on{true};
    bool disc_texture_on{true};
    bool show_overlay{false};

    // Rendering knobs — not part of the physics, but wired into the
    // backend's uniform so the ImGui panel can tune them live.
    std::uint32_t render_supersample{1};  // 1/2/4 — N×N subpixel grid
    float exposure{1.8f};                 // linear multiplier before tone-map
    float bloom_threshold{0.8f};          // luminance gate for bloom extract
    float bloom_strength{0.55f};          // how strongly to mix bloom into final
    float disc_turbulence{0.5f};          // spiraling-band amplitude [0,1]
    float disc_peak_T_K{4500.0f};         // peak disc temperature — warm orange-yellow.

    // Quality / responsiveness override. 0 means "use the backend default
    // (kDefaultMaxSteps / kDefaultHStep)". Setting these from a UI quality
    // preset (Draft / Default / Quality) lets the ImGui panel trade
    // visible noise for live framerate without re-writing pack_uniforms.
    std::uint32_t max_steps{0};
    float h_step{0.0f};

    // Radius-adaptive integrator step (SING_FLAG_ADAPTIVE_STEP; see
    // shared_shader/geodesic_math.h adaptive_h). Default off so the desktop
    // golden images stay byte-stable; the web demo enables it.
    bool adaptive_step_on{false};
};

struct CameraState {
    float position[3]{0.0f, 0.0f, 30.0f};
    float basis[9]{
        // 3x3 orientation, row-major
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    float fov_y_radians{1.047f};  // ~60°
};

struct ImageData {
    std::vector<std::uint8_t> pixels_rgba;
    std::uint32_t width{0};
    std::uint32_t height{0};
};

class RenderBackend {
public:
    virtual ~RenderBackend() = default;

    RenderBackend(const RenderBackend&) = delete;
    RenderBackend& operator=(const RenderBackend&) = delete;
    RenderBackend(RenderBackend&&) = delete;
    RenderBackend& operator=(RenderBackend&&) = delete;

    // Lifecycle.
    virtual bool initialize(WindowHandle window, RenderConfig config) = 0;
    virtual void shutdown() = 0;
    virtual void resize(std::uint32_t width, std::uint32_t height) = 0;

    // Per-frame.
    virtual void render_frame(const Scene& scene, const CameraState& camera) = 0;

    // For verification harness + screenshot export.
    virtual ImageData capture_frame() = 0;

    // Human-readable identifier: "Metal", "Vulkan", "CUDA".
    virtual const char* name() const = 0;

protected:
    RenderBackend() = default;
};

// Factory — picks the compiled-in backend appropriate for the current platform.
// If multiple backends are compiled in, prefers the platform-native one:
//   - APPLE  -> Metal
//   - WIN32  -> Vulkan
// Falls back to any backend available if the preferred one is absent.
std::unique_ptr<RenderBackend> create_default_backend();

}  // namespace singularity
