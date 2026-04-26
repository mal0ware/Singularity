// render/cuda/cuda_backend.hpp
//
// CUDA implementation of `RenderBackend` — offline only. The interactive
// desktop app never instantiates this; the live-app factory routes to
// Metal on Apple and Vulkan on Windows. `singularity_cuda_cli` links
// against this target directly so the live-app graph stays narrow.
//
// See docs/ARCHITECTURE.md §7 and docs/TODO.md Phase 8.

#ifndef SINGULARITY_RENDER_CUDA_CUDA_BACKEND_HPP
#define SINGULARITY_RENDER_CUDA_CUDA_BACKEND_HPP

#include "render_backend.hpp"

namespace singularity {

class CudaBackend : public RenderBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    // Lifecycle. `WindowHandle` is ignored — CUDA has no window. `initialize`
    // allocates the device-side RGBA8 buffer at `config.width x config.height`.
    bool initialize(WindowHandle window, RenderConfig config) override;
    void shutdown() override;
    void resize(std::uint32_t width, std::uint32_t height) override;

    void render_frame(const Scene& scene, const CameraState& camera) override;
    ImageData capture_frame() override;

    const char* name() const override { return "CUDA"; }

private:
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    // Device-side RGBA8 image. Opaque pointer so this header stays free of
    // <cuda_runtime.h> — callers don't need the CUDA headers on their
    // include path.
    void* d_image_{nullptr};
    // Disc-animation phase. Held at 0 for the first `render_frame` so
    // single-shot offline captures stay deterministic against the
    // committed goldens; advanced by `1/30 s` after each `render_frame`
    // returns, so multi-frame `cuda_cli --frames N` mode produces a
    // smoothly-animating disc when its PNG sequence is encoded at 30 fps.
    float frame_time_sec_{0.0f};
};

}  // namespace singularity

#endif  // SINGULARITY_RENDER_CUDA_CUDA_BACKEND_HPP
