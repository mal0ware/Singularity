// render/webgpu/webgpu_backend.hpp
//
// Pure-C++ façade over the WebGPU RenderBackend. Mirrors the structure
// of render/vulkan/vulkan_backend.hpp — pimpl'd so the rest of the
// codebase doesn't pull in Emscripten's <webgpu/webgpu.h> (a transitive
// ~20 kLOC of C types and macros) just to link against us.
//
// The backend is Emscripten-only; on a native build (Metal / Vulkan)
// the unique pointer factory returns an empty stub so the desktop
// binary has no dependency on WebGPU.

#pragma once

#include <functional>
#include <memory>

#include "render_backend.hpp"

namespace singularity::webgpu {

// The WebGPU analogue of OverlayCallback. Handed opaque handles the
// caller casts to the real types:
//   fb_handle          -> WGPUTextureView  (swapchain image view)
//   cmd_encoder_handle -> WGPUCommandEncoder
//   render_pass_handle -> WGPURenderPassEncoder   (already begun by backend)
//
// Kept symmetric with the Metal/Vulkan overlay contract so the app shell
// can swap the three with a single #ifdef. ImGui's WebGPU backend is
// driven off these three handles.
using OverlayCallback =
    std::function<void(void* fb_handle, void* cmd_encoder_handle, void* render_pass_handle)>;

class WebGPUBackend final : public RenderBackend {
public:
    WebGPUBackend();
    ~WebGPUBackend() override;

    WebGPUBackend(const WebGPUBackend&) = delete;
    WebGPUBackend& operator=(const WebGPUBackend&) = delete;

    bool initialize(WindowHandle window, RenderConfig config) override;
    void shutdown() override;
    void resize(std::uint32_t width, std::uint32_t height) override;
    void render_frame(const Scene& scene, const CameraState& camera) override;
    ImageData capture_frame() override;
    const char* name() const override { return "WebGPU"; }

    void set_overlay(OverlayCallback cb);

    // Opaque handles for ImGui-WebGPU bring-up. All return WGPUFoo as void*;
    // the app casts back. These mirror vk_instance_handle / vk_device_handle
    // / vk_graphics_queue_handle from the Vulkan backend.
    void* wgpu_device_handle() const;           // WGPUDevice
    void* wgpu_queue_handle() const;            // WGPUQueue
    void* wgpu_surface_handle() const;          // WGPUSurface  (canvas-bound)
    std::uint32_t wgpu_surface_format() const;  // WGPUTextureFormat enum value

    // GPU-measured time for the most recent completed frame (seconds).
    // WebGPU timestamp queries are optional; returns 0 when the device
    // didn't advertise the timestamp-query feature.
    double last_gpu_seconds() const;

    // Forward-declared out here so the .cpp can reference
    // WebGPUBackend::Impl in the signatures of its file-static helpers
    // without tripping access control. The definition still lives in
    // the .cpp, so the type remains opaque to callers.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<RenderBackend> create_webgpu_backend();

}  // namespace singularity::webgpu
