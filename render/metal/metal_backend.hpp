// render/metal/metal_backend.hpp
//
// Pure-C++ façade over the Objective-C++ Metal backend. The .mm file holds
// all the MTL* / CAMetalLayer / ImGui-Metal bookkeeping behind an opaque
// pimpl so the rest of the codebase — including the cross-platform app
// shell and the factory in `create_default_backend.cpp` — never has to be
// compiled as Objective-C++.

#pragma once

#include <functional>
#include <memory>

#include "render_backend.hpp"

namespace singularity::metal {

// Overlay callback — invoked inside the backend's blit render pass with
// enough handles to drive ImGui-Metal (and whatever else). All three
// pointers are opaque Obj-C handles that the caller __bridge-casts back
// to their real types:
//   rpd_handle         -> MTLRenderPassDescriptor*
//   cmd_buffer_handle  -> id<MTLCommandBuffer>
//   encoder_handle     -> id<MTLRenderCommandEncoder>
using OverlayCallback =
    std::function<void(void* rpd_handle, void* cmd_buffer_handle, void* encoder_handle)>;

// Concrete RenderBackend for Apple platforms. Constructed via
// `create_metal_backend()`; it's exposed separately from `create_default_backend`
// so tests and tools that unconditionally want the Metal path can ask for it.
class MetalBackend final : public RenderBackend {
public:
    MetalBackend();
    ~MetalBackend() override;

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    bool initialize(WindowHandle window, RenderConfig config) override;
    void shutdown() override;
    void resize(std::uint32_t width, std::uint32_t height) override;
    void render_frame(const Scene& scene, const CameraState& camera) override;
    ImageData capture_frame() override;
    const char* name() const override { return "Metal"; }

    // Register an overlay to be rendered inside the same render pass as the
    // tone-mapped blit. Called once at setup; invoked every frame by
    // render_frame. Pass an empty function to clear.
    void set_overlay(OverlayCallback cb);

    // Forward-declared; set after initialize() so the app shell's ImGui code
    // can look up the device/layer without reaching into the pimpl.
    void* metal_device_handle() const;  // MTLDevice*
    void* metal_layer_handle() const;   // CAMetalLayer*

    // GPU-measured time for the most recent completed command buffer
    // (seconds). Populated from MTLCommandBuffer.GPUEndTime − GPUStartTime.
    // Safe to read from any thread.
    double last_gpu_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<RenderBackend> create_metal_backend();

}  // namespace singularity::metal
