// render/vulkan/vulkan_backend.hpp
//
// Pure-C++ façade over the Vulkan RenderBackend. Mirrors
// render/metal/metal_backend.hpp in structure — pimpl'd so the rest
// of the codebase doesn't pull in Vulkan-Hpp (~30 kLOC of templates)
// just to link against us.

#pragma once

#include <functional>
#include <memory>

#include "render_backend.hpp"

namespace singularity::vulkan {

// Mirror of singularity::metal::OverlayCallback. Handed opaque Vulkan
// handles the caller casts to the real types:
//   fb_handle          -> VkFramebuffer (or VkImageView for dynamic rendering)
//   cmd_buffer_handle  -> VkCommandBuffer
//   render_pass_handle -> VkRenderPass (for ImGui init context)
using OverlayCallback =
    std::function<void(void* fb_handle, void* cmd_buffer_handle, void* render_pass_handle)>;

class VulkanBackend final : public RenderBackend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;

    bool initialize(WindowHandle window, RenderConfig config) override;
    void shutdown() override;
    void resize(std::uint32_t width, std::uint32_t height) override;
    void render_frame(const Scene& scene, const CameraState& camera) override;
    ImageData capture_frame() override;
    const char* name() const override { return "Vulkan"; }

    void set_overlay(OverlayCallback cb);

    // Opaque handles to Vulkan objects, for ImGui-Vulkan bring-up in the
    // app shell. All return VkFoo as `void*`; the app casts back.
    void* vk_instance_handle() const;         // VkInstance
    void* vk_physical_device_handle() const;  // VkPhysicalDevice
    void* vk_device_handle() const;           // VkDevice
    void* vk_graphics_queue_handle() const;   // VkQueue
    void* vk_render_pass_handle() const;      // VkRenderPass (blit pass)
    std::uint32_t vk_graphics_queue_family() const;
    std::uint32_t vk_swapchain_image_count() const;

    // GPU-measured time for the most recent completed frame (seconds).
    double last_gpu_seconds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Private initialize() splits — only meaningful when the Vulkan SDK
    // is available at build time; otherwise the class operates as a stub
    // and these are never called. Declared unconditionally because the
    // SINGULARITY_VULKAN_SDK_AVAILABLE macro is a PRIVATE define on the
    // singularity_render_vulkan target, where the .cpp lives — it's also
    // consistently set in every TU that includes this header.
    void build_swapchain_and_views();
    void build_descriptor_layouts();
    void build_render_pass();
    void build_pipelines();
    void build_sampler();
    void create_hdr_and_bloom();
    void create_framebuffers();
    void create_descriptor_pool();
    void create_per_frame_resources();
    void update_descriptor_sets();
};

std::unique_ptr<RenderBackend> create_vulkan_backend();

}  // namespace singularity::vulkan
