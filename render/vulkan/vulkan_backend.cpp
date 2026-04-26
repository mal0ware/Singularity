// render/vulkan/vulkan_backend.cpp
//
// Vulkan RenderBackend — mirror of render/metal/metal_backend.mm. Owns the
// VkInstance, VkDevice, swapchain, the geodesic compute kernel + bloom ping-
// pong + fullscreen blit render pass, and a timestamp query pool for GPU-
// measured frame timing. Uses Vulkan-Hpp RAII handles throughout so shutdown
// is the natural destructor sequence — no explicit vkDestroy* calls.
//
// Platform integration: the backend never links SDL. Instead
// WindowHandle::vulkan_create_surface is a callback the app shell sets to
// `SDL_Vulkan_CreateSurface(window, inst, NULL, &surface)`. That keeps the
// renderer window-system-agnostic.

#include "vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
// NOMINMAX suppresses the min/max macros windows.h otherwise defines, which
// collide with std::numeric_limits<T>::max() elsewhere in this TU.
// WIN32_LEAN_AND_MEAN cuts the unnecessary rpc / crypto / etc. subsystems.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

#if defined(SINGULARITY_VULKAN_SDK_AVAILABLE) && SINGULARITY_VULKAN_SDK_AVAILABLE
#define SINGULARITY_VULKAN_HPP_AVAILABLE 1
#include <vulkan/vulkan.hpp>

#include "uniforms.h"
#else
#define SINGULARITY_VULKAN_HPP_AVAILABLE 0
#endif

namespace singularity::vulkan {

namespace {

[[maybe_unused]] void warn_once(const char* msg) {
    static std::atomic<bool> emitted{false};
    bool expected = false;
    if (emitted.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr, "[singularity/vulkan] %s\n", msg);
    }
}

#if SINGULARITY_VULKAN_HPP_AVAILABLE

constexpr std::uint32_t kDefaultMaxSteps = 2500;
constexpr float kDefaultHStep = 0.4f;
constexpr std::size_t kMaxFramesInFlight = 3;

// Two timestamps per frame — begin and end of the recorded cmd buffer.
constexpr std::uint32_t kTimestampsPerFrame = 2;

// Tone-map/exposure defaults used for the --capture headless path, to match
// the Metal backend's choices (MetalBackend::capture_frame).
constexpr float kCaptureExposure = 1.0f;
constexpr float kCaptureBloomStrength = 0.35f;

// Mirrors the ExtractParams/BlitParams structs in bloom.hlsl/blit.hlsl.
// Kept 16-byte so the UBO binding matches the std140 expectation.
struct alignas(16) BloomParamsCpu {
    float threshold;
    float pad0;
    float pad1;
    float pad2;
};
struct alignas(16) BlitParamsCpu {
    float exposure;
    float bloom_strength;
    float pad0;
    float pad1;
};

void pack_uniforms(Uniforms& out,
                   const Scene& scene,
                   const CameraState& cam,
                   std::uint32_t w,
                   std::uint32_t h,
                   std::uint32_t frame_index,
                   float time_sec) {
    out.cam_pos = {cam.position[0], cam.position[1], cam.position[2], 0.0f};
    out.cam_right = {cam.basis[0], cam.basis[1], cam.basis[2], 0.0f};
    out.cam_up = {cam.basis[3], cam.basis[4], cam.basis[5], 0.0f};
    out.cam_fwd = {-cam.basis[6], -cam.basis[7], -cam.basis[8], 0.0f};

    out.tan_half_fov = std::tan(0.5f * cam.fov_y_radians);
    out.aspect = float(w) / float(h);

    out.mass_M = scene.mass_solar;
    out.spin_a = scene.spin_a_over_M * scene.mass_solar;
    out.rs = 2.0f * scene.mass_solar;

    out.disc_r_inner = scene.disc_inner_M * scene.mass_solar;
    out.disc_r_outer = scene.disc_outer_M * scene.mass_solar;
    out.disc_peak_T = scene.disc_peak_T_K;

    out.h_step = scene.h_step > 0.0f ? scene.h_step : kDefaultHStep;
    out.escape_r = 200.0f * scene.mass_solar;
    out.horizon_cut = 1.02f * out.rs;
    out.time_sec = time_sec;

    out.width = w;
    out.height = h;
    out.metric_type =
        (scene.metric == Scene::MetricType::Kerr) ? SING_METRIC_KERR : SING_METRIC_SCHWARZSCHILD;

    std::uint32_t flags = SING_FLAG_STARFIELD_ON;
    if (scene.disc_texture_on)
        flags |= SING_FLAG_DISC_ON;
    if (scene.disc_doppler_on)
        flags |= SING_FLAG_DOPPLER_ON;
    if (scene.disc_redshift_on)
        flags |= SING_FLAG_REDSHIFT_ON;
    out.flags = flags;

    out.max_steps = scene.max_steps != 0 ? scene.max_steps : kDefaultMaxSteps;
    out.frame_index = frame_index;
    out.supersample = scene.render_supersample;
    out.pad_b = 0;

    out.exposure = scene.exposure;
    out.bloom_threshold = scene.bloom_threshold;
    out.bloom_strength = scene.bloom_strength;
    out.disc_turbulence = scene.disc_turbulence;
}

std::vector<char> read_file_binary(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f)
        return {};
    const std::streamsize n = f.tellg();
    if (n <= 0)
        return {};
    std::vector<char> buf(static_cast<std::size_t>(n));
    f.seekg(0);
    f.read(buf.data(), n);
    if (!f)
        return {};
    return buf;
}

// Best-effort: the directory containing the current executable. Returns an
// empty path on failure — callers must handle that. Used to locate bundled
// SPIR-V next to the exe for installed builds.
std::filesystem::path get_executable_dir() {
#if defined(_WIN32)
    wchar_t buf[4096];
    const DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
    if (n == 0 || n >= 4096)
        return {};
    return std::filesystem::path(buf).parent_path();
#elif defined(__APPLE__)
    char buf[4096];
    std::uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) != 0)
        return {};
    std::error_code ec;
    auto resolved = std::filesystem::canonical(buf, ec);
    return ec ? std::filesystem::path(buf).parent_path() : resolved.parent_path();
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0)
        return {};
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path();
#endif
}

// Resolve the directory holding the SPIR-V artefacts. Precedence:
//   1. SINGULARITY_SPV_DIR env variable (explicit override)
//   2. <exe_dir>/shaders                  (packaged layout — the .msi lays
//                                          the shaders out exactly like this)
//   3. SINGULARITY_SPV_BUILD_DIR          (dev in-tree builds, set by CMake)
//   4. "shaders" relative to CWD          (last-resort for ad-hoc copies)
std::filesystem::path resolve_spv_dir() {
    if (const char* env = std::getenv("SINGULARITY_SPV_DIR")) {
        if (*env)
            return std::filesystem::path(env);
    }
    const auto exe_dir = get_executable_dir();
    if (!exe_dir.empty()) {
        const auto next_to_exe = exe_dir / "shaders";
        std::error_code ec;
        if (std::filesystem::exists(next_to_exe, ec) && !ec) {
            return next_to_exe;
        }
    }
#ifdef SINGULARITY_SPV_BUILD_DIR
    return std::filesystem::path(SINGULARITY_SPV_BUILD_DIR);
#else
    return std::filesystem::path("shaders");
#endif
}

vk::UniqueShaderModule load_shader_module(vk::Device device,
                                          const std::filesystem::path& spv_path) {
    auto bytes = read_file_binary(spv_path);
    if (bytes.empty() || bytes.size() % 4 != 0) {
        throw std::runtime_error("Vulkan: failed to load SPIR-V " + spv_path.string());
    }
    vk::ShaderModuleCreateInfo ci{};
    ci.codeSize = bytes.size();
    ci.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
    return device.createShaderModuleUnique(ci);
}

std::uint32_t
find_memory_type(vk::PhysicalDevice pd, std::uint32_t type_bits, vk::MemoryPropertyFlags props) {
    const vk::PhysicalDeviceMemoryProperties mem = pd.getMemoryProperties();
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Vulkan: no matching memory type");
}

void transition_image(vk::CommandBuffer cmd,
                      vk::Image img,
                      vk::ImageLayout old_layout,
                      vk::ImageLayout new_layout,
                      vk::PipelineStageFlags src_stage,
                      vk::PipelineStageFlags dst_stage,
                      vk::AccessFlags src_access,
                      vk::AccessFlags dst_access) {
    vk::ImageMemoryBarrier b{};
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    b.subresourceRange.baseMipLevel = 0;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount = 1;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    cmd.pipelineBarrier(src_stage, dst_stage, {}, {}, {}, b);
}

#endif  // SINGULARITY_VULKAN_HPP_AVAILABLE

}  // anonymous namespace

// ===========================================================================
// Impl
// ===========================================================================

#if SINGULARITY_VULKAN_HPP_AVAILABLE

struct VulkanBackend::Impl {
    // --- Instance / device ---------------------------------------------------
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::Queue graphics_queue;
    std::uint32_t graphics_queue_family = 0;
    float timestamp_period_ns = 1.0f;

    // --- Surface + swapchain -------------------------------------------------
    vk::UniqueSurfaceKHR surface;
    vk::UniqueSwapchainKHR swapchain;
    vk::Format swapchain_format = vk::Format::eB8G8R8A8Srgb;
    vk::Extent2D swapchain_extent;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::UniqueImageView> swapchain_views;
    std::vector<vk::UniqueFramebuffer> swapchain_framebuffers;

    // --- Render pass ---------------------------------------------------------
    vk::UniqueRenderPass blit_render_pass;

    // --- Offscreen HDR + bloom targets --------------------------------------
    struct ImageTarget {
        vk::UniqueImage image;
        vk::UniqueDeviceMemory memory;
        vk::UniqueImageView view;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    ImageTarget hdr;
    ImageTarget bloom_a;
    ImageTarget bloom_b;

    // --- Sampler -------------------------------------------------------------
    vk::UniqueSampler linear_sampler;

    // --- Descriptor-set layouts + pipeline layouts --------------------------
    vk::UniqueDescriptorSetLayout geodesic_dsl;
    vk::UniqueDescriptorSetLayout bloom_dsl;
    vk::UniqueDescriptorSetLayout blit_dsl;

    vk::UniquePipelineLayout geodesic_pl;
    vk::UniquePipelineLayout bloom_pl;
    vk::UniquePipelineLayout blit_pl;

    // --- Pipelines -----------------------------------------------------------
    vk::UniquePipeline geodesic_pipeline;
    vk::UniquePipeline bloom_extract_pipeline;
    vk::UniquePipeline bloom_blur_h_pipeline;
    vk::UniquePipeline bloom_blur_v_pipeline;
    vk::UniquePipeline blit_pipeline;

    // --- Descriptor pool -----------------------------------------------------
    vk::UniqueDescriptorPool descriptor_pool;

    // --- Per-frame resources -------------------------------------------------
    struct Buffer {
        vk::UniqueBuffer buffer;
        vk::UniqueDeviceMemory memory;
        void* mapped = nullptr;
        vk::DeviceSize size = 0;
    };
    struct FrameContext {
        vk::UniqueCommandPool pool;
        vk::UniqueCommandBuffer cmd;
        vk::UniqueSemaphore image_available;
        vk::UniqueSemaphore render_finished;
        vk::UniqueFence in_flight;
        Buffer uniforms;      // sizeof(Uniforms)
        Buffer bloom_params;  // sizeof(BloomParamsCpu)
        Buffer blit_params;   // sizeof(BlitParamsCpu)
        // Descriptor sets — owned by descriptor_pool, no explicit free.
        vk::DescriptorSet geodesic_ds;
        vk::DescriptorSet bloom_extract_ds;
        vk::DescriptorSet bloom_blur_h_ds;
        vk::DescriptorSet bloom_blur_v_ds;
        vk::DescriptorSet blit_ds;
        vk::UniqueQueryPool timestamp_pool;
        bool timestamps_submitted = false;
    };
    std::array<FrameContext, kMaxFramesInFlight> frames;

    // --- Capture staging (lazily created) -----------------------------------
    ImageTarget capture_staging;  // linear, host-visible
    vk::UniqueCommandPool capture_pool;
    vk::UniqueCommandBuffer capture_cmd;
    vk::UniqueFence capture_fence;

    // --- Misc ---------------------------------------------------------------
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t frame = 0;
    // Wall-clock zero set at backend initialize. Fed into the kernel as
    // `time_sec` so the disc animation phase is consistent across frames.
    // Offline / capture renders fire `render_frame` immediately after
    // initialize so elapsed ≈ 0 — keeps headless captures deterministic.
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool vsync = true;
    std::atomic<double> last_gpu_seconds{0.0};
    OverlayCallback overlay;
    std::filesystem::path spv_dir;

    // Previous acquired-image index, needed by present in render_frame.
    std::uint32_t image_index = 0;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    Buffer create_buffer(vk::DeviceSize size,
                         vk::BufferUsageFlags usage,
                         vk::MemoryPropertyFlags props,
                         bool map_persistent = false) {
        Buffer b;
        b.size = size;
        vk::BufferCreateInfo ci{};
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = vk::SharingMode::eExclusive;
        b.buffer = device->createBufferUnique(ci);

        const auto req = device->getBufferMemoryRequirements(*b.buffer);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(physical_device, req.memoryTypeBits, props);
        b.memory = device->allocateMemoryUnique(ai);
        device->bindBufferMemory(*b.buffer, *b.memory, 0);

        if (map_persistent) {
            b.mapped = device->mapMemory(*b.memory, 0, size);
        }
        return b;
    }

    ImageTarget create_image_target(
        std::uint32_t w,
        std::uint32_t h,
        vk::Format format,
        vk::ImageUsageFlags usage,
        vk::ImageAspectFlags aspect,
        vk::ImageTiling tiling = vk::ImageTiling::eOptimal,
        vk::MemoryPropertyFlags mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal) {
        ImageTarget t;
        vk::ImageCreateInfo ci{};
        ci.imageType = vk::ImageType::e2D;
        ci.format = format;
        ci.extent = vk::Extent3D{w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = vk::SampleCountFlagBits::e1;
        ci.tiling = tiling;
        ci.usage = usage;
        ci.sharingMode = vk::SharingMode::eExclusive;
        ci.initialLayout = vk::ImageLayout::eUndefined;
        t.image = device->createImageUnique(ci);

        const auto req = device->getImageMemoryRequirements(*t.image);
        vk::MemoryAllocateInfo ai{};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(physical_device, req.memoryTypeBits, mem_props);
        t.memory = device->allocateMemoryUnique(ai);
        device->bindImageMemory(*t.image, *t.memory, 0);

        vk::ImageViewCreateInfo vci{};
        vci.image = *t.image;
        vci.viewType = vk::ImageViewType::e2D;
        vci.format = format;
        vci.subresourceRange.aspectMask = aspect;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        t.view = device->createImageViewUnique(vci);

        t.layout = vk::ImageLayout::eUndefined;
        return t;
    }

    vk::UniquePipeline
    make_compute_pipeline(vk::ShaderModule sm, const char* entry, vk::PipelineLayout layout) {
        vk::PipelineShaderStageCreateInfo ss{};
        ss.stage = vk::ShaderStageFlagBits::eCompute;
        ss.module = sm;
        ss.pName = entry;
        vk::ComputePipelineCreateInfo ci{};
        ci.stage = ss;
        ci.layout = layout;
        auto r = device->createComputePipelineUnique({}, ci);
        if (r.result != vk::Result::eSuccess) {
            throw std::runtime_error("Vulkan: compute pipeline creation failed");
        }
        return std::move(r.value);
    }

    // One-shot upload for small UBO updates — called from render_frame.
    static void write_buffer(Buffer& b, const void* src, std::size_t n) {
        if (!b.mapped || n > b.size)
            return;
        std::memcpy(b.mapped, src, n);
    }
};

#else

struct VulkanBackend::Impl {
    // Non-Vulkan-Hpp stub state so the accessors below link cleanly on
    // machines without the SDK (the scaffolding path).
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::atomic<double> last_gpu_seconds{0.0};
    OverlayCallback overlay;
};

#endif  // SINGULARITY_VULKAN_HPP_AVAILABLE

// ===========================================================================
// Construction / destruction
// ===========================================================================

VulkanBackend::VulkanBackend() : impl_(std::make_unique<Impl>()) {}
VulkanBackend::~VulkanBackend() {
    shutdown();
}

// ===========================================================================
// initialize()
// ===========================================================================

bool VulkanBackend::initialize(WindowHandle window, RenderConfig config) {
#if !SINGULARITY_VULKAN_HPP_AVAILABLE
    (void)window;
    (void)config;
    warn_once("Vulkan-Hpp headers not available — backend is a stub.");
    return false;
#else
    impl_->width = config.width;
    impl_->height = config.height;
    impl_->vsync = config.vsync_enabled;
    impl_->spv_dir = resolve_spv_dir();

    try {
        // --- 1. Instance ---------------------------------------------------
        vk::ApplicationInfo app{};
        app.pApplicationName = "Singularity";
        app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        app.pEngineName = "Singularity";
        app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
        app.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char*> inst_exts{
            VK_KHR_SURFACE_EXTENSION_NAME,
        };
#if defined(_WIN32)
        inst_exts.push_back("VK_KHR_win32_surface");
#elif defined(__APPLE__)
        inst_exts.push_back("VK_EXT_metal_surface");
        inst_exts.push_back("VK_KHR_portability_enumeration");
#elif defined(__linux__)
        // SDL on Linux can use either Wayland or X11; we enable both extension
        // names so SDL_Vulkan_CreateSurface succeeds whichever backend it
        // chose at window-creation time. Missing extensions on a given
        // driver are silently dropped by the ICD list check below.
        inst_exts.push_back("VK_KHR_xlib_surface");
        inst_exts.push_back("VK_KHR_wayland_surface");
#endif

        // Drop any extension not advertised by the loader — avoids VK_ERROR
        // on machines that lack, e.g., Wayland.
        {
            const auto avail = vk::enumerateInstanceExtensionProperties();
            std::vector<const char*> kept;
            kept.reserve(inst_exts.size());
            for (const char* want : inst_exts) {
                for (const auto& have : avail) {
                    if (std::strcmp(want, have.extensionName) == 0) {
                        kept.push_back(want);
                        break;
                    }
                }
            }
            inst_exts = std::move(kept);
        }

        std::vector<const char*> inst_layers;
#ifndef NDEBUG
        {
            const auto avail = vk::enumerateInstanceLayerProperties();
            for (const auto& l : avail) {
                if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    inst_layers.push_back("VK_LAYER_KHRONOS_validation");
                    break;
                }
            }
        }
#endif

        vk::InstanceCreateInfo ici{};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = (std::uint32_t)inst_exts.size();
        ici.ppEnabledExtensionNames = inst_exts.data();
        ici.enabledLayerCount = (std::uint32_t)inst_layers.size();
        ici.ppEnabledLayerNames = inst_layers.data();
#if defined(__APPLE__)
        ici.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif
        impl_->instance = vk::createInstanceUnique(ici);

        // --- 4. Surface (before device so we can filter for present support)
        if (!window.vulkan_create_surface) {
            throw std::runtime_error("Vulkan: WindowHandle.vulkan_create_surface is null");
        }
        VkSurfaceKHR raw_surface = static_cast<VkSurfaceKHR>(window.vulkan_create_surface(
            static_cast<void*>(static_cast<VkInstance>(*impl_->instance))));
        if (!raw_surface) {
            throw std::runtime_error("Vulkan: surface creation callback returned null");
        }
        impl_->surface = vk::UniqueSurfaceKHR(raw_surface, *impl_->instance);

        // --- 2. Pick physical device --------------------------------------
        const auto physdevs = impl_->instance->enumeratePhysicalDevices();
        if (physdevs.empty()) {
            throw std::runtime_error("Vulkan: no physical devices");
        }
        auto score = [&](vk::PhysicalDevice pd, std::uint32_t* qfam_out) -> int {
            const auto props = pd.getProperties();
            const auto qprops = pd.getQueueFamilyProperties();
            std::uint32_t picked = UINT32_MAX;
            for (std::uint32_t i = 0; i < qprops.size(); ++i) {
                const bool graphics = (qprops[i].queueFlags & vk::QueueFlagBits::eGraphics)
                                      == vk::QueueFlagBits::eGraphics;
                const bool compute = (qprops[i].queueFlags & vk::QueueFlagBits::eCompute)
                                     == vk::QueueFlagBits::eCompute;
                const bool present = pd.getSurfaceSupportKHR(i, *impl_->surface);
                if (graphics && compute && present) {
                    picked = i;
                    break;
                }
            }
            if (picked == UINT32_MAX)
                return -1;
            // Require the swapchain extension.
            const auto exts = pd.enumerateDeviceExtensionProperties();
            bool has_sc = false;
            for (const auto& e : exts) {
                if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                    has_sc = true;
                    break;
                }
            }
            if (!has_sc)
                return -1;
            if (qfam_out)
                *qfam_out = picked;
            int s = 0;
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
                s += 1000;
            else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
                s += 500;
            s += (int)props.limits.maxImageDimension2D;
            return s;
        };
        int best = -1;
        for (auto pd : physdevs) {
            std::uint32_t qf = 0;
            int s = score(pd, &qf);
            if (s > best) {
                best = s;
                impl_->physical_device = pd;
                impl_->graphics_queue_family = qf;
            }
        }
        if (best < 0) {
            throw std::runtime_error("Vulkan: no suitable physical device");
        }
        const auto pdprops = impl_->physical_device.getProperties();
        impl_->timestamp_period_ns = pdprops.limits.timestampPeriod;

        // --- 3. Logical device --------------------------------------------
        const float qpri = 1.0f;
        vk::DeviceQueueCreateInfo dqci{};
        dqci.queueFamilyIndex = impl_->graphics_queue_family;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &qpri;

        std::vector<const char*> dev_exts{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#if defined(__APPLE__)
        dev_exts.push_back("VK_KHR_portability_subset");
#endif
        // Filter out extensions not advertised, same pattern as instance.
        {
            const auto avail = impl_->physical_device.enumerateDeviceExtensionProperties();
            std::vector<const char*> kept;
            for (const char* want : dev_exts) {
                for (const auto& have : avail) {
                    if (std::strcmp(want, have.extensionName) == 0) {
                        kept.push_back(want);
                        break;
                    }
                }
            }
            dev_exts = std::move(kept);
        }

        vk::DeviceCreateInfo dci{};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqci;
        dci.enabledExtensionCount = (std::uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.data();
        impl_->device = impl_->physical_device.createDeviceUnique(dci);
        impl_->graphics_queue = impl_->device->getQueue(impl_->graphics_queue_family, 0);

        // --- 5,6. Swapchain + image views ---------------------------------
        build_swapchain_and_views();

        // --- 7. Descriptor set layouts + pipelines + render pass ---------
        build_descriptor_layouts();
        build_render_pass();
        build_pipelines();
        build_sampler();

        // --- 8. Offscreen HDR + bloom + framebuffers ----------------------
        create_hdr_and_bloom();
        create_framebuffers();

        // --- 9. Descriptor pool, per-frame sync + cmd buffers + UBOs ------
        create_descriptor_pool();
        create_per_frame_resources();

        // Capture infrastructure — allocated lazily on first call.
        vk::CommandPoolCreateInfo cpci{};
        cpci.queueFamilyIndex = impl_->graphics_queue_family;
        cpci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer
                     | vk::CommandPoolCreateFlagBits::eTransient;
        impl_->capture_pool = impl_->device->createCommandPoolUnique(cpci);
        vk::CommandBufferAllocateInfo cbai{};
        cbai.commandPool = *impl_->capture_pool;
        cbai.level = vk::CommandBufferLevel::ePrimary;
        cbai.commandBufferCount = 1;
        auto bufs = impl_->device->allocateCommandBuffersUnique(cbai);
        impl_->capture_cmd = std::move(bufs[0]);
        impl_->capture_fence = impl_->device->createFenceUnique({});

        // Reset the animation clock so elapsed-since-init is measured from
        // the moment the backend is fully ready, not from the Impl ctor.
        impl_->start_time = std::chrono::steady_clock::now();

        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[singularity/vulkan] initialize failed: %s\n", e.what());
        impl_ = std::make_unique<Impl>();
        return false;
    }
#endif
}

// ===========================================================================
// Private helpers (initialize splits into these)
// ===========================================================================

#if SINGULARITY_VULKAN_HPP_AVAILABLE

// The below are free-function-like helpers living as static methods on
// VulkanBackend so they can access impl_ without forwarding everything.
// They're declared inside the anon namespace? No — inline-implemented here
// as private methods would require header churn. Use a PImpl pattern:
// they read/write impl_ directly.

void VulkanBackend::build_swapchain_and_views() {
    const auto caps = impl_->physical_device.getSurfaceCapabilitiesKHR(*impl_->surface);
    const auto fmts = impl_->physical_device.getSurfaceFormatsKHR(*impl_->surface);
    const auto modes = impl_->physical_device.getSurfacePresentModesKHR(*impl_->surface);

    // Prefer BGRA8_SRGB; fall back to the first supported format.
    vk::SurfaceFormatKHR surface_format =
        fmts.empty()
            ? vk::SurfaceFormatKHR{vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear}
            : fmts[0];
    for (const auto& f : fmts) {
        if (f.format == vk::Format::eB8G8R8A8Srgb
            && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            surface_format = f;
            break;
        }
    }
    impl_->swapchain_format = surface_format.format;

    // Present mode: FIFO for vsync (guaranteed), MAILBOX/IMMEDIATE for no-vsync.
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    if (!impl_->vsync) {
        for (auto m : modes) {
            if (m == vk::PresentModeKHR::eMailbox) {
                present_mode = m;
                break;
            }
            if (m == vk::PresentModeKHR::eImmediate)
                present_mode = m;
        }
    }

    // Extent: use window size clamped to surface caps.
    vk::Extent2D extent;
    if (caps.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        extent = caps.currentExtent;
    } else {
        extent.width =
            std::clamp(impl_->width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height =
            std::clamp(impl_->height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    impl_->swapchain_extent = extent;
    impl_->width = extent.width;
    impl_->height = extent.height;

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR sci{};
    sci.surface = *impl_->surface;
    sci.minImageCount = image_count;
    sci.imageFormat = surface_format.format;
    sci.imageColorSpace = surface_format.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage =
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
    sci.imageSharingMode = vk::SharingMode::eExclusive;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    sci.presentMode = present_mode;
    sci.clipped = VK_TRUE;
    impl_->swapchain = impl_->device->createSwapchainKHRUnique(sci);

    impl_->swapchain_images = impl_->device->getSwapchainImagesKHR(*impl_->swapchain);
    impl_->swapchain_views.clear();
    impl_->swapchain_views.reserve(impl_->swapchain_images.size());
    for (auto img : impl_->swapchain_images) {
        vk::ImageViewCreateInfo vci{};
        vci.image = img;
        vci.viewType = vk::ImageViewType::e2D;
        vci.format = surface_format.format;
        vci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        impl_->swapchain_views.push_back(impl_->device->createImageViewUnique(vci));
    }
}

void VulkanBackend::build_descriptor_layouts() {
    // Geodesic: binding 0 = UBO, binding 1 = storage image
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> b{};
        b[0].binding = 0;
        b[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        b[0].descriptorCount = 1;
        b[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        b[1].binding = 1;
        b[1].descriptorType = vk::DescriptorType::eStorageImage;
        b[1].descriptorCount = 1;
        b[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
        vk::DescriptorSetLayoutCreateInfo ci{};
        ci.bindingCount = (std::uint32_t)b.size();
        ci.pBindings = b.data();
        impl_->geodesic_dsl = impl_->device->createDescriptorSetLayoutUnique(ci);
    }
    // Bloom: binding 0 = sampled, 1 = storage, 2 = UBO
    {
        std::array<vk::DescriptorSetLayoutBinding, 3> b{};
        b[0].binding = 0;
        b[0].descriptorType = vk::DescriptorType::eSampledImage;
        b[0].descriptorCount = 1;
        b[0].stageFlags = vk::ShaderStageFlagBits::eCompute;
        b[1].binding = 1;
        b[1].descriptorType = vk::DescriptorType::eStorageImage;
        b[1].descriptorCount = 1;
        b[1].stageFlags = vk::ShaderStageFlagBits::eCompute;
        b[2].binding = 2;
        b[2].descriptorType = vk::DescriptorType::eUniformBuffer;
        b[2].descriptorCount = 1;
        b[2].stageFlags = vk::ShaderStageFlagBits::eCompute;
        vk::DescriptorSetLayoutCreateInfo ci{};
        ci.bindingCount = (std::uint32_t)b.size();
        ci.pBindings = b.data();
        impl_->bloom_dsl = impl_->device->createDescriptorSetLayoutUnique(ci);
    }
    // Blit fragment: 0 = sampled (hdr), 1 = sampled (bloom), 2 = sampler, 3 = UBO
    {
        std::array<vk::DescriptorSetLayoutBinding, 4> b{};
        b[0].binding = 0;
        b[0].descriptorType = vk::DescriptorType::eSampledImage;
        b[0].descriptorCount = 1;
        b[0].stageFlags = vk::ShaderStageFlagBits::eFragment;
        b[1].binding = 1;
        b[1].descriptorType = vk::DescriptorType::eSampledImage;
        b[1].descriptorCount = 1;
        b[1].stageFlags = vk::ShaderStageFlagBits::eFragment;
        b[2].binding = 2;
        b[2].descriptorType = vk::DescriptorType::eSampler;
        b[2].descriptorCount = 1;
        b[2].stageFlags = vk::ShaderStageFlagBits::eFragment;
        b[3].binding = 3;
        b[3].descriptorType = vk::DescriptorType::eUniformBuffer;
        b[3].descriptorCount = 1;
        b[3].stageFlags = vk::ShaderStageFlagBits::eFragment;
        vk::DescriptorSetLayoutCreateInfo ci{};
        ci.bindingCount = (std::uint32_t)b.size();
        ci.pBindings = b.data();
        impl_->blit_dsl = impl_->device->createDescriptorSetLayoutUnique(ci);
    }

    // Pipeline layouts.
    auto make_pl = [this](vk::DescriptorSetLayout dsl) {
        vk::PipelineLayoutCreateInfo ci{};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &dsl;
        return impl_->device->createPipelineLayoutUnique(ci);
    };
    impl_->geodesic_pl = make_pl(*impl_->geodesic_dsl);
    impl_->bloom_pl = make_pl(*impl_->bloom_dsl);
    impl_->blit_pl = make_pl(*impl_->blit_dsl);
}

void VulkanBackend::build_render_pass() {
    vk::AttachmentDescription color{};
    color.format = impl_->swapchain_format;
    color.samples = vk::SampleCountFlagBits::e1;
    color.loadOp = vk::AttachmentLoadOp::eDontCare;
    color.storeOp = vk::AttachmentStoreOp::eStore;
    color.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    color.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    color.initialLayout = vk::ImageLayout::eUndefined;
    color.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription sub{};
    sub.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &color_ref;

    vk::SubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.srcAccessMask = vk::AccessFlags{};
    dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo rpci{};
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;
    impl_->blit_render_pass = impl_->device->createRenderPassUnique(rpci);
}

void VulkanBackend::build_pipelines() {
    // Compute pipelines: geodesic + 3 bloom entries.
    {
        auto sm = load_shader_module(*impl_->device, impl_->spv_dir / "geodesic_kernel.spv");
        impl_->geodesic_pipeline = impl_->make_compute_pipeline(*sm, "main", *impl_->geodesic_pl);
    }
    {
        auto sm = load_shader_module(*impl_->device, impl_->spv_dir / "bloom_extract.spv");
        impl_->bloom_extract_pipeline =
            impl_->make_compute_pipeline(*sm, "extract", *impl_->bloom_pl);
    }
    {
        auto sm = load_shader_module(*impl_->device, impl_->spv_dir / "bloom_blur_h.spv");
        impl_->bloom_blur_h_pipeline =
            impl_->make_compute_pipeline(*sm, "blur_h", *impl_->bloom_pl);
    }
    {
        auto sm = load_shader_module(*impl_->device, impl_->spv_dir / "bloom_blur_v.spv");
        impl_->bloom_blur_v_pipeline =
            impl_->make_compute_pipeline(*sm, "blur_v", *impl_->bloom_pl);
    }

    // Blit graphics pipeline: VS + PS, fullscreen triangle, 3 verts, no inputs.
    auto vs_mod = load_shader_module(*impl_->device, impl_->spv_dir / "blit_vs.spv");
    auto ps_mod = load_shader_module(*impl_->device, impl_->spv_dir / "blit_ps.spv");

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages{};
    stages[0].stage = vk::ShaderStageFlagBits::eVertex;
    stages[0].module = *vs_mod;
    stages[0].pName = "main_vs";
    stages[1].stage = vk::ShaderStageFlagBits::eFragment;
    stages[1].module = *ps_mod;
    stages[1].pName = "main_ps";

    vk::PipelineVertexInputStateCreateInfo vi{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.topology = vk::PrimitiveTopology::eTriangleList;

    // Dynamic viewport + scissor avoid a recompile on resize.
    std::array<vk::DynamicState, 2> dynamics{vk::DynamicState::eViewport,
                                             vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.dynamicStateCount = (std::uint32_t)dynamics.size();
    dyn.pDynamicStates = dynamics.data();
    vk::PipelineViewportStateCreateInfo vp{};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.polygonMode = vk::PolygonMode::eFill;
    rs.cullMode = vk::CullModeFlagBits::eNone;
    rs.frontFace = vk::FrontFace::eCounterClockwise;
    rs.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                         | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    cba.blendEnable = VK_FALSE;
    vk::PipelineColorBlendStateCreateInfo cb{};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    vk::GraphicsPipelineCreateInfo gci{};
    gci.stageCount = (std::uint32_t)stages.size();
    gci.pStages = stages.data();
    gci.pVertexInputState = &vi;
    gci.pInputAssemblyState = &ia;
    gci.pViewportState = &vp;
    gci.pRasterizationState = &rs;
    gci.pMultisampleState = &ms;
    gci.pColorBlendState = &cb;
    gci.pDynamicState = &dyn;
    gci.layout = *impl_->blit_pl;
    gci.renderPass = *impl_->blit_render_pass;
    gci.subpass = 0;
    auto r = impl_->device->createGraphicsPipelineUnique({}, gci);
    if (r.result != vk::Result::eSuccess) {
        throw std::runtime_error("Vulkan: blit pipeline creation failed");
    }
    impl_->blit_pipeline = std::move(r.value);
}

void VulkanBackend::build_sampler() {
    vk::SamplerCreateInfo ci{};
    ci.magFilter = vk::Filter::eLinear;
    ci.minFilter = vk::Filter::eLinear;
    ci.mipmapMode = vk::SamplerMipmapMode::eLinear;
    ci.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    ci.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    ci.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    ci.maxLod = 1.0f;
    impl_->linear_sampler = impl_->device->createSamplerUnique(ci);
}

void VulkanBackend::create_hdr_and_bloom() {
    const std::uint32_t bw = std::max<std::uint32_t>(1, impl_->width / 2);
    const std::uint32_t bh = std::max<std::uint32_t>(1, impl_->height / 2);
    impl_->hdr = impl_->create_image_target(impl_->width,
                                            impl_->height,
                                            vk::Format::eR16G16B16A16Sfloat,
                                            vk::ImageUsageFlagBits::eStorage
                                                | vk::ImageUsageFlagBits::eSampled
                                                | vk::ImageUsageFlagBits::eTransferSrc,
                                            vk::ImageAspectFlagBits::eColor);
    impl_->bloom_a = impl_->create_image_target(bw,
                                                bh,
                                                vk::Format::eR16G16B16A16Sfloat,
                                                vk::ImageUsageFlagBits::eStorage
                                                    | vk::ImageUsageFlagBits::eSampled,
                                                vk::ImageAspectFlagBits::eColor);
    impl_->bloom_b = impl_->create_image_target(bw,
                                                bh,
                                                vk::Format::eR16G16B16A16Sfloat,
                                                vk::ImageUsageFlagBits::eStorage
                                                    | vk::ImageUsageFlagBits::eSampled,
                                                vk::ImageAspectFlagBits::eColor);
}

void VulkanBackend::create_framebuffers() {
    impl_->swapchain_framebuffers.clear();
    impl_->swapchain_framebuffers.reserve(impl_->swapchain_views.size());
    for (auto& v : impl_->swapchain_views) {
        vk::ImageView att = *v;
        vk::FramebufferCreateInfo fci{};
        fci.renderPass = *impl_->blit_render_pass;
        fci.attachmentCount = 1;
        fci.pAttachments = &att;
        fci.width = impl_->swapchain_extent.width;
        fci.height = impl_->swapchain_extent.height;
        fci.layers = 1;
        impl_->swapchain_framebuffers.push_back(impl_->device->createFramebufferUnique(fci));
    }
}

void VulkanBackend::create_descriptor_pool() {
    // One set per pipeline (5) per frame in flight (3) = 15 sets.
    // Buffer + image counts sized accordingly.
    std::array<vk::DescriptorPoolSize, 4> pool_sizes{{
        {vk::DescriptorType::eUniformBuffer, (std::uint32_t)(kMaxFramesInFlight * 6)},
        {vk::DescriptorType::eStorageImage, (std::uint32_t)(kMaxFramesInFlight * 4)},
        {vk::DescriptorType::eSampledImage, (std::uint32_t)(kMaxFramesInFlight * 6)},
        {vk::DescriptorType::eSampler, (std::uint32_t)(kMaxFramesInFlight * 2)},
    }};
    vk::DescriptorPoolCreateInfo ci{};
    ci.maxSets = (std::uint32_t)(kMaxFramesInFlight * 5);
    ci.poolSizeCount = (std::uint32_t)pool_sizes.size();
    ci.pPoolSizes = pool_sizes.data();
    ci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    impl_->descriptor_pool = impl_->device->createDescriptorPoolUnique(ci);
}

void VulkanBackend::create_per_frame_resources() {
    vk::CommandPoolCreateInfo cpci{};
    cpci.queueFamilyIndex = impl_->graphics_queue_family;
    cpci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
        auto& f = impl_->frames[i];
        f.pool = impl_->device->createCommandPoolUnique(cpci);
        vk::CommandBufferAllocateInfo cbai{};
        cbai.commandPool = *f.pool;
        cbai.level = vk::CommandBufferLevel::ePrimary;
        cbai.commandBufferCount = 1;
        auto bufs = impl_->device->allocateCommandBuffersUnique(cbai);
        f.cmd = std::move(bufs[0]);

        f.image_available = impl_->device->createSemaphoreUnique({});
        f.render_finished = impl_->device->createSemaphoreUnique({});
        vk::FenceCreateInfo fci{};
        fci.flags = vk::FenceCreateFlagBits::eSignaled;
        f.in_flight = impl_->device->createFenceUnique(fci);

        f.uniforms = impl_->create_buffer(sizeof(Uniforms),
                                          vk::BufferUsageFlagBits::eUniformBuffer,
                                          vk::MemoryPropertyFlagBits::eHostVisible
                                              | vk::MemoryPropertyFlagBits::eHostCoherent,
                                          /*map_persistent=*/true);
        f.bloom_params = impl_->create_buffer(sizeof(BloomParamsCpu),
                                              vk::BufferUsageFlagBits::eUniformBuffer,
                                              vk::MemoryPropertyFlagBits::eHostVisible
                                                  | vk::MemoryPropertyFlagBits::eHostCoherent,
                                              /*map_persistent=*/true);
        f.blit_params = impl_->create_buffer(sizeof(BlitParamsCpu),
                                             vk::BufferUsageFlagBits::eUniformBuffer,
                                             vk::MemoryPropertyFlagBits::eHostVisible
                                                 | vk::MemoryPropertyFlagBits::eHostCoherent,
                                             /*map_persistent=*/true);

        vk::QueryPoolCreateInfo qpci{};
        qpci.queryType = vk::QueryType::eTimestamp;
        qpci.queryCount = kTimestampsPerFrame;
        f.timestamp_pool = impl_->device->createQueryPoolUnique(qpci);
    }

    // Allocate descriptor sets for every frame × every pipeline.
    const std::array<vk::DescriptorSetLayout, 5> layouts_single{
        *impl_->geodesic_dsl,
        *impl_->bloom_dsl,
        *impl_->bloom_dsl,
        *impl_->bloom_dsl,
        *impl_->blit_dsl,
    };
    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
        vk::DescriptorSetAllocateInfo ai{};
        ai.descriptorPool = *impl_->descriptor_pool;
        ai.descriptorSetCount = (std::uint32_t)layouts_single.size();
        ai.pSetLayouts = layouts_single.data();
        const auto sets = impl_->device->allocateDescriptorSets(ai);
        auto& f = impl_->frames[i];
        f.geodesic_ds = sets[0];
        f.bloom_extract_ds = sets[1];
        f.bloom_blur_h_ds = sets[2];
        f.bloom_blur_v_ds = sets[3];
        f.blit_ds = sets[4];
    }

    // Wire image/buffer -> descriptor-set bindings. Buffer bindings
    // (uniforms, bloom_params, blit_params) are stable across frames and
    // resizes. Image bindings are stable until resize() recreates them.
    update_descriptor_sets();
}

void VulkanBackend::update_descriptor_sets() {
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<vk::DescriptorBufferInfo> buf_infos;
    std::vector<vk::DescriptorImageInfo> img_infos;
    // Reserve generously to prevent reallocation invalidating pointers
    // we've already inserted into `writes`.
    buf_infos.reserve(kMaxFramesInFlight * 8);
    img_infos.reserve(kMaxFramesInFlight * 12);

    const vk::ImageLayout sampled_ro = vk::ImageLayout::eShaderReadOnlyOptimal;
    const vk::ImageLayout storage_ly = vk::ImageLayout::eGeneral;

    for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
        auto& f = impl_->frames[i];

        // --- Geodesic: UBO(u), storage-image(output=hdr) ------------------
        {
            buf_infos.push_back({*f.uniforms.buffer, 0, VK_WHOLE_SIZE});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.geodesic_ds;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buf_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({nullptr, *impl_->hdr.view, storage_ly});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.geodesic_ds;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eStorageImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }

        // --- Bloom extract: sampled(hdr), storage(bloom_a), ubo(bloom_params)
        {
            img_infos.push_back({nullptr, *impl_->hdr.view, sampled_ro});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_extract_ds;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({nullptr, *impl_->bloom_a.view, storage_ly});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_extract_ds;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eStorageImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            buf_infos.push_back({*f.bloom_params.buffer, 0, VK_WHOLE_SIZE});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_extract_ds;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buf_infos.back();
            writes.push_back(w);
        }

        // --- Bloom blur H: sampled(bloom_a), storage(bloom_b), ubo ---------
        {
            img_infos.push_back({nullptr, *impl_->bloom_a.view, sampled_ro});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_h_ds;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({nullptr, *impl_->bloom_b.view, storage_ly});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_h_ds;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eStorageImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            buf_infos.push_back({*f.bloom_params.buffer, 0, VK_WHOLE_SIZE});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_h_ds;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buf_infos.back();
            writes.push_back(w);
        }

        // --- Bloom blur V: sampled(bloom_b), storage(bloom_a), ubo ---------
        {
            img_infos.push_back({nullptr, *impl_->bloom_b.view, sampled_ro});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_v_ds;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({nullptr, *impl_->bloom_a.view, storage_ly});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_v_ds;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eStorageImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            buf_infos.push_back({*f.bloom_params.buffer, 0, VK_WHOLE_SIZE});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.bloom_blur_v_ds;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buf_infos.back();
            writes.push_back(w);
        }

        // --- Blit: sampled(hdr), sampled(bloom_a), sampler, ubo(blit_params)
        {
            img_infos.push_back({nullptr, *impl_->hdr.view, sampled_ro});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.blit_ds;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({nullptr, *impl_->bloom_a.view, sampled_ro});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.blit_ds;
            w.dstBinding = 1;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampledImage;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            img_infos.push_back({*impl_->linear_sampler, nullptr, vk::ImageLayout::eUndefined});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.blit_ds;
            w.dstBinding = 2;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eSampler;
            w.pImageInfo = &img_infos.back();
            writes.push_back(w);
        }
        {
            buf_infos.push_back({*f.blit_params.buffer, 0, VK_WHOLE_SIZE});
            vk::WriteDescriptorSet w{};
            w.dstSet = f.blit_ds;
            w.dstBinding = 3;
            w.descriptorCount = 1;
            w.descriptorType = vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buf_infos.back();
            writes.push_back(w);
        }
    }
    impl_->device->updateDescriptorSets(writes, {});
}

#endif  // SINGULARITY_VULKAN_HPP_AVAILABLE

// ===========================================================================
// shutdown()
// ===========================================================================

void VulkanBackend::shutdown() {
#if SINGULARITY_VULKAN_HPP_AVAILABLE
    if (!impl_)
        return;
    if (impl_->device) {
        impl_->device->waitIdle();
    }
    // UniqueX destructors tear down in reverse-construction order.
#endif
}

// ===========================================================================
// resize()
// ===========================================================================

void VulkanBackend::resize(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0)
        return;
    if (width == impl_->width && height == impl_->height)
        return;
    impl_->width = width;
    impl_->height = height;
#if SINGULARITY_VULKAN_HPP_AVAILABLE
    if (!impl_->device)
        return;
    impl_->device->waitIdle();

    impl_->swapchain_framebuffers.clear();
    impl_->swapchain_views.clear();
    impl_->swapchain_images.clear();
    impl_->swapchain.reset();

    build_swapchain_and_views();

    // Recreate HDR + bloom at the new size.
    impl_->hdr.view.reset();
    impl_->hdr.memory.reset();
    impl_->hdr.image.reset();
    impl_->bloom_a.view.reset();
    impl_->bloom_a.memory.reset();
    impl_->bloom_a.image.reset();
    impl_->bloom_b.view.reset();
    impl_->bloom_b.memory.reset();
    impl_->bloom_b.image.reset();
    create_hdr_and_bloom();
    create_framebuffers();

    // Pipelines, descriptor sets, descriptor-set layouts, UBOs survive.
    // We only need to rebind the image views in the descriptor sets because
    // the views were rebuilt.
    update_descriptor_sets();

    // Capture staging — invalidate so the next capture allocates at the
    // new size.
    impl_->capture_staging.view.reset();
    impl_->capture_staging.memory.reset();
    impl_->capture_staging.image.reset();
#endif
}

// ===========================================================================
// render_frame()
// ===========================================================================

void VulkanBackend::render_frame(const Scene& scene, const CameraState& camera) {
#if !SINGULARITY_VULKAN_HPP_AVAILABLE
    (void)scene;
    (void)camera;
    warn_once("render_frame() — no Vulkan SDK at build time.");
    return;
#else
    if (!impl_->device || !impl_->swapchain)
        return;

    const std::size_t slot = impl_->frame % kMaxFramesInFlight;
    auto& f = impl_->frames[slot];

    // Wait for this frame's prior submission to complete.
    {
        const auto r = impl_->device->waitForFences(
            *f.in_flight, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
        (void)r;
    }

    // Read back timestamps from the previous submission in this slot.
    if (f.timestamps_submitted) {
        std::array<std::uint64_t, kTimestampsPerFrame> ts{};
        const auto qr = impl_->device->getQueryPoolResults(*f.timestamp_pool,
                                                           0,
                                                           kTimestampsPerFrame,
                                                           sizeof(ts),
                                                           ts.data(),
                                                           sizeof(std::uint64_t),
                                                           vk::QueryResultFlagBits::e64);
        if (qr == vk::Result::eSuccess && ts[1] > ts[0]) {
            const double ns = double(ts[1] - ts[0]) * double(impl_->timestamp_period_ns);
            impl_->last_gpu_seconds.store(ns * 1e-9, std::memory_order_relaxed);
        }
    }

    // Acquire next swapchain image. On OUT_OF_DATE, swallow and try again
    // next frame (the window will have sent a resize event).
    std::uint32_t image_index = 0;
    {
        vk::Result r;
        try {
            auto acquired =
                impl_->device->acquireNextImageKHR(*impl_->swapchain,
                                                   std::numeric_limits<std::uint64_t>::max(),
                                                   *f.image_available,
                                                   nullptr);
            r = acquired.result;
            image_index = acquired.value;
        } catch (const vk::OutOfDateKHRError&) {
            return;
        }
        if (r == vk::Result::eErrorOutOfDateKHR)
            return;
    }
    impl_->image_index = image_index;

    // Reset the fence AFTER confirming we have work to submit.
    impl_->device->resetFences(*f.in_flight);

    // Upload uniforms + bloom/blit params.
    Uniforms u{};
    const float elapsed =
        std::chrono::duration<float>(std::chrono::steady_clock::now() - impl_->start_time).count();
    pack_uniforms(u, scene, camera, impl_->width, impl_->height, impl_->frame, elapsed);
    Impl::write_buffer(f.uniforms, &u, sizeof(u));
    const BloomParamsCpu bp{scene.bloom_threshold, 0.0f, 0.0f, 0.0f};
    Impl::write_buffer(f.bloom_params, &bp, sizeof(bp));
    const BlitParamsCpu blp{scene.exposure, scene.bloom_strength, 0.0f, 0.0f};
    Impl::write_buffer(f.blit_params, &blp, sizeof(blp));

    // Record the command buffer.
    vk::CommandBuffer cmd = *f.cmd;
    cmd.reset({});
    vk::CommandBufferBeginInfo cbi{};
    cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(cbi);

    cmd.resetQueryPool(*f.timestamp_pool, 0, kTimestampsPerFrame);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *f.timestamp_pool, 0);

    // Transition HDR + bloom images to GENERAL for compute writes.
    transition_image(cmd,
                     *impl_->hdr.image,
                     vk::ImageLayout::eUndefined,
                     vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {},
                     vk::AccessFlagBits::eShaderWrite);
    transition_image(cmd,
                     *impl_->bloom_a.image,
                     vk::ImageLayout::eUndefined,
                     vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {},
                     vk::AccessFlagBits::eShaderWrite);
    transition_image(cmd,
                     *impl_->bloom_b.image,
                     vk::ImageLayout::eUndefined,
                     vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTopOfPipe,
                     vk::PipelineStageFlagBits::eComputeShader,
                     {},
                     vk::AccessFlagBits::eShaderWrite);

    // Geodesic dispatch.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *impl_->geodesic_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *impl_->geodesic_pl, 0, {f.geodesic_ds}, {});
    const std::uint32_t gw = (impl_->width + 7) / 8;
    const std::uint32_t gh = (impl_->height + 7) / 8;
    cmd.dispatch(gw, gh, 1);

    // Barrier: HDR write -> HDR read by bloom/blit + by next geodesic frame.
    {
        vk::ImageMemoryBarrier b{};
        b.oldLayout = vk::ImageLayout::eGeneral;
        b.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = *impl_->hdr.image;
        b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        b.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader
                                | vk::PipelineStageFlagBits::eFragmentShader,
                            {},
                            {},
                            {},
                            b);
    }

    const std::uint32_t bw = std::max<std::uint32_t>(1, impl_->width / 2);
    const std::uint32_t bh = std::max<std::uint32_t>(1, impl_->height / 2);
    const std::uint32_t bgw = (bw + 7) / 8;
    const std::uint32_t bgh = (bh + 7) / 8;

    // Bloom extract: HDR -> bloom_a.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *impl_->bloom_extract_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *impl_->bloom_pl, 0, {f.bloom_extract_ds}, {});
    cmd.dispatch(bgw, bgh, 1);

    // Barrier: bloom_a write -> read.
    auto bloom_barrier = [&](vk::Image img) {
        vk::ImageMemoryBarrier b{};
        b.oldLayout = vk::ImageLayout::eGeneral;
        b.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        b.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            {},
                            {},
                            b);
    };
    auto reverse_barrier = [&](vk::Image img) {
        vk::ImageMemoryBarrier b{};
        b.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        b.newLayout = vk::ImageLayout::eGeneral;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        b.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        b.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            {},
                            {},
                            b);
    };
    bloom_barrier(*impl_->bloom_a.image);

    // Bloom blur H: bloom_a -> bloom_b.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *impl_->bloom_blur_h_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *impl_->bloom_pl, 0, {f.bloom_blur_h_ds}, {});
    cmd.dispatch(bgw, bgh, 1);
    bloom_barrier(*impl_->bloom_b.image);
    // bloom_a goes back to GENERAL for blur_v's write.
    reverse_barrier(*impl_->bloom_a.image);

    // Bloom blur V: bloom_b -> bloom_a.
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *impl_->bloom_blur_v_pipeline);
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *impl_->bloom_pl, 0, {f.bloom_blur_v_ds}, {});
    cmd.dispatch(bgw, bgh, 1);
    bloom_barrier(*impl_->bloom_a.image);  // final bloom in bloom_a

    // --- Blit render pass ---------------------------------------------
    vk::ClearValue clear_color{};
    clear_color.color = vk::ClearColorValue{std::array<float, 4>{0.f, 0.f, 0.f, 1.f}};
    vk::RenderPassBeginInfo rpbi{};
    rpbi.renderPass = *impl_->blit_render_pass;
    rpbi.framebuffer = *impl_->swapchain_framebuffers[image_index];
    rpbi.renderArea.offset = vk::Offset2D{0, 0};
    rpbi.renderArea.extent = impl_->swapchain_extent;
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear_color;
    cmd.beginRenderPass(rpbi, vk::SubpassContents::eInline);

    vk::Viewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width = float(impl_->swapchain_extent.width);
    vp.height = float(impl_->swapchain_extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    cmd.setViewport(0, vp);
    vk::Rect2D sc{{0, 0}, impl_->swapchain_extent};
    cmd.setScissor(0, sc);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *impl_->blit_pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *impl_->blit_pl, 0, {f.blit_ds}, {});
    cmd.draw(3, 1, 0, 0);

    if (impl_->overlay) {
        // Pass the VkHandle *values* as void*. Vulkan dispatchable handles
        // (VkCommandBuffer, VkRenderPass, VkFramebuffer) are already typedef'd
        // to pointer-like types, so casting the raw handle through void*
        // round-trips cleanly via `(VkCommandBuffer)cb_void` on the other
        // side.
        auto fb_raw = static_cast<VkFramebuffer>(*impl_->swapchain_framebuffers[image_index]);
        auto rp_raw = static_cast<VkRenderPass>(*impl_->blit_render_pass);
        auto cb_raw = static_cast<VkCommandBuffer>(cmd);
        impl_->overlay(reinterpret_cast<void*>(fb_raw),
                       reinterpret_cast<void*>(cb_raw),
                       reinterpret_cast<void*>(rp_raw));
    }

    cmd.endRenderPass();

    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *f.timestamp_pool, 1);
    cmd.end();

    // Submit.
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo si{};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &(*f.image_available);
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &(*f.render_finished);
    impl_->graphics_queue.submit(si, *f.in_flight);
    f.timestamps_submitted = true;

    // Present.
    vk::PresentInfoKHR pi{};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &(*f.render_finished);
    pi.swapchainCount = 1;
    pi.pSwapchains = &(*impl_->swapchain);
    pi.pImageIndices = &image_index;
    try {
        const auto r = impl_->graphics_queue.presentKHR(pi);
        (void)r;
    } catch (const vk::OutOfDateKHRError&) {
        // Swallow — next render_frame will re-acquire, resize event will
        // drive the actual swapchain rebuild.
    }

    ++impl_->frame;
#endif
}

// ===========================================================================
// capture_frame()
// ===========================================================================

ImageData VulkanBackend::capture_frame() {
    ImageData out;
    out.width = impl_->width;
    out.height = impl_->height;
    out.pixels_rgba.assign(std::size_t(out.width) * out.height * 4, 0);
#if SINGULARITY_VULKAN_HPP_AVAILABLE
    if (!impl_->device || impl_->swapchain_images.empty())
        return out;

    impl_->device->waitIdle();

    // Allocate or reuse the staging image at current size. Use R8G8B8A8_UNORM
    // (we source from the swapchain whose format is _SRGB; Vulkan will do the
    // sRGB->linear decode at sample time for HDR-like uses, but a plain image
    // copy is bit-exact and gives us the sRGB-encoded bytes directly, which
    // is what stb_image_write wants).
    if (!impl_->capture_staging.image
        || impl_->capture_staging.layout == vk::ImageLayout::eUndefined) {
        impl_->capture_staging = impl_->create_image_target(
            out.width,
            out.height,
            vk::Format::eR8G8B8A8Unorm,
            vk::ImageUsageFlagBits::eTransferDst,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageTiling::eLinear,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    // Source: the last-presented swapchain image.
    const std::uint32_t src_idx = (impl_->frame == 0) ? 0 : (impl_->image_index);
    vk::Image src_image = impl_->swapchain_images[src_idx];

    vk::CommandBuffer cmd = *impl_->capture_cmd;
    cmd.reset({});
    vk::CommandBufferBeginInfo cbi{};
    cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(cbi);

    // src: PRESENT_SRC -> TRANSFER_SRC_OPTIMAL
    transition_image(cmd,
                     src_image,
                     vk::ImageLayout::ePresentSrcKHR,
                     vk::ImageLayout::eTransferSrcOptimal,
                     vk::PipelineStageFlagBits::eBottomOfPipe,
                     vk::PipelineStageFlagBits::eTransfer,
                     vk::AccessFlagBits::eMemoryRead,
                     vk::AccessFlagBits::eTransferRead);
    // dst: UNDEFINED (or GENERAL) -> TRANSFER_DST_OPTIMAL
    transition_image(cmd,
                     *impl_->capture_staging.image,
                     vk::ImageLayout::eUndefined,
                     vk::ImageLayout::eTransferDstOptimal,
                     vk::PipelineStageFlagBits::eTopOfPipe,
                     vk::PipelineStageFlagBits::eTransfer,
                     {},
                     vk::AccessFlagBits::eTransferWrite);

    // Blit (handles format conversion BGRA_SRGB -> RGBA_UNORM via copy
    // semantics: we use vkCmdBlitImage to let the driver swizzle + linearize).
    vk::ImageBlit blit{};
    blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.srcOffsets[1] = vk::Offset3D{(std::int32_t)impl_->swapchain_extent.width,
                                      (std::int32_t)impl_->swapchain_extent.height,
                                      1};
    blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.dstOffsets[1] = vk::Offset3D{(std::int32_t)out.width, (std::int32_t)out.height, 1};
    cmd.blitImage(src_image,
                  vk::ImageLayout::eTransferSrcOptimal,
                  *impl_->capture_staging.image,
                  vk::ImageLayout::eTransferDstOptimal,
                  blit,
                  vk::Filter::eLinear);

    // dst: TRANSFER_DST -> GENERAL (for host read)
    transition_image(cmd,
                     *impl_->capture_staging.image,
                     vk::ImageLayout::eTransferDstOptimal,
                     vk::ImageLayout::eGeneral,
                     vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eHost,
                     vk::AccessFlagBits::eTransferWrite,
                     vk::AccessFlagBits::eHostRead);
    // src back to PRESENT
    transition_image(cmd,
                     src_image,
                     vk::ImageLayout::eTransferSrcOptimal,
                     vk::ImageLayout::ePresentSrcKHR,
                     vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eBottomOfPipe,
                     vk::AccessFlagBits::eTransferRead,
                     vk::AccessFlagBits::eMemoryRead);

    cmd.end();

    impl_->device->resetFences(*impl_->capture_fence);
    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    impl_->graphics_queue.submit(si, *impl_->capture_fence);
    const auto r = impl_->device->waitForFences(
        *impl_->capture_fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
    (void)r;

    // Map and copy row-by-row (account for rowPitch != width*4).
    vk::ImageSubresource sub{};
    sub.aspectMask = vk::ImageAspectFlagBits::eColor;
    sub.mipLevel = 0;
    sub.arrayLayer = 0;
    const auto layout =
        impl_->device->getImageSubresourceLayout(*impl_->capture_staging.image, sub);
    void* mapped = impl_->device->mapMemory(*impl_->capture_staging.memory, 0, VK_WHOLE_SIZE);
    auto* src = static_cast<const std::uint8_t*>(mapped) + layout.offset;
    std::uint8_t* dst = out.pixels_rgba.data();
    for (std::uint32_t y = 0; y < out.height; ++y) {
        std::memcpy(dst + std::size_t(y) * out.width * 4,
                    src + std::size_t(y) * layout.rowPitch,
                    std::size_t(out.width) * 4);
    }
    impl_->device->unmapMemory(*impl_->capture_staging.memory);
    impl_->capture_staging.layout = vk::ImageLayout::eGeneral;
#else
    (void)out;
#endif
    return out;
}

// ===========================================================================
// Accessors
// ===========================================================================

void VulkanBackend::set_overlay(OverlayCallback cb) {
    impl_->overlay = std::move(cb);
}

#if SINGULARITY_VULKAN_HPP_AVAILABLE
void* VulkanBackend::vk_instance_handle() const {
    return impl_->instance ? reinterpret_cast<void*>(static_cast<VkInstance>(*impl_->instance))
                           : nullptr;
}
void* VulkanBackend::vk_physical_device_handle() const {
    return reinterpret_cast<void*>(static_cast<VkPhysicalDevice>(impl_->physical_device));
}
void* VulkanBackend::vk_device_handle() const {
    return impl_->device ? reinterpret_cast<void*>(static_cast<VkDevice>(*impl_->device)) : nullptr;
}
void* VulkanBackend::vk_graphics_queue_handle() const {
    return reinterpret_cast<void*>(static_cast<VkQueue>(impl_->graphics_queue));
}
void* VulkanBackend::vk_render_pass_handle() const {
    return impl_->blit_render_pass
               ? reinterpret_cast<void*>(static_cast<VkRenderPass>(*impl_->blit_render_pass))
               : nullptr;
}
std::uint32_t VulkanBackend::vk_graphics_queue_family() const {
    return impl_->graphics_queue_family;
}
std::uint32_t VulkanBackend::vk_swapchain_image_count() const {
    return static_cast<std::uint32_t>(impl_->swapchain_images.size());
}
#else
void* VulkanBackend::vk_instance_handle() const {
    return nullptr;
}
void* VulkanBackend::vk_physical_device_handle() const {
    return nullptr;
}
void* VulkanBackend::vk_device_handle() const {
    return nullptr;
}
void* VulkanBackend::vk_graphics_queue_handle() const {
    return nullptr;
}
void* VulkanBackend::vk_render_pass_handle() const {
    return nullptr;
}
std::uint32_t VulkanBackend::vk_graphics_queue_family() const {
    return 0;
}
std::uint32_t VulkanBackend::vk_swapchain_image_count() const {
    return 0;
}
#endif

double VulkanBackend::last_gpu_seconds() const {
    return impl_->last_gpu_seconds.load(std::memory_order_relaxed);
}

std::unique_ptr<RenderBackend> create_vulkan_backend() {
    return std::make_unique<VulkanBackend>();
}

}  // namespace singularity::vulkan
