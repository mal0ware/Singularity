// render/webgpu/webgpu_backend.cpp
//
// WebGPU RenderBackend — Dawn-standards-track body via --use-port=emdawnwebgpu.
//
// Async device init is the big shape change vs. Vulkan: the browser
// resolves navigator.gpu.requestAdapter + adapter.requestDevice over
// microtasks, so initialize() cannot block on the device arriving. It
// kicks off the chain, returns true, and caller-side render_frame()
// no-ops until ready_ flips true inside the chained callback. The
// web harness in web/main.cpp drives the same state machine.
//
// Resource graph mirrors render/vulkan/vulkan_backend.cpp 1:1 — see the
// scaffold header comment for the full mapping. This .cpp replaces the
// scaffold stubs; the per-frame compute (geodesic + 3 bloom) + render
// (blit) + present flow is now live.

#include "webgpu_backend.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#include <webgpu/webgpu_cpp.h>
#define SINGULARITY_WEBGPU_AVAILABLE 1
#else
#define SINGULARITY_WEBGPU_AVAILABLE 0
#endif

#include "uniforms.h"

namespace singularity::webgpu {

namespace {

void warn_once(const char* msg) {
    static std::atomic<bool> emitted{false};
    bool expected = false;
    if (emitted.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr, "[singularity/webgpu] %s\n", msg);
    }
}

#if SINGULARITY_WEBGPU_AVAILABLE

// A single emdawnwebgpu instance lives for the process lifetime. Dawn's
// emscripten port is happy with a nullptr descriptor.
wgpu::Instance& g_instance() {
    static wgpu::Instance inst = wgpuCreateInstance(nullptr);
    return inst;
}

// Slurp an entire file into a std::string. The WGSL sources are staged
// by CMake (--preload-file render/webgpu/shaders@/shaders) so they sit
// under /shaders/*.wgsl inside the Emscripten virtual FS.
std::string load_text_file(const char* path) {
    std::string out;
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "[singularity/webgpu] failed to open %s\n", path);
        return out;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) {
        out.resize(static_cast<std::size_t>(n));
        std::fread(out.data(), 1, static_cast<std::size_t>(n), f);
    }
    std::fclose(f);
    return out;
}

wgpu::ShaderModule
make_module(const wgpu::Device& device, const std::string& wgsl, const char* label) {
    wgpu::ShaderSourceWGSL src{};
    src.code = wgsl.c_str();

    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &src;
    desc.label = label;
    return device.CreateShaderModule(&desc);
}

// Pack host-side Scene + CameraState into the 160-byte Uniforms struct
// the WGSL kernel expects. Mirrors render/metal/metal_backend.mm
// pack_uniforms and the equivalent in render/vulkan. All three backends
// consume the same shared_shader/uniforms.h layout.
void pack_uniforms(Uniforms& out,
                   const Scene& scene,
                   const CameraState& cam,
                   std::uint32_t w,
                   std::uint32_t h,
                   std::uint32_t frame_index,
                   float time_sec) {
    auto v4 = [](float x, float y, float z) {
        sing_vec3 v;
        v.x = x;
        v.y = y;
        v.z = z;
        v.w = 0.0f;
        return v;
    };
    out.cam_pos = v4(cam.position[0], cam.position[1], cam.position[2]);
    out.cam_right = v4(cam.basis[0], cam.basis[1], cam.basis[2]);
    out.cam_up = v4(cam.basis[3], cam.basis[4], cam.basis[5]);
    // basis row 2 is -forward by convention, matching Metal/Vulkan.
    out.cam_fwd = v4(-cam.basis[6], -cam.basis[7], -cam.basis[8]);

    out.mass_M = scene.mass_solar;
    out.spin_a = scene.spin_a_over_M * scene.mass_solar;
    out.rs = 2.0f * scene.mass_solar;
    out.tan_half_fov = std::tan(0.5f * cam.fov_y_radians);

    out.disc_r_inner = scene.disc_inner_M;
    out.disc_r_outer = scene.disc_outer_M;
    out.disc_peak_T = scene.disc_peak_T_K;
    out.aspect = (h == 0) ? 1.0f : float(w) / float(h);

    out.h_step = scene.h_step > 0.0f ? scene.h_step : 0.12f;
    // Dynamic escape radius. Fixed 200M wastes budget when the camera sits
    // at 30M; residual deflection past 3× the camera radius shifts the
    // sampled sky by ~a pixel at 60° FOV. Two hard constraints, learned the
    // review's way: it must exceed the camera radius (the kernel's escape
    // check fires before anything else, so escape_r < cam_r blanks the whole
    // frame — the zoom clamp allows 400M), and it must clear the disc's
    // outer edge (the escape check also runs before crossing detection).
    // Web-only behaviour — desktop backends keep fixed 200M and their
    // goldens.
    const float cam_r =
        std::sqrt(cam.position[0] * cam.position[0] + cam.position[1] * cam.position[1]
                  + cam.position[2] * cam.position[2]);
    out.escape_r = std::max(std::max(3.0f * cam_r, 1.5f * scene.disc_outer_M), 60.0f);
    // Kerr sets its own tighter horizon cut in-kernel; this covers the
    // Schwarzschild path.
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
    if (scene.adaptive_step_on)
        flags |= SING_FLAG_ADAPTIVE_STEP;
    out.flags = flags;

    out.max_steps = scene.max_steps != 0 ? scene.max_steps : 800u;
    out.frame_index = frame_index;
    out.supersample =
        std::max<std::uint32_t>(1u, std::min<std::uint32_t>(scene.render_supersample, 4u));
    out.pad_b = 0u;

    out.exposure = scene.exposure;
    out.bloom_threshold = scene.bloom_threshold;
    out.bloom_strength = scene.bloom_strength;
    out.disc_turbulence = scene.disc_turbulence;
}

struct ExtractParams {
    float threshold;
    float pad0, pad1, pad2;
};

struct BlitParams {
    float exposure;
    float bloom_strength;
    float pad0, pad1;
};

#endif  // SINGULARITY_WEBGPU_AVAILABLE

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct WebGPUBackend::Impl {
#if SINGULARITY_WEBGPU_AVAILABLE
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::TextureFormat surface_format = wgpu::TextureFormat::BGRA8Unorm;

    // HDR + bloom storage textures (re-created on resize).
    wgpu::Texture hdr_tex;
    wgpu::TextureView hdr_sampled;
    wgpu::TextureView hdr_storage;
    wgpu::Texture bloom_a, bloom_b;
    wgpu::TextureView bloom_a_sampled, bloom_a_storage;
    wgpu::TextureView bloom_b_sampled, bloom_b_storage;

    wgpu::Sampler blit_sampler;

    wgpu::Buffer uniforms_buf;
    wgpu::Buffer extract_params_buf;
    wgpu::Buffer blit_params_buf;

    wgpu::BindGroupLayout geodesic_bgl;
    wgpu::BindGroupLayout bloom_bgl;
    wgpu::BindGroupLayout blit_bgl;

    wgpu::ComputePipeline geodesic_pipeline;
    wgpu::ComputePipeline bloom_extract_pipeline;
    wgpu::ComputePipeline bloom_blur_h_pipeline;
    wgpu::ComputePipeline bloom_blur_v_pipeline;
    wgpu::RenderPipeline blit_pipeline;

    wgpu::BindGroup geodesic_bg;
    wgpu::BindGroup bloom_extract_bg;
    wgpu::BindGroup bloom_blur_h_bg;
    wgpu::BindGroup bloom_blur_v_bg;
    wgpu::BindGroup blit_bg;

    bool surface_configured = false;
#endif

    std::atomic<bool> ready{false};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Internal render scale — geodesic/HDR/bloom run at scale × canvas;
    // the blit's linear sampler upscales to the full-size surface.
    float render_scale = 1.0f;
    std::uint32_t render_w() const {
        return std::max<std::uint32_t>(1u, std::uint32_t(float(width) * render_scale + 0.5f));
    }
    std::uint32_t render_h() const {
        return std::max<std::uint32_t>(1u, std::uint32_t(float(height) * render_scale + 0.5f));
    }
    std::uint32_t frame = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool vsync = true;
    std::atomic<double> last_gpu_seconds{0.0};

    OverlayCallback overlay;
};

WebGPUBackend::WebGPUBackend() : impl_(std::make_unique<Impl>()) {}
WebGPUBackend::~WebGPUBackend() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#if SINGULARITY_WEBGPU_AVAILABLE
// Forward decls for the helpers that run once the async device arrives.
static void build_pipelines_and_layouts(WebGPUBackend::Impl& im);
static void configure_surface_and_resources(WebGPUBackend::Impl& im);
#endif

bool WebGPUBackend::initialize(WindowHandle window, RenderConfig config) {
#if !SINGULARITY_WEBGPU_AVAILABLE
    (void)window;
    (void)config;
    warn_once("WebGPU unavailable (native build) — backend is a stub.");
    return false;
#else
    (void)window;
    impl_->width = config.width;
    impl_->height = config.height;
    impl_->vsync = config.vsync_enabled;

    // Kick off the async chain: request adapter → request device →
    // set up pipelines → configure surface. Everything below lives
    // inside the chained lambdas; render_frame() early-exits until
    // ready flips true.
    g_instance().RequestAdapter(
        nullptr,
        wgpu::CallbackMode::AllowSpontaneous,
        [this](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView msg) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::fprintf(stderr,
                             "[singularity/webgpu] RequestAdapter failed: %.*s\n",
                             int(msg.length),
                             msg.data);
                return;
            }
            impl_->adapter = a;

            wgpu::DeviceDescriptor desc{};
            desc.label = "singularity-device";
            desc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView m) {
                    std::fprintf(stderr,
                                 "[singularity/webgpu] UncapturedError (type=%d): %.*s\n",
                                 int(type),
                                 int(m.length),
                                 m.data);
                });

            impl_->adapter.RequestDevice(
                &desc,
                wgpu::CallbackMode::AllowSpontaneous,
                [this](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView msg) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        std::fprintf(stderr,
                                     "[singularity/webgpu] RequestDevice failed: %.*s\n",
                                     int(msg.length),
                                     msg.data);
                        return;
                    }
                    impl_->device = d;
                    impl_->queue = impl_->device.GetQueue();

                    build_pipelines_and_layouts(*impl_);
                    configure_surface_and_resources(*impl_);
                    // Reset animation clock so disc phase starts at 0 when
                    // the first real frame goes out, not at Impl ctor time.
                    impl_->start_time = std::chrono::steady_clock::now();
                    impl_->ready.store(true, std::memory_order_release);
                    std::fprintf(stderr,
                                 "[singularity/webgpu] device ready, %ux%u\n",
                                 impl_->width,
                                 impl_->height);
                });
        });
    return true;
#endif
}

void WebGPUBackend::shutdown() {
#if SINGULARITY_WEBGPU_AVAILABLE
    if (!impl_)
        return;
    // The wgpu:: RAII types release on destruction; resetting the Impl
    // handles tear-down in reverse construction order without explicit
    // calls. Flip ready first so any in-flight render call bails.
    impl_->ready.store(false, std::memory_order_release);
#endif
}

#if SINGULARITY_WEBGPU_AVAILABLE

static wgpu::TextureView create_view(const wgpu::Texture& tex,
                                     wgpu::TextureFormat fmt,
                                     wgpu::TextureAspect aspect = wgpu::TextureAspect::All) {
    wgpu::TextureViewDescriptor d{};
    d.format = fmt;
    d.dimension = wgpu::TextureViewDimension::e2D;
    d.baseMipLevel = 0;
    d.mipLevelCount = 1;
    d.baseArrayLayer = 0;
    d.arrayLayerCount = 1;
    d.aspect = aspect;
    return tex.CreateView(&d);
}

static void build_pipelines_and_layouts(WebGPUBackend::Impl& im) {
    // --- Load WGSL -----------------------------------------------------------
    const std::string geo_src = load_text_file("/shaders/geodesic_kernel.wgsl");
    const std::string bloom_src = load_text_file("/shaders/bloom.wgsl");
    const std::string blit_src = load_text_file("/shaders/blit.wgsl");
    if (geo_src.empty() || bloom_src.empty() || blit_src.empty()) {
        std::fprintf(stderr, "[singularity/webgpu] missing WGSL sources in /shaders/\n");
        return;
    }
    wgpu::ShaderModule geo_mod = make_module(im.device, geo_src, "geodesic_kernel");
    wgpu::ShaderModule bloom_mod = make_module(im.device, bloom_src, "bloom");
    wgpu::ShaderModule blit_mod = make_module(im.device, blit_src, "blit");

    // --- Bind-group layouts --------------------------------------------------
    // Geodesic (set 0): binding 0 uniform, binding 1 storage texture (rgba16f).
    {
        wgpu::BindGroupLayoutEntry entries[2]{};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[0].buffer.minBindingSize = sizeof(Uniforms);

        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
        entries[1].storageTexture.format = wgpu::TextureFormat::RGBA16Float;
        entries[1].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

        wgpu::BindGroupLayoutDescriptor d{};
        d.entryCount = 2;
        d.entries = entries;
        im.geodesic_bgl = im.device.CreateBindGroupLayout(&d);
    }

    // Bloom (set 0): sampled tex, storage tex, uniform (ExtractParams).
    {
        wgpu::BindGroupLayoutEntry entries[3]{};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Compute;
        entries[0].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
        entries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Compute;
        entries[1].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
        entries[1].storageTexture.format = wgpu::TextureFormat::RGBA16Float;
        entries[1].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Compute;
        entries[2].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[2].buffer.minBindingSize = sizeof(ExtractParams);

        wgpu::BindGroupLayoutDescriptor d{};
        d.entryCount = 3;
        d.entries = entries;
        im.bloom_bgl = im.device.CreateBindGroupLayout(&d);
    }

    // Blit (set 0): hdr sampled, bloom sampled, sampler, uniform (BlitParams).
    // Filterable float + filtering sampler — the blit is also the upscale
    // path when the internal render scale is below 1.
    {
        wgpu::BindGroupLayoutEntry entries[4]{};
        entries[0].binding = 0;
        entries[0].visibility = wgpu::ShaderStage::Fragment;
        entries[0].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;

        entries[1].binding = 1;
        entries[1].visibility = wgpu::ShaderStage::Fragment;
        entries[1].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;

        entries[2].binding = 2;
        entries[2].visibility = wgpu::ShaderStage::Fragment;
        entries[2].sampler.type = wgpu::SamplerBindingType::Filtering;

        entries[3].binding = 3;
        entries[3].visibility = wgpu::ShaderStage::Fragment;
        entries[3].buffer.type = wgpu::BufferBindingType::Uniform;
        entries[3].buffer.minBindingSize = sizeof(BlitParams);

        wgpu::BindGroupLayoutDescriptor d{};
        d.entryCount = 4;
        d.entries = entries;
        im.blit_bgl = im.device.CreateBindGroupLayout(&d);
    }

    // --- Pipeline layouts ----------------------------------------------------
    wgpu::PipelineLayout geodesic_pl, bloom_pl, blit_pl;
    {
        wgpu::BindGroupLayout bgls[1] = {im.geodesic_bgl};
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = 1;
        d.bindGroupLayouts = bgls;
        geodesic_pl = im.device.CreatePipelineLayout(&d);
    }
    {
        wgpu::BindGroupLayout bgls[1] = {im.bloom_bgl};
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = 1;
        d.bindGroupLayouts = bgls;
        bloom_pl = im.device.CreatePipelineLayout(&d);
    }
    {
        wgpu::BindGroupLayout bgls[1] = {im.blit_bgl};
        wgpu::PipelineLayoutDescriptor d{};
        d.bindGroupLayoutCount = 1;
        d.bindGroupLayouts = bgls;
        blit_pl = im.device.CreatePipelineLayout(&d);
    }

    // --- Compute pipelines ---------------------------------------------------
    auto make_compute =
        [&](wgpu::PipelineLayout pl, wgpu::ShaderModule mod, const char* entry, const char* label) {
            wgpu::ComputePipelineDescriptor d{};
            d.label = label;
            d.layout = pl;
            d.compute.module = mod;
            d.compute.entryPoint = entry;
            return im.device.CreateComputePipeline(&d);
        };
    im.geodesic_pipeline = make_compute(geodesic_pl, geo_mod, "main", "geodesic");
    im.bloom_extract_pipeline = make_compute(bloom_pl, bloom_mod, "extract", "bloom_extract");
    im.bloom_blur_h_pipeline = make_compute(bloom_pl, bloom_mod, "blur_h", "bloom_blur_h");
    im.bloom_blur_v_pipeline = make_compute(bloom_pl, bloom_mod, "blur_v", "bloom_blur_v");

    // --- Render pipeline (blit) ---------------------------------------------
    {
        wgpu::ColorTargetState color{};
        color.format = im.surface_format;

        wgpu::FragmentState fs{};
        fs.module = blit_mod;
        fs.entryPoint = "main_ps";
        fs.targetCount = 1;
        fs.targets = &color;

        wgpu::RenderPipelineDescriptor d{};
        d.label = "blit";
        d.layout = blit_pl;
        d.vertex.module = blit_mod;
        d.vertex.entryPoint = "main_vs";
        d.fragment = &fs;
        d.primitive.topology = wgpu::PrimitiveTopology::TriangleList;

        im.blit_pipeline = im.device.CreateRenderPipeline(&d);
    }

    // --- Static resources: uniform buffers + sampler ------------------------
    {
        wgpu::BufferDescriptor d{};
        d.size = sizeof(Uniforms);
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        d.label = "uniforms";
        im.uniforms_buf = im.device.CreateBuffer(&d);
    }
    {
        wgpu::BufferDescriptor d{};
        d.size = sizeof(ExtractParams);
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        d.label = "extract_params";
        im.extract_params_buf = im.device.CreateBuffer(&d);
    }
    {
        wgpu::BufferDescriptor d{};
        d.size = sizeof(BlitParams);
        d.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        d.label = "blit_params";
        im.blit_params_buf = im.device.CreateBuffer(&d);
    }
    {
        // Linear so the blit doubles as the upscale filter when the
        // internal render scale drops below 1 (rgba16float is filterable
        // in core WebGPU).
        wgpu::SamplerDescriptor d{};
        d.minFilter = wgpu::FilterMode::Linear;
        d.magFilter = wgpu::FilterMode::Linear;
        d.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
        d.addressModeU = wgpu::AddressMode::ClampToEdge;
        d.addressModeV = wgpu::AddressMode::ClampToEdge;
        im.blit_sampler = im.device.CreateSampler(&d);
    }
}

// Create / re-create the HDR + bloom textures + bind groups at the
// current *internal* resolution (render_scale × canvas). Called from init,
// resize, and every internal-scale change.
static void build_size_dependent_resources(WebGPUBackend::Impl& im) {
    const std::uint32_t w = im.render_w();
    const std::uint32_t h = im.render_h();
    const std::uint32_t bw = std::max<std::uint32_t>(w / 2, 1);
    const std::uint32_t bh = std::max<std::uint32_t>(h / 2, 1);

    auto make_tex = [&](std::uint32_t tw, std::uint32_t th, const char* label) {
        wgpu::TextureDescriptor d{};
        d.label = label;
        d.size = {tw, th, 1};
        d.format = wgpu::TextureFormat::RGBA16Float;
        d.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::StorageBinding
                  | wgpu::TextureUsage::CopySrc;
        d.mipLevelCount = 1;
        d.sampleCount = 1;
        d.dimension = wgpu::TextureDimension::e2D;
        return im.device.CreateTexture(&d);
    };

    im.hdr_tex = make_tex(w, h, "hdr");
    im.hdr_sampled = create_view(im.hdr_tex, wgpu::TextureFormat::RGBA16Float);
    im.hdr_storage = create_view(im.hdr_tex, wgpu::TextureFormat::RGBA16Float);

    im.bloom_a = make_tex(bw, bh, "bloom_a");
    im.bloom_a_sampled = create_view(im.bloom_a, wgpu::TextureFormat::RGBA16Float);
    im.bloom_a_storage = create_view(im.bloom_a, wgpu::TextureFormat::RGBA16Float);

    im.bloom_b = make_tex(bw, bh, "bloom_b");
    im.bloom_b_sampled = create_view(im.bloom_b, wgpu::TextureFormat::RGBA16Float);
    im.bloom_b_storage = create_view(im.bloom_b, wgpu::TextureFormat::RGBA16Float);

    // Rebuild bind groups that reference the re-created views.
    {
        wgpu::BindGroupEntry entries[2]{};
        entries[0].binding = 0;
        entries[0].buffer = im.uniforms_buf;
        entries[0].size = sizeof(Uniforms);
        entries[1].binding = 1;
        entries[1].textureView = im.hdr_storage;
        wgpu::BindGroupDescriptor d{};
        d.layout = im.geodesic_bgl;
        d.entryCount = 2;
        d.entries = entries;
        im.geodesic_bg = im.device.CreateBindGroup(&d);
    }
    auto make_bloom_bg = [&](wgpu::TextureView src, wgpu::TextureView dst) {
        wgpu::BindGroupEntry entries[3]{};
        entries[0].binding = 0;
        entries[0].textureView = src;
        entries[1].binding = 1;
        entries[1].textureView = dst;
        entries[2].binding = 2;
        entries[2].buffer = im.extract_params_buf;
        entries[2].size = sizeof(ExtractParams);
        wgpu::BindGroupDescriptor d{};
        d.layout = im.bloom_bgl;
        d.entryCount = 3;
        d.entries = entries;
        return im.device.CreateBindGroup(&d);
    };
    im.bloom_extract_bg = make_bloom_bg(im.hdr_sampled, im.bloom_a_storage);
    im.bloom_blur_h_bg = make_bloom_bg(im.bloom_a_sampled, im.bloom_b_storage);
    im.bloom_blur_v_bg = make_bloom_bg(im.bloom_b_sampled, im.bloom_a_storage);

    {
        wgpu::BindGroupEntry entries[4]{};
        entries[0].binding = 0;
        entries[0].textureView = im.hdr_sampled;
        entries[1].binding = 1;
        entries[1].textureView = im.bloom_a_sampled;
        entries[2].binding = 2;
        entries[2].sampler = im.blit_sampler;
        entries[3].binding = 3;
        entries[3].buffer = im.blit_params_buf;
        entries[3].size = sizeof(BlitParams);
        wgpu::BindGroupDescriptor d{};
        d.layout = im.blit_bgl;
        d.entryCount = 4;
        d.entries = entries;
        im.blit_bg = im.device.CreateBindGroup(&d);
    }
}

static void configure_surface_and_resources(WebGPUBackend::Impl& im) {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc{};
    canvas_desc.selector = "#canvas";

    wgpu::SurfaceDescriptor d{};
    d.nextInChain = &canvas_desc;
    im.surface = g_instance().CreateSurface(&d);

    wgpu::SurfaceCapabilities caps{};
    im.surface.GetCapabilities(im.adapter, &caps);
    // Prefer *-UnormSrgb so the hardware does the sRGB encode at present.
    // The blit fragment writes linear HDR colour (post-ACES but pre-gamma)
    // and relies on the framebuffer's sRGB view for gamma-correct output,
    // matching the Metal CAMetalLayer path. Fall back to the platform's
    // advertised first format if no sRGB variant is listed.
    im.surface_format = (caps.formatCount > 0) ? caps.formats[0] : wgpu::TextureFormat::BGRA8Unorm;
    for (std::size_t i = 0; i < caps.formatCount; ++i) {
        const auto f = caps.formats[i];
        if (f == wgpu::TextureFormat::BGRA8UnormSrgb || f == wgpu::TextureFormat::RGBA8UnormSrgb) {
            im.surface_format = f;
            break;
        }
    }

    wgpu::SurfaceConfiguration cfg{};
    cfg.device = im.device;
    cfg.format = im.surface_format;
    cfg.usage = wgpu::TextureUsage::RenderAttachment;
    cfg.width = im.width;
    cfg.height = im.height;
    cfg.alphaMode = wgpu::CompositeAlphaMode::Auto;
    cfg.presentMode = im.vsync ? wgpu::PresentMode::Fifo : wgpu::PresentMode::Immediate;
    im.surface.Configure(&cfg);
    im.surface_configured = true;

    build_size_dependent_resources(im);
}
#endif  // SINGULARITY_WEBGPU_AVAILABLE

void WebGPUBackend::resize(std::uint32_t width, std::uint32_t height) {
    if (width == impl_->width && height == impl_->height)
        return;
    if (width == 0 || height == 0)
        return;
    impl_->width = width;
    impl_->height = height;
#if SINGULARITY_WEBGPU_AVAILABLE
    if (!impl_->ready.load(std::memory_order_acquire))
        return;

    wgpu::SurfaceConfiguration cfg{};
    cfg.device = impl_->device;
    cfg.format = impl_->surface_format;
    cfg.usage = wgpu::TextureUsage::RenderAttachment;
    cfg.width = impl_->width;
    cfg.height = impl_->height;
    cfg.alphaMode = wgpu::CompositeAlphaMode::Auto;
    cfg.presentMode = impl_->vsync ? wgpu::PresentMode::Fifo : wgpu::PresentMode::Immediate;
    impl_->surface.Configure(&cfg);
    build_size_dependent_resources(*impl_);
#endif
}

void WebGPUBackend::set_internal_scale(float scale) {
    scale = std::max(0.25f, std::min(1.0f, scale));
    if (std::fabs(scale - impl_->render_scale) < 1e-3f)
        return;
    impl_->render_scale = scale;
#if SINGULARITY_WEBGPU_AVAILABLE
    if (!impl_->ready.load(std::memory_order_acquire))
        return;
    // The surface (canvas) is untouched — only the HDR/bloom chain and its
    // bind groups are re-created at the new internal resolution.
    build_size_dependent_resources(*impl_);
#endif
}

float WebGPUBackend::internal_scale() const {
    return impl_->render_scale;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------

void WebGPUBackend::render_frame(const Scene& scene, const CameraState& camera) {
    (void)scene;
    (void)camera;
#if SINGULARITY_WEBGPU_AVAILABLE
    if (!impl_->ready.load(std::memory_order_acquire))
        return;

    // --- Uniform uploads ----------------------------------------------------
    Uniforms uni{};
    const auto now = std::chrono::steady_clock::now();
    const float elapsed = std::chrono::duration<float>(now - impl_->start_time).count();
    pack_uniforms(uni, scene, camera, impl_->render_w(), impl_->render_h(), impl_->frame, elapsed);
    impl_->queue.WriteBuffer(impl_->uniforms_buf, 0, &uni, sizeof(uni));

    ExtractParams ep{};
    ep.threshold = scene.bloom_threshold;
    ep.pad0 = ep.pad1 = ep.pad2 = 0.0f;
    impl_->queue.WriteBuffer(impl_->extract_params_buf, 0, &ep, sizeof(ep));

    BlitParams bp{};
    bp.exposure = scene.exposure;
    bp.bloom_strength = scene.bloom_strength;
    bp.pad0 = bp.pad1 = 0.0f;
    impl_->queue.WriteBuffer(impl_->blit_params_buf, 0, &bp, sizeof(bp));

    // --- Acquire swapchain image -------------------------------------------
    wgpu::SurfaceTexture st{};
    impl_->surface.GetCurrentTexture(&st);
    if (st.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal
        && st.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        // Canvas was resized between submissions or the surface was lost.
        // Bail and let the next tick re-acquire.
        return;
    }
    wgpu::TextureView backbuffer = st.texture.CreateView();

    // --- Encode the frame --------------------------------------------------
    wgpu::CommandEncoder enc = impl_->device.CreateCommandEncoder();

    // Compute: geodesic → bloom extract → blur_h → blur_v.
    {
        wgpu::ComputePassEncoder cp = enc.BeginComputePass();

        cp.SetPipeline(impl_->geodesic_pipeline);
        cp.SetBindGroup(0, impl_->geodesic_bg);
        const std::uint32_t rw = impl_->render_w();
        const std::uint32_t rh = impl_->render_h();
        const std::uint32_t gx = (rw + 7) / 8;
        const std::uint32_t gy = (rh + 7) / 8;
        cp.DispatchWorkgroups(gx, gy, 1);

        const std::uint32_t bx = (std::max<std::uint32_t>(rw / 2, 1) + 7) / 8;
        const std::uint32_t by = (std::max<std::uint32_t>(rh / 2, 1) + 7) / 8;

        cp.SetPipeline(impl_->bloom_extract_pipeline);
        cp.SetBindGroup(0, impl_->bloom_extract_bg);
        cp.DispatchWorkgroups(bx, by, 1);

        cp.SetPipeline(impl_->bloom_blur_h_pipeline);
        cp.SetBindGroup(0, impl_->bloom_blur_h_bg);
        cp.DispatchWorkgroups(bx, by, 1);

        cp.SetPipeline(impl_->bloom_blur_v_pipeline);
        cp.SetBindGroup(0, impl_->bloom_blur_v_bg);
        cp.DispatchWorkgroups(bx, by, 1);

        cp.End();
    }

    // Render: fullscreen blit composites HDR + bloom + ACES.
    {
        wgpu::RenderPassColorAttachment color{};
        color.view = backbuffer;
        color.loadOp = wgpu::LoadOp::Clear;
        color.storeOp = wgpu::StoreOp::Store;
        color.clearValue = {0, 0, 0, 1};

        wgpu::RenderPassDescriptor rpd{};
        rpd.colorAttachmentCount = 1;
        rpd.colorAttachments = &color;

        wgpu::RenderPassEncoder rp = enc.BeginRenderPass(&rpd);
        rp.SetPipeline(impl_->blit_pipeline);
        rp.SetBindGroup(0, impl_->blit_bg);
        rp.Draw(3);

        if (impl_->overlay) {
            impl_->overlay(&backbuffer, &enc, &rp);
        }
        rp.End();
    }

    wgpu::CommandBuffer cb = enc.Finish();
    impl_->queue.Submit(1, &cb);
    // Emscripten's Dawn port presents implicitly at main-loop iteration
    // boundary (next requestAnimationFrame), so no explicit Present call.
    ++impl_->frame;
#endif
}

ImageData WebGPUBackend::capture_frame() {
    ImageData out;
    out.width = impl_->width;
    out.height = impl_->height;
    out.pixels_rgba.assign(std::size_t(out.width) * out.height * 4, 0);
#if SINGULARITY_WEBGPU_AVAILABLE
    // WebGPU readback is async (MapAsync). A synchronous capture_frame
    // contract can't be honoured from the main thread in the browser
    // without Asyncify or a worker + SharedArrayBuffer. Deferred to a
    // follow-up — the desktop backends cover the golden-image harness.
    warn_once("capture_frame(): async readback is a follow-up; returning zeroed image.");
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

void WebGPUBackend::set_overlay(OverlayCallback cb) {
    impl_->overlay = std::move(cb);
}

#if SINGULARITY_WEBGPU_AVAILABLE
void* WebGPUBackend::wgpu_device_handle() const {
    return static_cast<void*>(impl_->device.Get());
}
void* WebGPUBackend::wgpu_queue_handle() const {
    return static_cast<void*>(impl_->queue.Get());
}
void* WebGPUBackend::wgpu_surface_handle() const {
    return static_cast<void*>(impl_->surface.Get());
}
std::uint32_t WebGPUBackend::wgpu_surface_format() const {
    return static_cast<std::uint32_t>(impl_->surface_format);
}
#else
void* WebGPUBackend::wgpu_device_handle() const {
    return nullptr;
}
void* WebGPUBackend::wgpu_queue_handle() const {
    return nullptr;
}
void* WebGPUBackend::wgpu_surface_handle() const {
    return nullptr;
}
std::uint32_t WebGPUBackend::wgpu_surface_format() const {
    return 0u;
}
#endif

double WebGPUBackend::last_gpu_seconds() const {
    return impl_->last_gpu_seconds.load(std::memory_order_relaxed);
}

std::unique_ptr<RenderBackend> create_webgpu_backend() {
    return std::make_unique<WebGPUBackend>();
}

}  // namespace singularity::webgpu
