// render/metal/metal_backend.mm
//
// Objective-C++ implementation of the Metal RenderBackend. See
// docs/ARCHITECTURE.md §5 for the design.
//
// Responsibilities, roughly in the order they appear in the frame pipeline:
//   1. Own the MTLDevice, MTLCommandQueue, CAMetalLayer, drawable bookkeeping.
//   2. Own the compute pipeline (geodesic kernel) and render pipeline (blit).
//   3. Triple-buffer a uniforms buffer so CPU frame N+2 never contends with
//      GPU frame N.
//   4. Hold an HDR intermediate texture (rgba16Float) that the compute kernel
//      writes and the blit fragment reads.
//   5. Each frame:  encode compute -> encode render (blit + ImGui-later) ->
//      present -> commit.
//
// Kept intentionally terse. The Metal API is verbose enough without extra
// helper layers — the repeated dispatch/encode idioms are clearer inline.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "metal_backend.hpp"
#include "uniforms.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dispatch/dispatch.h>
#include <stdexcept>
#include <string>

namespace {

// Maximum number of frames in flight. Three is the Metal sweet spot — GPU
// pipeline depth is ~2 frames, so 3 buffers guarantees the CPU is never
// waiting on a GPU-owned buffer unless things have gone badly wrong.
constexpr std::size_t kMaxFramesInFlight = 3;

// Default ray-tracer integration budget. Chosen so the kernel matches the
// CPU reference's 5000-step limit but with a larger affine step for GPU
// throughput. Empirically: 2500 steps at h=0.4 produces images
// indistinguishable from 5000 steps at h=0.2 for camera distances ≥ 20M.
constexpr std::uint32_t kDefaultMaxSteps = 2500;
constexpr float kDefaultHStep = 0.4f;

// File name of the packaged Metal library. Baked into the bundle's
// Resources/, or sitting next to the executable for direct-run builds.
constexpr const char *kMetallibFilename = "default.metallib";

// Locate default.metallib by trying, in order:
//   1. NSBundle mainBundle's Resources/
//   2. Directory of the current executable (for `cmake --build && ./singularity`
//      outside a bundle)
// Returns a file URL the MTLDevice can load directly. Throws if not found.
NSURL *locate_metallib() {
    // Try the running app bundle first — this is the production path.
    NSBundle *bundle = [NSBundle mainBundle];
    NSURL *url = [bundle URLForResource:@"default" withExtension:@"metallib"];
    if (url)
        return url;

    // Fall back to <exe_dir>/default.metallib. `mainBundle.executablePath` is
    // non-nil even for non-bundle binaries (points at the raw Mach-O).
    NSString *exe = [bundle executablePath];
    if (exe) {
        NSString *dir = [exe stringByDeletingLastPathComponent];
        NSString *candidate =
            [dir stringByAppendingPathComponent:[NSString stringWithUTF8String:kMetallibFilename]];
        if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
            return [NSURL fileURLWithPath:candidate];
        }
    }
    return nil;
}

// Fill a Uniforms struct from the cross-platform Scene + CameraState. The
// CameraState exposes a row-major 3x3 basis; we decompose it into forward
// (row 2, negated — look-down-Z convention), right (row 0), up (row 1).
void pack_uniforms(Uniforms &out, const singularity::Scene &scene,
                   const singularity::CameraState &cam, std::uint32_t w, std::uint32_t h,
                   std::uint32_t frame_index, float time_sec) {
    out.cam_pos = {cam.position[0], cam.position[1], cam.position[2], 0.0f};
    out.cam_right = {cam.basis[0], cam.basis[1], cam.basis[2], 0.0f};
    out.cam_up = {cam.basis[3], cam.basis[4], cam.basis[5], 0.0f};
    // Convention: the third row is `-forward` so a unit z+ basis maps to
    // looking toward -z. Negate here so `cam_fwd` is the direction of travel.
    out.cam_fwd = {-cam.basis[6], -cam.basis[7], -cam.basis[8], 0.0f};

    out.tan_half_fov = std::tan(0.5f * cam.fov_y_radians);
    out.aspect = float(w) / float(h);

    out.mass_M = scene.mass_solar;
    out.spin_a = scene.spin_a_over_M * scene.mass_solar;
    out.rs = 2.0f * scene.mass_solar;

    out.disc_r_inner = scene.disc_inner_M * scene.mass_solar;
    out.disc_r_outer = scene.disc_outer_M * scene.mass_solar;
    // Peak disc temperature — user-configurable. The 4500 K Scene default
    // sits in the warm orange-yellow range (Kip Thorne's Interstellar look);
    // higher values shift toward white-blue and stop reading as "accretion
    // disc" — they look like welder's arc.
    out.disc_peak_T = scene.disc_peak_T_K;

    out.h_step = scene.h_step > 0.0f ? scene.h_step : kDefaultHStep;
    out.escape_r = 200.0f * scene.mass_solar;
    out.horizon_cut = 1.02f * out.rs;

    out.width = w;
    out.height = h;
    out.metric_type = (scene.metric == singularity::Scene::MetricType::Kerr)
                          ? SING_METRIC_KERR
                          : SING_METRIC_SCHWARZSCHILD;

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
    out.time_sec = time_sec;

    out.exposure = scene.exposure;
    out.bloom_threshold = scene.bloom_threshold;
    out.bloom_strength = scene.bloom_strength;
    out.disc_turbulence = scene.disc_turbulence;
}

} // namespace

namespace singularity::metal {

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    CAMetalLayer *layer = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> geodesic_pso = nil;
    id<MTLComputePipelineState> bloom_extract_pso = nil;
    id<MTLComputePipelineState> bloom_blur_h_pso = nil;
    id<MTLComputePipelineState> bloom_blur_v_pso = nil;
    id<MTLRenderPipelineState> blit_pso = nil;
    id<MTLTexture> hdr_texture = nil;
    id<MTLTexture> bloom_a = nil; // ping
    id<MTLTexture> bloom_b = nil; // pong
    std::array<id<MTLBuffer>, kMaxFramesInFlight> uniforms{};
    dispatch_semaphore_t in_flight_sem = nullptr;

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t frame = 0;
    // Wall-clock zero set at backend initialize. Fed into the kernel as
    // `time_sec` so the disc animation phase is consistent across frames.
    // Headless --capture renders fire `render_frame` immediately after
    // initialize so elapsed ≈ 0 — keeps headless captures deterministic.
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool vsync = true;

    // GPU-measured frame time (seconds). Written from the command-buffer
    // completion handler, read atomically by the UI.
    std::atomic<double> last_gpu_seconds{0.0};

    OverlayCallback overlay;

    ~Impl() {
        // Under ARC, dispatch objects are ref-counted by the compiler — no
        // dispatch_release needed. Null the handle so shutdown() is a no-op.
        in_flight_sem = nullptr;
    }

    // Helper: build a compute PSO for a named kernel function. Consolidates
    // the error-message assembly since four kernels use identical setup.
    id<MTLComputePipelineState> make_compute_pso(NSString *fn_name, std::string *err_out) {
        id<MTLFunction> fn = [library newFunctionWithName:fn_name];
        if (!fn) {
            if (err_out) {
                *err_out = "compute function missing: ";
                *err_out += [fn_name UTF8String];
            }
            return nil;
        }
        NSError *err = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                                error:&err];
        if (!pso && err_out) {
            *err_out = std::string("compute PSO failed for ") + [fn_name UTF8String] + ": " +
                       [[err localizedDescription] UTF8String];
        }
        return pso;
    }

    bool build_pipelines(std::string *err_out) {
        geodesic_pso = make_compute_pso(@"geodesic_kernel", err_out);
        if (!geodesic_pso)
            return false;
        bloom_extract_pso = make_compute_pso(@"bloom_extract", err_out);
        if (!bloom_extract_pso)
            return false;
        bloom_blur_h_pso = make_compute_pso(@"bloom_blur_h", err_out);
        if (!bloom_blur_h_pso)
            return false;
        bloom_blur_v_pso = make_compute_pso(@"bloom_blur_v", err_out);
        if (!bloom_blur_v_pso)
            return false;

        // Render pipeline — blit_vertex + blit_fragment.
        id<MTLFunction> vs_fn = [library newFunctionWithName:@"blit_vertex"];
        id<MTLFunction> fs_fn = [library newFunctionWithName:@"blit_fragment"];
        if (!vs_fn || !fs_fn) {
            if (err_out)
                *err_out = "blit vertex/fragment entry points missing";
            return false;
        }
        NSError *err = nil;
        MTLRenderPipelineDescriptor *rpd = [[MTLRenderPipelineDescriptor alloc] init];
        rpd.vertexFunction = vs_fn;
        rpd.fragmentFunction = fs_fn;
        rpd.colorAttachments[0].pixelFormat = layer.pixelFormat;
        blit_pso = [device newRenderPipelineStateWithDescriptor:rpd error:&err];
        if (!blit_pso) {
            if (err_out) {
                *err_out = "render PSO creation failed: ";
                if (err)
                    *err_out += [[err localizedDescription] UTF8String];
            }
            return false;
        }
        return true;
    }

    void create_hdr_texture() {
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        td.storageMode = MTLStorageModePrivate;
        hdr_texture = [device newTextureWithDescriptor:td];
    }

    void create_bloom_textures() {
        // 1/4 resolution — half in each axis. Cap the minimum so at tiny
        // window sizes (or capture resolutions) we don't get zero-sized
        // textures, which would error in Metal.
        const std::uint32_t bw = std::max<std::uint32_t>(1, width / 2);
        const std::uint32_t bh = std::max<std::uint32_t>(1, height / 2);
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:bw
                                                              height:bh
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        td.storageMode = MTLStorageModePrivate;
        bloom_a = [device newTextureWithDescriptor:td];
        bloom_b = [device newTextureWithDescriptor:td];
    }

    void create_uniforms_buffers() {
        for (auto &b : uniforms) {
            b = [device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
        }
    }
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() { shutdown(); }

bool MetalBackend::initialize(WindowHandle window, RenderConfig config) {
    @autoreleasepool {
        impl_->width = config.width;
        impl_->height = config.height;
        impl_->vsync = config.vsync_enabled;

        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            NSLog(@"[singularity] MTLCreateSystemDefaultDevice returned nil");
            return false;
        }
        impl_->queue = [impl_->device newCommandQueue];

        impl_->layer = (__bridge CAMetalLayer *)window.native_view;
        if (!impl_->layer) {
            NSLog(@"[singularity] WindowHandle.native_view is nil (expected CAMetalLayer*)");
            return false;
        }
        impl_->layer.device = impl_->device;
        impl_->layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
        impl_->layer.framebufferOnly = YES;
        impl_->layer.drawableSize = CGSizeMake(impl_->width, impl_->height);
        impl_->layer.displaySyncEnabled = config.vsync_enabled;

        NSURL *lib_url = locate_metallib();
        if (!lib_url) {
            NSLog(@"[singularity] default.metallib not found (bundle Resources/ or exe dir)");
            return false;
        }
        NSError *err = nil;
        impl_->library = [impl_->device newLibraryWithURL:lib_url error:&err];
        if (!impl_->library) {
            NSLog(@"[singularity] failed to load metallib: %@", err.localizedDescription);
            return false;
        }

        std::string pipe_err;
        if (!impl_->build_pipelines(&pipe_err)) {
            NSLog(@"[singularity] %s", pipe_err.c_str());
            return false;
        }

        impl_->create_hdr_texture();
        impl_->create_bloom_textures();
        impl_->create_uniforms_buffers();
        impl_->in_flight_sem = dispatch_semaphore_create(kMaxFramesInFlight);
        // Reset the animation clock so elapsed-since-init is measured
        // from the moment the backend is ready, not from Impl ctor.
        impl_->start_time = std::chrono::steady_clock::now();
        return true;
    }
}

void MetalBackend::shutdown() {
    if (!impl_)
        return;
    // Drain any in-flight frames before letting the Obj-C objects release.
    if (impl_->in_flight_sem) {
        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
            dispatch_semaphore_wait(impl_->in_flight_sem, DISPATCH_TIME_FOREVER);
        }
        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i) {
            dispatch_semaphore_signal(impl_->in_flight_sem);
        }
    }
    impl_->geodesic_pso = nil;
    impl_->bloom_extract_pso = nil;
    impl_->bloom_blur_h_pso = nil;
    impl_->bloom_blur_v_pso = nil;
    impl_->blit_pso = nil;
    impl_->hdr_texture = nil;
    impl_->bloom_a = nil;
    impl_->bloom_b = nil;
    for (auto &b : impl_->uniforms)
        b = nil;
    impl_->library = nil;
    impl_->queue = nil;
    impl_->layer = nil;
    impl_->device = nil;
}

void MetalBackend::resize(std::uint32_t width, std::uint32_t height) {
    if (width == impl_->width && height == impl_->height)
        return;
    if (width == 0 || height == 0)
        return;
    impl_->width = width;
    impl_->height = height;
    impl_->layer.drawableSize = CGSizeMake(width, height);
    impl_->create_hdr_texture();
    impl_->create_bloom_textures();
}

namespace {

// Shared helper: dispatch a full-texture compute kernel in 8×8 threadgroups.
void dispatch_full(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                   NSUInteger width, NSUInteger height) {
    [enc setComputePipelineState:pso];
    const MTLSize threads_per_grid = MTLSizeMake(width, height, 1);
    const MTLSize threads_per_tg = MTLSizeMake(8, 8, 1);
    [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_tg];
}

struct BlitParamsCPU {
    float exposure;
    float bloom_strength;
    float pad0;
    float pad1;
};

} // namespace

void MetalBackend::render_frame(const Scene &scene, const CameraState &camera) {
    @autoreleasepool {
        if (!impl_->layer)
            return;

        dispatch_semaphore_wait(impl_->in_flight_sem, DISPATCH_TIME_FOREVER);

        const std::size_t slot = impl_->frame % kMaxFramesInFlight;
        Uniforms *u_ptr = static_cast<Uniforms *>([impl_->uniforms[slot] contents]);
        const float elapsed =
            std::chrono::duration<float>(std::chrono::steady_clock::now() - impl_->start_time)
                .count();
        pack_uniforms(*u_ptr, scene, camera, impl_->width, impl_->height, impl_->frame, elapsed);

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        dispatch_semaphore_t sem = impl_->in_flight_sem;
        const auto t_enqueue = CFAbsoluteTimeGetCurrent();
        std::atomic<double> *gpu_out = &impl_->last_gpu_seconds;
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
            // GPU-measured elapsed time where available; fall back to wall-
            // clock enqueue→complete if the GPU timing isn't populated.
            const double elapsed = (done.GPUEndTime > 0.0)
                                       ? (done.GPUEndTime - done.GPUStartTime)
                                       : (CFAbsoluteTimeGetCurrent() - t_enqueue);
            gpu_out->store(elapsed, std::memory_order_relaxed);
            dispatch_semaphore_signal(sem);
        }];

        const NSUInteger bw = (NSUInteger)std::max<std::uint32_t>(1, impl_->width / 2);
        const NSUInteger bh = (NSUInteger)std::max<std::uint32_t>(1, impl_->height / 2);

        // --- Single compute encoder for geodesic + bloom ping-pong -----
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // Geodesic ray-trace → HDR texture.
            [enc setTexture:impl_->hdr_texture atIndex:0];
            [enc setBuffer:impl_->uniforms[slot] offset:0 atIndex:0];
            dispatch_full(enc, impl_->geodesic_pso, impl_->width, impl_->height);

            // Bloom extract: hdr → bloom_a (at half-res).
            const float threshold = scene.bloom_threshold;
            [enc setTexture:impl_->hdr_texture atIndex:0];
            [enc setTexture:impl_->bloom_a atIndex:1];
            [enc setBytes:&threshold length:sizeof(float) atIndex:0];
            dispatch_full(enc, impl_->bloom_extract_pso, bw, bh);

            // Separable Gaussian: bloom_a → bloom_b (h), bloom_b → bloom_a (v).
            [enc setTexture:impl_->bloom_a atIndex:0];
            [enc setTexture:impl_->bloom_b atIndex:1];
            dispatch_full(enc, impl_->bloom_blur_h_pso, bw, bh);

            [enc setTexture:impl_->bloom_b atIndex:0];
            [enc setTexture:impl_->bloom_a atIndex:1];
            dispatch_full(enc, impl_->bloom_blur_v_pso, bw, bh);

            [enc endEncoding];
        }

        // --- Render pass: tone-map HDR + bloom into the drawable -------
        id<CAMetalDrawable> drawable = [impl_->layer nextDrawable];
        if (drawable) {
            MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

            id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
            BlitParamsCPU bp{};
            bp.exposure = scene.exposure;
            bp.bloom_strength = scene.bloom_strength;
            [enc setRenderPipelineState:impl_->blit_pso];
            [enc setFragmentTexture:impl_->hdr_texture atIndex:0];
            [enc setFragmentTexture:impl_->bloom_a atIndex:1];
            [enc setFragmentBytes:&bp length:sizeof(bp) atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

            if (impl_->overlay) {
                impl_->overlay((__bridge void *)rpd, (__bridge void *)cmd, (__bridge void *)enc);
            }

            [enc endEncoding];
            [cmd presentDrawable:drawable];
        }

        [cmd commit];
        ++impl_->frame;
    }
}

ImageData MetalBackend::capture_frame() {
    @autoreleasepool {
        ImageData out;
        out.width = impl_->width;
        out.height = impl_->height;
        const std::size_t bytes = std::size_t(out.width) * out.height * 4;
        out.pixels_rgba.assign(bytes, 0);
        if (!impl_->hdr_texture)
            return out;

        // Blit HDR into a shared-storage RGBA8 texture, wait, then read.
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                                               width:out.width
                                                              height:out.height
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> dst = [impl_->device newTextureWithDescriptor:td];

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = dst;
        rpd.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
        // Capture uses the same blit params as the last live frame — it's
        // called right after render_frame in the headless path, so the
        // bloom texture is already populated.
        BlitParamsCPU bp{};
        bp.exposure = 1.0f;
        bp.bloom_strength = 0.35f;
        [enc setRenderPipelineState:impl_->blit_pso];
        [enc setFragmentTexture:impl_->hdr_texture atIndex:0];
        [enc setFragmentTexture:impl_->bloom_a atIndex:1];
        [enc setFragmentBytes:&bp length:sizeof(bp) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        [dst getBytes:out.pixels_rgba.data()
            bytesPerRow:out.width * 4
             fromRegion:MTLRegionMake2D(0, 0, out.width, out.height)
            mipmapLevel:0];
        return out;
    }
}

void MetalBackend::set_overlay(OverlayCallback cb) { impl_->overlay = std::move(cb); }
void *MetalBackend::metal_device_handle() const { return (__bridge void *)impl_->device; }
void *MetalBackend::metal_layer_handle() const { return (__bridge void *)impl_->layer; }
double MetalBackend::last_gpu_seconds() const {
    return impl_->last_gpu_seconds.load(std::memory_order_relaxed);
}

std::unique_ptr<RenderBackend> create_metal_backend() { return std::make_unique<MetalBackend>(); }

} // namespace singularity::metal
