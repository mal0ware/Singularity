// render/cuda/cuda_backend.cu
//
// CUDA backend — drives `singularity_geodesic_kernel` (defined in
// kernels/geodesic_kernel.cu) over a device-side RGBA8 buffer.
// `cuda_cli/main.cpp` is the only consumer; the live desktop app does
// not link this target.
//
// Frame pipeline:
//   1. pack_uniforms_cuda() builds a Uniforms struct from Scene + camera —
//      same field layout the Metal/Vulkan backends use, so the kernel reads
//      identical inputs across all three GPU APIs.
//   2. The kernel runs one thread per output pixel (16×16 blocks), writes
//      tone-mapped sRGB-encoded RGBA8 directly to d_image_.
//   3. capture_frame() does a D→H copy of d_image_ for PNG export.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "../../shared_shader/uniforms.h"
#include "cuda_backend.hpp"

namespace singularity {

// Defined in kernels/geodesic_kernel.cu.
__global__ void singularity_geodesic_kernel(std::uint8_t* out, Uniforms u);

namespace {

// Same constants the Metal + Vulkan backends use so the per-frame physics
// budget matches across backends. Mirrored — not shared via a header — so
// tweaking one backend's defaults can't accidentally retune the others.
constexpr std::uint32_t kDefaultMaxSteps = 2500;
constexpr float kDefaultHStep = 0.4f;

void check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(
            stderr, "singularity CUDA backend: %s failed: %s\n", where, cudaGetErrorString(err));
    }
}

Uniforms pack_uniforms_cuda(
    const Scene& scene, const CameraState& cam, std::uint32_t w, std::uint32_t h, float time_sec) {
    Uniforms out{};
    out.cam_pos = {cam.position[0], cam.position[1], cam.position[2], 0.0f};
    out.cam_right = {cam.basis[0], cam.basis[1], cam.basis[2], 0.0f};
    out.cam_up = {cam.basis[3], cam.basis[4], cam.basis[5], 0.0f};
    // Same forward-vector convention as the Vulkan/Metal backends —
    // negate the third basis row so the camera looks down -Z by default.
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
    // CUDA path uses a virtual frame-time accumulator (incremented by
    // 1/30 s after each render_frame in the backend) rather than a wall
    // clock. The first render lands at 0 — keeps single-shot goldens
    // deterministic. Subsequent renders pick up 1/30, 2/30, … so a
    // multi-frame `cuda_cli --frames N` PNG sequence encoded at 30 fps
    // shows the disc rotating at exactly its real Keplerian rate.
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
    out.frame_index = 0;
    out.supersample = scene.render_supersample == 0 ? 1u : scene.render_supersample;
    out.pad_b = 0;

    out.exposure = scene.exposure;
    out.bloom_threshold = scene.bloom_threshold;
    out.bloom_strength = scene.bloom_strength;
    out.disc_turbulence = scene.disc_turbulence;
    return out;
}

}  // namespace

CudaBackend::CudaBackend() = default;

CudaBackend::~CudaBackend() {
    shutdown();
}

bool CudaBackend::initialize(WindowHandle /*window*/, RenderConfig config) {
    width_ = config.width;
    height_ = config.height;
    frame_time_sec_ = 0.0f;
    const std::size_t bytes = static_cast<std::size_t>(width_) * height_ * 4u;
    check(cudaMalloc(&d_image_, bytes), "cudaMalloc");
    return d_image_ != nullptr;
}

void CudaBackend::shutdown() {
    if (d_image_ != nullptr) {
        check(cudaFree(d_image_), "cudaFree");
        d_image_ = nullptr;
    }
    width_ = 0;
    height_ = 0;
}

void CudaBackend::resize(std::uint32_t width, std::uint32_t height) {
    if (width == width_ && height == height_) {
        return;
    }
    shutdown();
    RenderConfig cfg{};
    cfg.width = width;
    cfg.height = height;
    initialize(WindowHandle{}, cfg);
}

void CudaBackend::render_frame(const Scene& scene, const CameraState& camera) {
    if (d_image_ == nullptr) {
        return;
    }
    const Uniforms u = pack_uniforms_cuda(scene, camera, width_, height_, frame_time_sec_);
    const dim3 block(16, 16);
    const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    singularity_geodesic_kernel<<<grid, block>>>(static_cast<std::uint8_t*>(d_image_), u);
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    // Advance the virtual clock by one 30-fps frame after the kernel
    // completes — see `frame_time_sec_` in the header for the rationale.
    frame_time_sec_ += 1.0f / 30.0f;
}

ImageData CudaBackend::capture_frame() {
    ImageData out;
    out.width = width_;
    out.height = height_;
    if (d_image_ == nullptr) {
        return out;
    }
    const std::size_t bytes = static_cast<std::size_t>(width_) * height_ * 4u;
    out.pixels_rgba.resize(bytes);
    check(cudaMemcpy(out.pixels_rgba.data(), d_image_, bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy D->H");
    return out;
}

}  // namespace singularity
