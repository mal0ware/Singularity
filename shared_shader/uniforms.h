// render/metal/shaders/uniforms.h
//
// Host ↔ GPU uniform block. Included from both the MSL kernels and the
// Objective-C++ backend, so both sides always agree on layout.
//
// ABI note: MSL's `float3` has size *and* alignment 16 bytes — not 12.
// Using a 12-byte C++ struct with manual 4-byte padding after each vec3
// produces a layout the GPU reads wrong (every field after the first vec3
// lands at the wrong offset, the kernel gets garbage for the camera
// basis, and every pixel ends up black). Every vector field here is
// therefore backed by a 16-byte `float4`; the shader accesses `.xyz`.

#ifndef SINGULARITY_RENDER_METAL_UNIFORMS_H
#define SINGULARITY_RENDER_METAL_UNIFORMS_H

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
using namespace metal;
typedef float4 sing_vec3;  // backed by float4; use .xyz
typedef uint sing_uint;
#elif defined(__HLSL_VERSION) || defined(_HLSL) || defined(__SPIRV__)
// HLSL / DXC → SPIR-V for the Vulkan backend. float4 and uint are
// native primitives here; the struct layout below then lines up with
// the MSL + host layouts because all three flavours use 16-byte-aligned
// 3-vectors backed by float4 storage.
typedef float4 sing_vec3;
typedef uint sing_uint;
#else
#include <cstdint>
// 16-byte / 16-aligned to match MSL's `float3`/`float4` layout.
struct alignas(16) sing_vec3 {
    float x;
    float y;
    float z;
    float w;
};
typedef std::uint32_t sing_uint;
#endif

// Bit flags packed into Uniforms::flags.
#define SING_FLAG_DOPPLER_ON (1u << 0)
#define SING_FLAG_REDSHIFT_ON (1u << 1)
#define SING_FLAG_DISC_ON (1u << 2)
#define SING_FLAG_STARFIELD_ON (1u << 3)
// Radius-adaptive integrator step (see adaptive_h in geodesic_math.h).
// Off by default on the desktop backends so their golden images stay
// byte-stable; the web backend turns it on for interactive framerates.
#define SING_FLAG_ADAPTIVE_STEP (1u << 4)

// Metric selector packed into Uniforms::metric_type.
#define SING_METRIC_SCHWARZSCHILD 0u
#define SING_METRIC_KERR 1u

struct Uniforms {
    // Camera basis (Cartesian in the local inertial frame of the BH center).
    sing_vec3 cam_pos;    //  0 — position (geometrized units)
    sing_vec3 cam_fwd;    // 16 — unit forward
    sing_vec3 cam_right;  // 32 — unit right
    sing_vec3 cam_up;     // 48 — unit up (w holds tan_half_fov)
    // w components of cam_up through cam_pos are free to reuse; see below.

    // Scene.
    float mass_M;        // gravitational mass in geometrized units
    float spin_a;        // 0 ≤ a ≤ M, zero for Schwarzschild
    float rs;            // 2·M (cached so the kernel avoids the mul)
    float tan_half_fov;  // tan(fov_y / 2)

    // Thin-disc annulus.
    float disc_r_inner;
    float disc_r_outer;
    float disc_peak_T;  // peak temperature in K
    float aspect;       // width / height

    // Integrator / viewport.
    float h_step;       // RK4 step in affine parameter λ
    float escape_r;     // bail radius — ray has escaped
    float horizon_cut;  // bail radius — ray is captured (1.02 rs)
    float time_sec;     // for animation

    sing_uint width;
    sing_uint height;
    sing_uint metric_type;  // SING_METRIC_*
    sing_uint flags;        // SING_FLAG_*

    sing_uint max_steps;    // integrator step budget per ray
    sing_uint frame_index;  // for future jitter / accumulation
    sing_uint supersample;  // 1 / 2 / 4 per-pixel subsample grid
    sing_uint pad_b;

    // Cinematic controls — blit / bloom.
    float exposure;         // linear multiplier pre tone-map (default 1.0)
    float bloom_threshold;  // luminance above which bloom extracts
    float bloom_strength;   // how hard to add bloom back after tone-map
    float disc_turbulence;  // procedural surface detail amplitude [0,1]
};

#endif  // SINGULARITY_RENDER_METAL_UNIFORMS_H
