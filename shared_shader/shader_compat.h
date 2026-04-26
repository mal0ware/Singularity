// shared_shader/shader_compat.h
//
// Platform bridge for the shared physics headers. Defines DEVICE, INLINE, and
// CONSTANT so the same geodesic/color math compiles in Metal Shading Language,
// HLSL (compiled to SPIR-V via DXC for Vulkan), CUDA C++, and host C++ (so the
// CPU unit tests can exercise the exact same code paths the GPU runs).
//
// See docs/ARCHITECTURE.md §4 for the rationale — shader sharing is what
// keeps four backends from drifting into four subtly different metrics.

#ifndef SINGULARITY_SHARED_SHADER_COMPAT_H
#define SINGULARITY_SHARED_SHADER_COMPAT_H

#if defined(__METAL_VERSION__)
// Metal Shading Language — a C++14 dialect with explicit address spaces.
#define DEVICE
#define INLINE inline
#define CONSTANT constant
#include <metal_stdlib>
using namespace metal;
typedef float3 vec3;
typedef float4 vec4;

#elif defined(__HLSL_VERSION) || defined(_HLSL) || defined(__SPIRV__)
// HLSL, compiled to SPIR-V by DXC for the Vulkan backend. A small
// alias layer papers over the cases where MSL / GLSL and HLSL spell
// the same intrinsic differently, so shared_shader/ and the per-
// backend shaders can use one vocabulary.
#define DEVICE
#define INLINE inline
#define CONSTANT static const
typedef float3 vec3;
typedef float4 vec4;
#define mix(a, b, t) lerp((a), (b), (t))
#define fract(x) frac(x)
// MSL `atan2(y, x)` ↔ HLSL `atan2(y, x)` — identical, no alias.
// MSL `isfinite(x)` is SM4+-native in HLSL too; no alias needed.

#elif defined(__CUDACC__)
// CUDA C++ — NVCC defines __CUDACC__ for device-side compilation.
#define DEVICE __device__
#define INLINE __forceinline__
#define CONSTANT __constant__
#include <cuda_runtime.h>
typedef float3 vec3;
typedef float4 vec4;

#else
// Host C++ — used by the CPU unit tests, CLI, and verification harness so
// every shader function can be exercised directly without a GPU.
#define DEVICE
#define INLINE inline
#define CONSTANT constexpr
#include <cmath>

#include "physics/vec_types.hpp"
// Bring the C math overloads into the unqualified namespace so code that
// writes `sin(x)` (portable shader style) compiles on the host branch
// without every math header being rewritten.
using std::abs;
using std::cos;
using std::fabs;
using std::sin;
using std::sqrt;
using std::tan;
#endif

#endif  // SINGULARITY_SHARED_SHADER_COMPAT_H
