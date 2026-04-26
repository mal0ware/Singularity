---
layout: default
title: Architecture
permalink: /architecture/
---

# Architecture & Methods — Singularity

> How the simulator is built. Code organization, the backend abstraction layer, the Metal and Vulkan compute pipelines, the CUDA stretch backend, shader sharing across MSL/HLSL/CUDA, the build system, CI/CD across two platforms, distribution, and testing strategy. The *physics* of what each kernel computes lives in `PHYSICS.md`; this doc covers the *engineering* of how it computes it across three GPU APIs and one C++ codebase.

---

## 1. High-level architecture

```
                    ┌──────────────────────────────────────┐
                    │       singularity (cross-platform)   │
                    │                                      │
                    │  ┌────────────────────────────────┐  │
                    │  │ App shell (SDL3 + Dear ImGui)  │  │
                    │  │ - window, input, settings UI   │  │
                    │  └────────────────┬───────────────┘  │
                    │                   │                  │
                    │  ┌────────────────┼────────────────┐ │
                    │  │ Renderer (selects backend)      │ │
                    │  │  ┌────────────────────────────┐ │ │
                    │  │  │   abstract RenderBackend   │ │ │
                    │  │  └────────────────────────────┘ │ │
                    │  │            │       │            │ │
                    │  │      ┌─────┼───┐  ┌┼──────────┐ │ │
                    │  │      │ Metal   │  │ Vulkan    │ │ │
                    │  │      │ Backend │  │ Backend   │ │ │
                    │  │      └─────────┘  └───────────┘ │ │
                    │  └─────────────────────────────────┘ │
                    │                   │                  │
                    │  ┌────────────────┼────────────────┐ │
                    │  │   shared C++ physics core       │ │
                    │  │   - State, Camera, Settings     │ │
                    │  │   - geodesic_rhs() (in headers) │ │
                    │  │   - Christoffel symbols         │ │
                    │  └─────────────────────────────────┘ │
                    └──────────────────────────────────────┘

      ┌─────────────────┐                     ┌──────────────────────┐
      │ singularity_cli │                     │ singularity_cuda_cli │
      │ (headless,      │                     │ (NVIDIA-only,        │
      │  drives backend │                     │  offline 4K/8K       │
      │  for Python     │                     │  renderer using      │
      │  verification)  │                     │  CUDA backend)       │
      └─────────────────┘                     └──────────────────────┘
              ▲                                          ▲
              │                                          │
      ┌───────┴──────────────────────────────────────────┴───────┐
      │ verification/ (Python)                                   │
      │ - SymPy: symbolic Christoffel derivation                 │
      │ - SciPy: independent reference integrator                │
      │ - pytest: drives all tests, including                    │
      │   golden-image diffs and backend-equivalence             │
      └──────────────────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────────────────┐
      │ web/ (Next.js + TypeScript + WebGPU)                     │
      │ - imports physics core compiled to WASM via Emscripten   │
      │ - WGSL kernels dispatched by TS host                     │
      └──────────────────────────────────────────────────────────┘
```

**Key property of this layout:** the physics math lives in *one* place — shared C++ headers under `core/include/physics/`. Every backend (Metal, Vulkan, CUDA, WebGPU) calls those headers from its own kernel boilerplate. A physics bug fixed in `schwarzschild.hpp` is fixed everywhere simultaneously. The backend-equivalence test (`verification/test_backend_equivalence.py`) catches any drift if it appears.

## 2. Repository layout

```
singularity/
├── README.md
├── docs/
│   ├── PRD.md
│   ├── PHYSICS.md
│   ├── ARCHITECTURE.md
│   └── TODO.md
├── core/                       # platform-agnostic C++ core
│   ├── include/
│   │   ├── physics/            # SHARED across all backends
│   │   │   ├── state.hpp       # State vector
│   │   │   ├── schwarzschild.hpp
│   │   │   ├── kerr.hpp
│   │   │   └── integrator.hpp
│   │   ├── camera.hpp
│   │   ├── settings.hpp
│   │   └── scene.hpp
│   └── src/
│       └── (cpu-side helpers, settings JSON, etc.)
├── render/
│   ├── include/
│   │   └── render_backend.hpp  # The abstraction
│   ├── metal/                  # Metal implementation
│   │   ├── metal_backend.hpp
│   │   ├── metal_backend.mm    # Objective-C++ for ARC interop
│   │   └── shaders/
│   │       ├── geodesic_kernel.metal
│   │       ├── disc_intersection.metal
│   │       └── blit.metal
│   ├── vulkan/                 # Vulkan implementation
│   │   ├── vulkan_backend.hpp
│   │   ├── vulkan_backend.cpp
│   │   └── shaders/
│   │       ├── geodesic_kernel.hlsl   # → DXC → SPIR-V
│   │       ├── disc_intersection.hlsl
│   │       └── blit.hlsl
│   └── cuda/                   # CUDA stretch implementation
│       ├── cuda_backend.cu
│       └── kernels/
│           └── geodesic_kernel.cu
├── shared_shader/              # SHARED math used by every backend's shaders
│   ├── shader_compat.h         # platform macros (DEVICE, INLINE, etc.)
│   ├── geodesic_math.h         # the actual physics, callable from MSL/HLSL/CUDA
│   └── color_math.h
├── app/                        # cross-platform app shell
│   ├── main.cpp
│   ├── app_shell.cpp           # SDL3 window, ImGui setup
│   └── settings_ui.cpp
├── cli/                        # headless binary for verification
│   └── main.cpp
├── cuda_cli/                   # offline renderer (stretch)
│   └── main.cpp
├── tests/                      # C++ unit tests (Catch2)
│   ├── test_camera.cpp
│   ├── test_integrator.cpp
│   ├── test_schwarzschild.cpp
│   └── test_kerr.cpp
├── verification/               # Python physics verification
│   ├── christoffel_sympy.py
│   ├── test_photon_sphere.py
│   ├── test_deflection.py
│   ├── test_isco.py
│   ├── test_redshift.py
│   ├── test_golden_images.py
│   ├── test_backend_equivalence.py
│   ├── golden/                 # Reference PNGs (per backend)
│   │   ├── metal/
│   │   └── vulkan/
│   └── conftest.py
├── web/                        # Next.js docs site + WebGPU demo
│   ├── app/
│   ├── content/                # MDX, mirrors PHYSICS/ARCHITECTURE
│   ├── components/Demo/
│   └── public/wasm/            # Built-from-core WASM artifacts
├── third_party/
│   ├── metal-cpp/              # Apple
│   ├── Vulkan-Hpp/             # Khronos
│   ├── SDL/                    # SDL3
│   ├── imgui/                  # Dear ImGui
│   ├── catch2/
│   └── stb/
├── CMakeLists.txt              # cross-platform, conditional backend selection
├── BUILDING.md                 # platform-specific build instructions
├── .clang-format
├── .clang-tidy
└── .github/workflows/
    ├── ci.yml                  # matrix: macos-14 + windows-2022
    ├── release.yml             # tag → signed .dmg + .msi
    └── docs.yml                # web/ → Vercel
```

## 3. The backend abstraction

The interface is *small* on purpose. Over-abstracting is the failure mode.

```cpp
// render/include/render_backend.hpp
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace singularity {

struct WindowHandle {
    void* native_window;        // NSWindow* on Mac, HWND on Windows
    void* native_view;          // CAMetalLayer*, NSView* on Mac; ignored on Win
};

struct RenderConfig {
    uint32_t width;
    uint32_t height;
    bool vsync_enabled;
};

struct Scene {
    enum class MetricType { Schwarzschild, Kerr };
    MetricType metric;
    float mass_solar;
    float spin_a_over_M;        // 0 for Schwarzschild
    float disc_inner_M;
    float disc_outer_M;
    bool disc_doppler_on;
    bool disc_redshift_on;
    bool disc_texture_on;
    bool show_overlay;
};

struct CameraState {
    float position[3];
    float basis[9];             // 3x3 orientation matrix, row-major
    float fov_y_radians;
};

struct ImageData {
    std::vector<uint8_t> pixels_rgba;
    uint32_t width;
    uint32_t height;
};

class RenderBackend {
public:
    virtual ~RenderBackend() = default;

    // Lifecycle
    virtual bool initialize(WindowHandle window, RenderConfig config) = 0;
    virtual void shutdown() = 0;
    virtual void resize(uint32_t width, uint32_t height) = 0;

    // Per-frame
    virtual void render_frame(const Scene& scene, const CameraState& camera) = 0;

    // Optional: for verification + screenshot export
    virtual ImageData capture_frame() = 0;

    // Backend identification
    virtual const char* name() const = 0;  // "Metal", "Vulkan", "CUDA"
};

// Factory selects based on platform + build config
std::unique_ptr<RenderBackend> create_default_backend();

} // namespace singularity
```

**What this interface deliberately omits:**

- No `create_texture` / `create_buffer` / `create_pipeline` — those live inside each backend's implementation. Exposing them would force the abstraction to know about API-specific concepts (descriptor sets, MTLBuffer types, etc.) and the abstraction would leak.
- No "draw a triangle" or "bind shader" — too low-level. The abstraction is at the level of "render a black hole frame," not "submit a draw call."
- No Vulkan-style explicit synchronization — each backend handles its own internal command buffer/queue lifecycle.

**What's inside each backend:**

- Each backend creates its own device, command queue/list, pipeline state objects, ImGui integration, swapchain.
- The "scene" is small enough (<100 bytes) to upload as a uniform every frame.
- Texture loading (skybox, accretion-disc LUT) happens at `initialize()` time using a backend-specific path — but the *file* loaded is the same PNG.

## 4. Sharing physics math across MSL, HLSL, and CUDA

### 4.1 The compatibility header

```c
// shared_shader/shader_compat.h
//
// Defines DEVICE, INLINE, and other macros so the same math headers compile
// in MSL, HLSL, CUDA, and even host C++.

#if defined(__METAL_VERSION__)
    // Metal Shading Language (a C++14 dialect)
    #define DEVICE
    #define INLINE inline
    #define CONSTANT constant
    #include <metal_stdlib>
    using namespace metal;
    typedef float3 vec3;
    typedef float4 vec4;
#elif defined(__HLSL_VERSION) || defined(_HLSL)
    // HLSL (used for Vulkan via DXC → SPIR-V)
    #define DEVICE
    #define INLINE inline
    #define CONSTANT static const
    typedef float3 vec3;
    typedef float4 vec4;
#elif defined(__CUDACC__)
    // CUDA C++
    #define DEVICE __device__
    #define INLINE __forceinline__
    #define CONSTANT __constant__
    #include <cuda_runtime.h>
    typedef float3 vec3;
    typedef float4 vec4;
#else
    // Host C++ (so we can unit-test the math on CPU)
    #define DEVICE
    #define INLINE inline
    #define CONSTANT constexpr
    #include "core/include/physics/vec_types.hpp"  // typedefs vec3, vec4
#endif
```

### 4.2 The shared math header

```c
// shared_shader/geodesic_math.h
#include "shader_compat.h"

struct State {
    float t, r, theta, phi;
    float ut, ur, utheta, uphi;
};

DEVICE INLINE State geodesic_rhs_schwarzschild(State s, float rs) {
    float f = 1.0f - rs / s.r;
    float r2 = s.r * s.r;
    float sin_t = sin(s.theta);
    float cos_t = cos(s.theta);

    State d;
    d.t = s.ut;
    d.r = s.ur;
    d.theta = s.utheta;
    d.phi = s.uphi;

    // Acceleration terms — Christoffel symbols per PHYSICS.md §3
    d.ut = -(rs / (r2 * f)) * s.ut * s.ur;
    d.ur = -(rs * f / (2.0f * r2)) * s.ut * s.ut
           + (rs / (2.0f * r2 * f)) * s.ur * s.ur
           + s.r * f * (s.utheta * s.utheta + sin_t * sin_t * s.uphi * s.uphi);
    d.utheta = -2.0f * s.ur * s.utheta / s.r
               + sin_t * cos_t * s.uphi * s.uphi;
    d.uphi = -2.0f * s.ur * s.uphi / s.r
             - 2.0f * (cos_t / sin_t) * s.utheta * s.uphi;
    return d;
}

DEVICE INLINE State rk4_step(State y, float h, float rs) {
    State k1 = geodesic_rhs_schwarzschild(y, rs);
    State k2 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k1, 0.5f * h)), rs);
    State k3 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k2, 0.5f * h)), rs);
    State k4 = geodesic_rhs_schwarzschild(state_add(y, state_scale(k3, h)), rs);
    State sum = state_add(state_add(k1, state_scale(k2, 2.0f)),
                          state_add(state_scale(k3, 2.0f), k4));
    return state_add(y, state_scale(sum, h / 6.0f));
}
```

This *one* file is included from:
- `render/metal/shaders/geodesic_kernel.metal`
- `render/vulkan/shaders/geodesic_kernel.hlsl`
- `render/cuda/kernels/geodesic_kernel.cu`
- `tests/test_schwarzschild.cpp` (host CPU build for unit-testing the math directly)

A bug fixed once is fixed everywhere.

## 5. The Metal backend

### 5.1 Pipeline state objects

One `MTLDevice`, one `MTLCommandQueue`, two pipeline states:

- **`geodesicPipeline`** — `MTLComputePipelineState` for `geodesic_kernel`. Binds: skybox texture (read), accretion-disc LUT, scene-uniforms buffer, output color texture (write).
- **`blitPipeline`** — `MTLRenderPipelineState` for `blit_vertex` + `blit_fragment`. Tone-maps the compute output to the drawable. Single fullscreen triangle.

Both PSOs are created at `initialize()` and reused every frame.

### 5.2 Per-frame command buffer

```cpp
auto cmdBuf = commandQueue->commandBuffer();

auto computeEnc = cmdBuf->computeCommandEncoder();
computeEnc->setComputePipelineState(geodesicPipeline);
computeEnc->setTexture(skyboxTexture, 0);
computeEnc->setTexture(outputTexture, 1);
computeEnc->setBuffer(uniformsBuffer[currentFrame % 3], 0, 0);  // triple-buffered
computeEnc->dispatchThreads(MTL::Size(width, height, 1),
                             MTL::Size(32, 32, 1));            // M-series sweet spot
computeEnc->endEncoding();

auto rpd = view->currentRenderPassDescriptor();
auto renderEnc = cmdBuf->renderCommandEncoder(rpd);
renderEnc->setRenderPipelineState(blitPipeline);
renderEnc->setFragmentTexture(outputTexture, 0);
renderEnc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
ImGuiOverlay::render(renderEnc);   // ImGui draws into the same render pass
renderEnc->endEncoding();

cmdBuf->presentDrawable(view->currentDrawable());
cmdBuf->commit();
```

Triple-buffered uniforms prevent CPU/GPU contention.

### 5.3 Window interop via SDL3

```cpp
SDL_Window* window = SDL_CreateWindow("Singularity", 1280, 720,
                                      SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE);
SDL_MetalView metal_view = SDL_Metal_CreateView(window);
CAMetalLayer* layer = (CAMetalLayer*)SDL_Metal_GetLayer(metal_view);
metalBackend->initialize({window, layer}, config);
```

## 6. The Vulkan backend

### 6.1 What's different

Vulkan is more verbose than Metal. The same Metal setup expands to:

- Instance + physical device selection + queue family selection
- Logical device + compute queue + graphics queue (often the same)
- Swapchain creation with explicit `VkSurfaceKHR` from SDL3
- Descriptor set layouts + descriptor pools + descriptor sets
- Pipeline layout + compute pipeline + graphics pipeline
- Per-frame: command pool + command buffer + fences + semaphores
- Explicit memory allocation (use VMA — Vulkan Memory Allocator — to make this sane)

Roughly 2-3× the code of the Metal backend, but it follows a well-known recipe (Sascha Willems's Vulkan samples are the canonical reference).

### 6.2 HLSL → SPIR-V → Vulkan

Shaders authored in HLSL. Compiled at build time:

```bash
dxc -T cs_6_0 -E main -spirv \
    -fspv-target-env=vulkan1.3 \
    -I shared_shader/ \
    geodesic_kernel.hlsl -Fo geodesic_kernel.spv
```

`shared_shader/shader_compat.h` provides the `DEVICE`/`INLINE`/`vec3` aliases that let the same `geodesic_math.h` compile in both MSL and HLSL.

### 6.3 Per-frame command buffer

```cpp
vk::CommandBuffer cmd = currentFrame.commandBuffer;
cmd.begin({});

// Compute pass
cmd.bindPipeline(vk::PipelineBindPoint::eCompute, geodesicPipeline);
cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, geodesicLayout,
                       0, currentFrame.descriptorSet, {});
cmd.dispatch((width + 31) / 32, (height + 31) / 32, 1);

// Barrier: compute output → fragment shader read
vk::ImageMemoryBarrier2 barrier{ ... };
cmd.pipelineBarrier2({ ..., barrier });

// Render pass for blit + ImGui
cmd.beginRenderingKHR({...});
cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, blitPipeline);
cmd.draw(3, 1, 0, 0);
ImGuiOverlay::render(cmd);
cmd.endRenderingKHR();

cmd.end();
graphicsQueue.submit2(...);
swapchain.present(...);
```

Vulkan-Hpp gives RAII and exception-safe handles, removing most of the manual `vkDestroy*` bookkeeping of raw Vulkan.

## 7. The CUDA backend (Phase 8 stretch)

### 7.1 Why CUDA gets its own backend

CUDA is *offline only* in our use — high-resolution stills and video for the docs site, leveraging the 3090's compute headroom. It's not used for the interactive desktop apps because:

- CUDA-graphics interop is messy (CUDA ↔ DirectX or CUDA ↔ Vulkan), more complexity than the project needs.
- The whole *value* of a CUDA backend is unbounded compute time per frame — at 256 samples per pixel and 8K resolution, even a 3090 takes seconds per frame.

So `singularity_cuda_cli` is a separate binary that takes a JSON scene config and outputs PNG frames or an FFmpeg-encoded MP4.

### 7.2 CUDA kernel structure

```cuda
__global__ void geodesic_kernel(
    cudaTextureObject_t skybox,
    cudaTextureObject_t disc_lut,
    Uniforms u,
    uchar4* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 accum = make_float4(0, 0, 0, 0);
    for (int s = 0; s < u.samples_per_pixel; ++s) {
        float2 jitter = halton_2d(s);
        Ray r = generate_ray(u.camera, x, y, width, height, jitter);
        accum += integrate_ray(r, u, skybox, disc_lut);
    }
    accum /= float(u.samples_per_pixel);
    output[y * width + x] = tonemap_to_srgb(accum);
}
```

Same `geodesic_math.h` shared with Metal and Vulkan. The kernel boilerplate adds antialiasing (Halton sequence supersampling) and removes the real-time constraint.

## 8. Camera & ray generation

### 8.1 Pinhole model (CPU side, in `core/`)

```cpp
struct Ray {
    float3 origin;
    float3 direction;
};

Ray generate_ray(const CameraState& cam, uint32_t x, uint32_t y,
                 uint32_t w, uint32_t h, float2 jitter = {0.5f, 0.5f})
{
    float aspect = float(w) / float(h);
    float fov_scale = tan(cam.fov_y_radians * 0.5f);

    float2 ndc = (float2{float(x) + jitter.x, float(y) + jitter.y}
                  / float2{float(w), float(h)}) * 2.0f - 1.0f;
    ndc.x *= aspect * fov_scale;
    ndc.y *= fov_scale;

    float3 dir_cam = normalize(float3{ndc.x, ndc.y, -1.0f});
    float3 dir_world = mul(cam.basis, dir_cam);
    return Ray{cam.position, dir_world};
}
```

This same function compiles in MSL, HLSL, CUDA, and host C++.

### 8.2 Conversion to spacetime initial state

Cartesian `(position, direction)` → Boyer-Lindquist `(t, r, θ, φ)` with `t = 0`. Four-velocity components set so that `g_μν u^μ u^ν = 0` (solve for `u^t`).

Reference: JMO §3.1.

### 8.3 Orbital camera controls

CPU-side state: `azimuth`, `elevation`, `distance`. SDL3 mouse drag updates azimuth/elevation; scroll updates distance. Camera basis reconstructed each frame from these three scalars and the BH center.

## 9. Build system — CMake

### 9.1 Top-level structure

```cmake
cmake_minimum_required(VERSION 3.27)
project(singularity LANGUAGES CXX)

option(SINGULARITY_BACKEND_METAL "Build Metal backend" ${APPLE})
option(SINGULARITY_BACKEND_VULKAN "Build Vulkan backend" ${WIN32})
option(SINGULARITY_BACKEND_CUDA "Build CUDA backend" OFF)
option(SINGULARITY_BUILD_WEB "Build WASM target via Emscripten" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/SDL)
add_subdirectory(third_party/imgui)

add_library(singularity_core STATIC
    core/src/camera.cpp
    core/src/settings.cpp
    core/src/scene.cpp
)
target_include_directories(singularity_core PUBLIC core/include shared_shader)

if(SINGULARITY_BACKEND_METAL)
    enable_language(OBJCXX)
    add_subdirectory(render/metal)
endif()

if(SINGULARITY_BACKEND_VULKAN)
    find_package(Vulkan REQUIRED)
    add_subdirectory(render/vulkan)
endif()

if(SINGULARITY_BACKEND_CUDA)
    enable_language(CUDA)
    add_subdirectory(render/cuda)
endif()

add_executable(singularity app/main.cpp app/app_shell.cpp app/settings_ui.cpp)
target_link_libraries(singularity PRIVATE singularity_core SDL3::SDL3 imgui)

if(SINGULARITY_BACKEND_METAL)
    target_link_libraries(singularity PRIVATE singularity_render_metal)
endif()
if(SINGULARITY_BACKEND_VULKAN)
    target_link_libraries(singularity PRIVATE singularity_render_vulkan)
endif()
```

### 9.2 Shader compilation

Per-platform custom commands. For Metal:

```cmake
# render/metal/CMakeLists.txt
set(METAL_SHADERS
    shaders/geodesic_kernel.metal
    shaders/disc_intersection.metal
    shaders/blit.metal
)

foreach(SHADER ${METAL_SHADERS})
    get_filename_component(NAME ${SHADER} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
        COMMAND xcrun -sdk macosx metal -c ${SHADER}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
                -I ${CMAKE_SOURCE_DIR}/shared_shader
        DEPENDS ${SHADER}
    )
    list(APPEND AIR_FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air)
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES}
            -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS ${AIR_FILES}
)
```

For Vulkan:

```cmake
# render/vulkan/CMakeLists.txt
foreach(SHADER ${VULKAN_SHADERS})
    get_filename_component(NAME ${SHADER} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv
        COMMAND ${DXC_EXECUTABLE} -T cs_6_0 -E main -spirv
                -fspv-target-env=vulkan1.3
                -I ${CMAKE_SOURCE_DIR}/shared_shader
                ${SHADER} -Fo ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv
        DEPENDS ${SHADER}
    )
endforeach()
```

### 9.3 Compiler flags

```
-std=c++20
-Wall -Wextra -Wpedantic
-Wno-c++98-compat
-O3 -ffast-math       # release
-O0 -g -fsanitize=address,undefined  # debug
```

`-ffast-math` is a deliberate choice: we are not doing scientific-grade physics with NaN propagation. The verification harness catches any algorithm-level numerical issue; `-ffast-math` lets the compiler vectorize aggressively.

On Windows / MSVC, the equivalents are `/std:c++20 /W4 /O2 /fp:fast`.

## 10. CI/CD — GitHub Actions

### 10.1 Matrix CI

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - clang-format --dry-run --Werror
      - clang-tidy on all .cpp/.hpp
      - black + ruff on verification/

  build-test:
    strategy:
      matrix:
        os: [macos-14, windows-2022]
        include:
          - os: macos-14
            backend: metal
          - os: windows-2022
            backend: vulkan
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: cmake -B build -DSINGULARITY_BACKEND_${{ matrix.backend == 'metal' && 'METAL' || 'VULKAN' }}=ON
      - name: Build
        run: cmake --build build --config Release
      - name: Catch2 tests
        run: ctest --test-dir build --output-on-failure
      - name: Python verification
        run: |
          pip install -r verification/requirements.txt
          pytest verification/

  build-app:
    needs: [lint, build-test]
    strategy:
      matrix:
        os: [macos-14, windows-2022]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Build .app / .exe
        run: cmake -B build && cmake --build build --config Release
      - uses: actions/upload-artifact@v4
        with:
          name: singularity-${{ matrix.os }}
          path: build/Release/
```

### 10.2 Release pipeline

`release.yml` fires on tag push (`v*`):

- macOS job: import Apple Developer cert from secrets, `codesign --options runtime`, `xcrun notarytool submit --wait`, `xcrun stapler staple`, `create-dmg` to build `.dmg`.
- Windows job: build with `/Release`, package with `WiX` into `.msi`. If EV cert available, `signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /n "Mal" Singularity.msi`. If not, ship unsigned and document SmartScreen warning workaround.
- Both upload to GitHub Releases.

### 10.3 Docs deploy

`docs.yml` on changes to `web/` or `docs/`:

- Build WASM: `emcmake cmake -B build-wasm -DSINGULARITY_BUILD_WEB=ON && cmake --build build-wasm`
- Copy WASM + JS glue to `web/public/wasm/`
- `npm run build` in `web/`
- Vercel auto-deploys from `main`; preview from PR branches

## 11. Distribution

### 11.1 macOS — `.dmg`

Signed with Developer ID Application certificate, hardened runtime enabled, notarized through Apple's notary service, stapled. Workflow documented in `docs/RELEASE.md`.

Entitlements:

```xml
<key>com.apple.security.cs.disable-library-validation</key><true/>
<key>com.apple.security.cs.allow-jit</key><false/>
<key>com.apple.security.network.client</key><true/>
```

### 11.2 Windows — `.msi`

Built with WiX Toolset. Without EV cert, the installer triggers a Windows SmartScreen warning ("Microsoft Defender SmartScreen prevented an unrecognized app...") — users must click "More info" → "Run anyway." Documented in download instructions.

EV cert is $400/yr from Sectigo or DigiCert. Skipped at v1.0 unless user feedback demands it.

### 11.3 Update mechanism

v1.0: none. Manual download from GitHub releases page. Sparkle (macOS) and WinSparkle (Windows) considered for v2.0.

## 12. Testing strategy

Three test levels, each catching a different class of bug:

| Level | Tool | Catches | Runs in CI |
|---|---|---|---|
| **Unit** | Catch2 | Logic bugs in pure-C++ helpers (camera math, settings parsing, integrator state arithmetic) | ✓ both platforms |
| **Property / verification** | Python + SymPy + SciPy | Physics bugs (wrong Christoffel sign, drifting `E`, wrong photon sphere radius) | ✓ both platforms |
| **Visual regression** | Golden images via perceptual hash | Wrong tone mapping, color space, accretion disc orientation | ✓ both platforms (per-backend goldens) |
| **Backend equivalence** | Cross-backend perceptual hash | Drift between Metal and Vulkan implementations | ✓ when both backends compiled |

### 12.1 Why a Python harness

The C++ kernel is fast but opaque. Re-deriving the Christoffel symbols in SymPy and asserting they match the hand-coded versions catches algebra errors that no amount of unit testing in C++ would find — because the C++ is the suspect. The harness uses SciPy's `solve_ivp` (adaptive Dormand-Prince) as an independent reference integrator; if the C++ RK4 and SciPy DOPRI5 disagree by more than tolerance, one of them is wrong.

### 12.2 Backend-equivalence test

Same scene, render on Metal, render on Vulkan, compare perceptual hashes. Tolerance set to allow benign GPU floating-point variation (~4 hash bits) but catch real drift (>10).

This test is the load-bearing piece of the cross-platform claim: it asserts the abstraction *worked*.

### 12.3 Golden images

Ten 256×256 PNGs of canonical scenes per backend. Stored in `verification/golden/{metal,vulkan}/`. Regenerated only when physics changes deliberately:

1. Run `python verification/regenerate_golden.py --backend metal --backend vulkan`
2. Manually inspect every regenerated image
3. Commit with a message starting `[golden]` to make the change auditable

## 13. Performance budget

### 13.1 Mac (M2 base) at 1280×720, 60 FPS = 16.6ms total

| Pass | Budget | Notes |
|---|---|---|
| Compute (geodesic_kernel) | 12ms | The whole game |
| Render (blit + tone map) | 1ms | Trivial |
| ImGui overlay | 0.5ms | Negligible |
| Drawable present | 0.5ms | OS overhead |

Default Mac resolution is **1280×720** at 30 FPS (33ms budget) — gives 26ms for compute, generous. 60 FPS achieved on M3 Pro+.

### 13.2 Windows (RTX 3070) at 1920×1080, 60 FPS = 16.6ms

NVIDIA's compute throughput gives us roughly 5× the Mac's headroom at the same resolution. Default Win resolution is **1920×1080** at 60 FPS, with 4K offered as an option.

### 13.3 CUDA offline — no real-time constraint

256 samples per pixel × 8K = ~67M rays per frame × ~5000 RK4 steps each. On a 3090, this is roughly 20-60 seconds per frame for Schwarzschild, 60-180 seconds per frame for Kerr. Acceptable for offline.

## 14. Open technical decisions

These need answers before the relevant phase begins:

1. **Tone-mapping curve.** ACES Filmic vs Reinhard. Phase 7 decision; default to Reinhard until then.
2. **Skybox texture format.** Equirectangular (one PNG, easier authoring) vs cubemap (six PNGs, faster sampling). Phase 2 decision.
3. **Half-precision integration?** `half` storage in MSL/HLSL would halve register pressure but risks precision loss near the horizon. Test in Phase 6.
4. **WASM scope.** Phase 1's 2D model is the obvious target. The 3D Schwarzschild kernel could also run in the browser via WebGPU compute — decide at Phase 7.
5. **VMA on Vulkan or roll our own allocator?** Use VMA. The "I wrote my own GPU allocator" bullet is not worth the time cost.

## 15. References

- Apple, [Metal Shading Language Specification v3.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- Apple, [LearnMetalCPP samples](https://developer.apple.com/metal/cpp/)
- Khronos, [Vulkan-Hpp documentation](https://github.com/KhronosGroup/Vulkan-Hpp)
- Sascha Willems, [Vulkan-Samples](https://github.com/SaschaWillems/Vulkan)
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I*, Springer (1993)
- James, Tunzelmann, Franklin, Thorne (JMO), [*Class. Quantum Grav.* 32 065001](https://doi.org/10.1088/0264-9381/32/6/065001) (2015)
- Kip Thorne, *The Science of Interstellar*, W.W. Norton (2014)
