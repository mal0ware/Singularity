# Phase 7 — WebGPU + WASM browser demo

Status: **planned, not yet implemented**. The Vulkan backend (Phase 4/5) and
the macOS Metal backend are both runtime-ready. Phase 7 extends the same
renderer into the browser. This doc is the runway for that implementation.
It exists because the WebGPU port is straightforward but large enough that
starting from a blank page would waste a couple of hours re-deriving design
decisions we already made for Metal/Vulkan.

## The plan in one line

Port `render/vulkan/{shaders,vulkan_backend.*}` → `render/webgpu/` via the
Emscripten `webgpu.h` surface; compile the result to WASM with
`SINGULARITY_BUILD_WEB=ON`; load the WASM from a minimal `web/index.html`
that wires WebGPU to a `<canvas>`.

## Why WebGPU (not WebGL)

- Compute shaders. The geodesic integrator is a compute kernel; WebGL 2
  has no compute stage. Doable with fragment-shader trickery (render-to-
  texture as a compute surrogate) but painful, and we don't need to when
  every modern browser (Chrome 113+, Safari 17.4+, Edge 113+) ships WebGPU
  by default today.
- The binding model (bind groups ≈ descriptor sets, bind-group layouts ≈
  descriptor-set layouts) maps 1:1 to what we already have in Vulkan.
  Metal-style argument buffers map similarly.

## Binding tables (reuse as-is from Vulkan)

The Vulkan descriptor-set layouts established in
`render/vulkan/vulkan_backend.cpp` `build_descriptor_layouts()` translate
directly to WebGPU `GPUBindGroupLayoutDescriptor`s. Keep the binding
numbers identical so the shaders port without renumbering.

### Geodesic compute (set 0)

| Binding | Vulkan type           | WebGPU type                           | HLSL/WGSL |
|---------|------------------------|----------------------------------------|-----------|
| 0       | `UNIFORM_BUFFER`       | `buffer{type:"uniform"}`               | `Uniforms` |
| 1       | `STORAGE_IMAGE`        | `storageTexture{format:"rgba16float", access:"write-only"}` | `output`  |

### Bloom (set 0, shared by extract + blur_h + blur_v)

| Binding | Vulkan type           | WebGPU type                           |
|---------|------------------------|----------------------------------------|
| 0       | `SAMPLED_IMAGE`        | `texture{sampleType:"unfilterable-float"}` |
| 1       | `STORAGE_IMAGE`        | `storageTexture{format:"rgba16float", access:"write-only"}` |
| 2       | `UNIFORM_BUFFER`       | `buffer{type:"uniform"}` (`ExtractParams`) |

### Blit fragment (set 0)

| Binding | Vulkan type           | WebGPU type                           |
|---------|------------------------|----------------------------------------|
| 0       | `SAMPLED_IMAGE`        | `texture{sampleType:"float"}` (hdr)   |
| 1       | `SAMPLED_IMAGE`        | `texture{sampleType:"float"}` (bloom) |
| 2       | `SAMPLER`              | `sampler{type:"filtering"}`           |
| 3       | `UNIFORM_BUFFER`       | `buffer{type:"uniform"}` (`BlitParams`) |

## HLSL → WGSL translation cheatsheet (our shaders specifically)

| HLSL                                  | WGSL                                             |
|---------------------------------------|--------------------------------------------------|
| `[[vk::binding(b, s)]] cbuffer X`     | `@group(s) @binding(b) var<uniform> x: X;`       |
| `[[vk::binding(b, s)]] RWTexture2D`   | `@group(s) @binding(b) var x: texture_storage_2d<rgba16float, write>;` |
| `[[vk::binding(b, s)]] Texture2D<float4>` | `@group(s) @binding(b) var x: texture_2d<f32>;` |
| `[[vk::binding(b, s)]] SamplerState`  | `@group(s) @binding(b) var x: sampler;`          |
| `float4 main(uint2 gid : SV_DispatchThreadID)` | `@compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) gid: vec3u)` |
| `[numthreads(8, 8, 1)]`               | `@compute @workgroup_size(8, 8, 1)`              |
| `main_vs(uint vid : SV_VertexID)`     | `@vertex fn main_vs(@builtin(vertex_index) vid: u32)` |
| `main_ps(... : SV_Target)`            | `@fragment fn main_ps(...) -> @location(0) vec4f` |
| `float3(x,y,z)`                       | `vec3f(x,y,z)`                                   |
| `lerp(a,b,t)`                         | `mix(a,b,t)`                                     |
| `frac(x)`                             | `fract(x)`                                       |
| `atan2(y,x)`                          | `atan2(y,x)` (same)                              |
| `clamp/saturate`                      | `clamp(x,0,1)` for saturate; `clamp` same        |
| `Texture2D.Load(int3)`                | `textureLoad(t, coord, 0)`                       |
| `Texture2D.Sample(s, uv)`             | `textureSample(t, s, uv)`                        |
| `RWTexture2D[gid] = v`                | `textureStore(t, gid, v)`                        |
| `dst.GetDimensions(w, h)`             | `let d = textureDimensions(dst); w = d.x; h = d.y;` |

### The shared-physics-headers problem

Our HLSL/MSL shaders `#include` the math from `shared_shader/`. WGSL has
**no preprocessor**. Two tenable approaches:

1. **Inline the math.** Paste the relevant snippets from `geodesic_math.h`,
   `kerr_math.h`, `kerr_hamilton.h`, `disc_intersection.h` directly into
   `render/webgpu/shaders/geodesic_kernel.wgsl`. Maintain drift risk: when
   Phase 8's wormhole metric lands in the shared header, the WGSL copy
   needs a parallel edit. Mitigation: a small Python script that runs at
   build time, slurps the .h files, and produces a `.wgsl` by string
   replacement (`#define lerp_scalar mix`, handle braces / types). This
   script already exists in spirit in a handful of engines — we'd write
   our own ~100-line version.
2. **SPIR-V -> WGSL cross-compile.** `naga` (the Rust crate used by
   `wgpu`) can ingest our existing `geodesic_kernel.spv` and emit WGSL.
   This is what `wgpu-native` does internally. Pros: zero shader re-port.
   Cons: naga is a Rust build-time dependency; the HLSL compute features
   we use (`int3` fetch coordinates, `[[vk::binding]]` attributes) all
   round-trip fine, but the output is less readable than hand-authored
   WGSL.

**Recommendation**: start with approach (1). The kernel is ~400 lines;
hand-porting is a one-off afternoon. If drift becomes a real problem
(say, Phase 8 wormhole), pivot to approach (2).

## Emscripten toolchain

```bash
# One-time setup per dev machine:
git clone https://github.com/emscripten-core/emsdk ~/emsdk
cd ~/emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Confirm (should print a version ≥ 3.1.60):
em++ --version
```

## CMake skeleton

**`render/webgpu/CMakeLists.txt`**:

```cmake
# Only compiled when SINGULARITY_BUILD_WEB=ON, which itself forces
# Emscripten per the top-level sanity check.
add_library(singularity_render_webgpu STATIC
    webgpu_backend.cpp
    webgpu_backend.hpp)
target_include_directories(singularity_render_webgpu
    PUBLIC  ${CMAKE_SOURCE_DIR}/render/include
    PRIVATE ${CMAKE_SOURCE_DIR}/shared_shader)
# Emscripten's webgpu.h is in $EMSDK/upstream/emscripten/cache/sysroot.
# -sUSE_WEBGPU=1 enables it at link time.
target_compile_options(singularity_render_webgpu PRIVATE
    -sUSE_WEBGPU=1)
target_link_options(singularity_render_webgpu PRIVATE
    -sUSE_WEBGPU=1
    -sWASM=1
    -sEXPORTED_FUNCTIONS=['_main','_singularity_render_frame','_singularity_resize']
    --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/shaders@/shaders)
add_library(singularity::render_webgpu ALIAS singularity_render_webgpu)
```

**Top-level `CMakeLists.txt`** — replace the current `add_executable(app)`
with:

```cmake
if(SINGULARITY_BUILD_WEB)
    add_subdirectory(render/webgpu)
    add_executable(singularity_web web/main.cpp)
    target_link_libraries(singularity_web PRIVATE singularity::render_webgpu singularity::core)
    set_target_properties(singularity_web PROPERTIES SUFFIX ".html")
    target_link_options(singularity_web PRIVATE
        -sUSE_WEBGPU=1 -sWASM=1 --shell-file ${CMAKE_SOURCE_DIR}/web/shell.html)
endif()
```

## File creation order

Do them in this order — each step compiles before starting the next:

1. **`render/webgpu/shaders/{geodesic_kernel,bloom,blit}.wgsl`** — port
   the math by hand using the cheatsheet above. Verify with
   `naga --validate < file.wgsl` (naga-cli via cargo).
2. **`render/webgpu/webgpu_backend.{hpp,cpp}`** — clone
   `vulkan_backend.{hpp,cpp}`, replace every `vk::X` with the Emscripten
   `wgpu::X` or C `WGPUX`. The resource flow is identical: instance ->
   adapter -> device -> swapchain -> pipelines -> bind groups -> frame.
3. **`render/webgpu/CMakeLists.txt`** — as sketched above.
4. **`web/main.cpp`** — a single `int main()` that:
   - Requests a WebGPU adapter via `emscripten_webgpu_get_device()`.
   - Creates a swapchain bound to the `<canvas>`.
   - Instantiates `singularity::webgpu::create_webgpu_backend()`.
   - Runs the event loop via `emscripten_set_main_loop`.
5. **`web/index.html`** — `<canvas id="canvas">` + `<script src="singularity_web.js">`.
   Template comes for free via `-sSHELL_FILE=web/shell.html`.
6. **Top-level CMake** — the `SINGULARITY_BUILD_WEB` branch above.
7. **CI** — add a third leg to `build-test`: Emscripten on ubuntu-latest,
   `emcmake cmake -B build-wasm -DSINGULARITY_BUILD_WEB=ON` then
   `cmake --build build-wasm`. Artefact: the `.wasm` + `.js` + `.html`.

## Validation (once it runs)

- Browser: Chrome 113+ or Safari 17.4+ with `chrome://flags/#enable-unsafe-webgpu`
  set (or nothing — WebGPU is on by default on recent builds).
- Expect: same Schwarzschild render as the native app, at ~half the
  framerate due to WASM + WebGPU driver overhead. M2 Mac Chrome: ~30 FPS
  at 960×540 is realistic for the first pass.
- Golden-image test: extend `verification/test_backend_equivalence.py`
  with a Playwright-driven headless Chrome run that pulls a `canvas.toDataURL()`
  PNG, perceptual-hashes against the Metal golden. Not in scope for the
  first WebGPU commit — add when the second Phase 7 ship gate approaches.

## References

- Emscripten WebGPU: https://emscripten.org/docs/api_reference/webgpu.html
- WebGPU Samples: https://github.com/webgpu/webgpu-samples (esp. `computeBoids` for compute-only pipelines)
- HLSL -> WGSL mapping: https://www.w3.org/TR/WGSL/ §3.2 (type equivalence)
- Naga (for optional SPIR-V -> WGSL path): https://github.com/gfx-rs/wgpu/tree/trunk/naga

## Ship-gate alignment

Phase 7's `v1.0 SHIP` exit criterion in `docs/TODO.md` requires the
WebGPU demo to run in Chrome. This doc is the plan; implementation is a
separate commit. A conservative estimate: **15–25 focused hours**
(shader ports + backend port + Emscripten glue + web harness + CI),
assuming the Vulkan backend as a template.
