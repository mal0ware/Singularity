# Phase 8 — CUDA offline renderer

Status: **kernel ported, Halton supersampling live**. The geodesic
ray-trace runs on the device, Schwarzschild + Kerr both render
recognisable black holes that pass the cross-backend equivalence test
against the Vulkan goldens. The CUDA-vs-CUDA regression test
(`verification/test_cuda_goldens.py`) and the cross-backend arm of
`verification/test_backend_equivalence.py` both gate the kernel.
Remaining Phase 8 items: 4K/8K still production, sequential-frame
mode for video, FFmpeg subprocess wrapper, v1.1 ship.

## What landed

`render/cuda/kernels/geodesic_kernel.cu` is now a real port of the
Metal kernel — same physics, same shared_shader/* headers, same
camera-ray construction. The only structural differences vs.
Metal/Vulkan are:

- **Tone-map inline**, not in a separate blit pass. The realtime
  paths render to `rgba16f` and apply ACES + bloom in a fragment-
  shader blit; the offline path has no swapchain so a Narkowicz ACES
  + sRGB encode runs per-sample inside the kernel and writes RGBA8
  directly to the output buffer.
- **Bloom omitted**. A separable Gaussian on a downsampled HDR target
  is two extra kernel launches; not worth it for PNG export.
- **`u.supersample` reinterpreted as direct SPP count** (1..1024)
  rather than the N×N grid size the realtime backends use. Halton(2,3)
  drives subpixel jitter; SPP=1 is special-cased to land at pixel
  centre so the goldens stay aligned with the Vulkan baselines.
- **Per-sample tone-mapping**. Each sample tone-maps to LDR before
  averaging, rather than averaging HDR and tone-mapping once.
  Linear-HDR averaging is closer to physically-correct exposure
  metering, but it dilutes the bright disc (HDR peaks ~50–100 from
  the g⁴ factor) with adjacent starfield/shadow samples (~0.01).
  Per-sample is the standard Mitsuba/Cycles pattern for filmic
  output and keeps the realtime-backend look while smoothing edges.

## Why CUDA

`singularity_cuda_cli` is the **offline** renderer — separate binary
from `singularity` (live app, Metal/Vulkan). It trades interactive
framerate for 16–256 samples-per-pixel on an RTX 3090/4090, producing
4K/8K stills and 4K video that the real-time backends can't match
without multi-frame temporal accumulation. Three reasons CUDA over
extending Vulkan compute:

- Math-heavy kernels are easier to author in CUDA C++ (unbounded loops,
  double precision, host-device `__host__ __device__` templating) than
  in HLSL / SPIR-V.
- Offline means no swapchain / surface / present gymnastics — the
  backend is `cudaMalloc` → kernel launch → `cudaMemcpy` → PNG.
- NVIDIA-only is acceptable for the Phase 8 stretch; the portable live
  path already ships on Metal + Vulkan.

## Prereqs on the dev machine

1. **CUDA Toolkit 12.4 or newer** (covers Ampere + Ada fatbins):
   <https://developer.nvidia.com/cuda-downloads>. On Windows the installer
   needs admin and ~3 GB. Confirm `nvcc --version` prints a 12.x after
   install.
2. **An NVIDIA GPU** with driver matching the toolkit — the stub kernel
   will `cudaMalloc` fail at runtime on a driver mismatch.
3. **CMake 3.27+** (already the project minimum) — wires
   `enable_language(CUDA)` and `find_package(CUDAToolkit)`.

## Build + smoke

CUDA needs `cl.exe` (the MSVC host compiler) on the PATH. Activate
vcvars64 from a VS Build Tools install before running CMake; Ninja
is the simplest generator:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-cuda -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DSINGULARITY_BACKEND_CUDA=ON `
    -DSINGULARITY_BACKEND_VULKAN=OFF
cmake --build build-cuda --target singularity_cuda_cli -j 4

# Schwarzschild headline still — 256x256 SPP=1 lands at the same
# camera the Vulkan goldens use.
.\build-cuda\cuda_cli\singularity_cuda_cli.exe --output schw.png --res 256x256

# Headline 4K Kerr still — under 2 minutes on Ampere.
.\build-cuda\cuda_cli\singularity_cuda_cli.exe `
    --output kerr_a094_4k.png --res 3840x2160 --samples-per-pixel 256 `
    --scene scenes/kerr_a094.scene
```

`CUDA_ARCHITECTURES "80;86;89"` on the backend target covers Ampere
consumer (30-series), Ampere datacenter (A100), and Ada (40-series).

## Validation

Two pytest gates run against the kernel today, both opt-in via the
`singularity_cuda_cli` fixture (skipped cleanly when no CUDA build
is present):

1. **CUDA-vs-CUDA regression** — `verification/test_cuda_goldens.py`
   pHash-diffs three baselines (`schw`, `kerr_a05`, `kerr_a094`) at
   256×256 SPP=1 against the committed `verification/golden/cuda/*.png`.
   Tight tolerance (distance ≤ 4) — same backend on the same hardware
   should hash-match within pHash's resolution. Mirror of the Vulkan
   golden gate.
2. **Cross-backend equivalence** —
   `verification/test_backend_equivalence.py::test_cuda_structurally_matches_vulkan_golden`
   diffs CUDA at SPP=1 against the existing Vulkan goldens. Tolerance
   20 (calibrated: schw lands at distance=4, kerr_a094 at 12; 8 bits
   of headroom for tone-map drift). Directly satisfies the Phase 8
   TODO line "CUDA at 1 sample/pixel should match Metal/Vulkan within
   tolerance."

Conservation checks already exist for the host C++ paths in
`verification/test_kerr_geometry.py` — running them through the CUDA
kernel would need a side-channel to dump per-step state from a single
ray. Open work; not gating Phase 8 today.

## Performance

Measured on Ampere (current dev box) — full 4K Schwarzschild at
256 SPP renders in **~1:19 wall-clock**. Comfortably under the
2-minute headline budget; 64 SPP at the same resolution lands closer
to 25 s. Scaling holds linearly with `width × height × spp` so an 8K
Kerr at 256 SPP is roughly 5–6 minutes — fine for a one-shot
ship-render, too slow for live preview.

Profile with **Nsight Compute** if the kernel ever regresses past
~3 minutes for 4K/256 SPP — the usual culprits:

- Divergent execution inside the integrator step when some threads in a
  warp cross the horizon earlier than others. Mitigate with persistent-
  thread compaction (one warp = one packet of ≤32 active rays, refill
  from a work queue when any thread terminates).
- `float` ↔ `double` churn inside `geodesic_rhs`. We use `float` in the
  kernels today; double precision would halve throughput on the 3090
  (FP64:FP32 ratio = 1:32 on GA102). Keep `float` unless a specific
  metric term demonstrably needs the range.

## Ship-gate alignment

Phase 8's `v1.1 SHIP` in `docs/TODO.md` requires three artefacts:

- 4K Schwarzschild still on a 3090.
- 4K Kerr `a/M=0.94` still.
- 30-second 4K Kerr video (sequential PNG + FFmpeg encode).

The scaffolded `cuda_cli` covers single-frame PNG. Video needs the
sequential-frames mode (trivial outer loop over camera states) plus
an FFmpeg subprocess invocation — docs/TODO.md Phase 8 CLI rows call
this out and they stay open after the geodesic port lands.

## References

- CUDA Toolkit docs: <https://docs.nvidia.com/cuda/>
- Nsight Compute (kernel profiler): <https://developer.nvidia.com/nsight-compute>
- Halton sequence construction: Physically Based Rendering §7.4.3 (PBRT)
- Persistent-thread ray-trace pattern: Aila & Laine, "Understanding the
  Efficiency of Ray Traversal on GPUs" (HPG 2009) — still relevant for
  divergent-step compute workloads.
