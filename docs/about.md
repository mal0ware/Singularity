---
layout: default
title: About
permalink: /about/
description: Project overview, technology stack, physics validation, license, and citation.
---

# About

Singularity is a real-time, physically accurate black-hole renderer. It integrates the null geodesic equation in Schwarzschild and Kerr spacetimes per pixel per frame on the GPU, then composites the resulting photon traces (disc samples, lensed background, event-horizon shadow) into a cinematic HDR pipeline: bloom, ACES tone-mapping, sRGB output.

The defining engineering choice is that the entire physics layer lives in `shared_shader/` as a set of header-only C++ files that compile unchanged into Metal Shading Language (MSL is C++14-based), HLSL targeting SPIR-V for Vulkan, CUDA C++, and host-side C++ for the verification harness. WGSL has no preprocessor and is the one exception; the WGSL kernels are hand-translated against the same algorithms with the cheatsheet in [`PHASE7_WEBGPU.md`](https://github.com/mal0ware/Singularity/blob/main/docs/PHASE7_WEBGPU.md). Every backend therefore consumes the same source-of-truth physics. CI's backend-equivalence test enforces that rendered output agrees within perceptual-hash tolerance across backends.

## Backends

**Metal (live).** Native macOS Apple Silicon backend. Full cinematic pipeline: HDR target, bloom extract and blur, ACES tone-mapping, sRGB. Procedural disc bands, orbital camera, ImGui panel, GPU-measured frame timings, screenshot export, settings persistence. Ad-hoc signed `.app` bundle.

**Vulkan (code-complete).** Windows and Linux. Built on Vulkan-Hpp: instance, physical-device scoring, device, swapchain, five pipelines (geodesic compute, three bloom passes, blit), HDR + bloom ping-pong, per-frame descriptors, timestamp queries. CI builds the `.msi` artefact via CPack/WiX on every push. Runtime GPU verification on real Windows hardware is the last open Phase 5 box.

**WebGPU (live).** Browser backend via Emscripten's `--use-port=emdawnwebgpu` (the Dawn standards-track port). WGSL ports of the geodesic, bloom, and blit kernels. Auto-orbits on load; drag to orbit, scroll to zoom. The settings panel is a DOM overlay rather than in-canvas ImGui, with widget changes dispatched to the wasm module via `Module.ccall`. CI runs a headless Chromium + SwiftShader smoke test on every push.

**CUDA (live).** NVIDIA Ampere or later, compute capability 8.0+. Offline-only: per-pixel-per-thread ray-trace with up to 1024 samples per pixel via Halton(2,3) sequences, with per-sample tone-mapping for filmic averaging. A 4K Kerr `a/M = 0.94` still renders in about 2:42 on an RTX 3090 at 256 SPP. 8K runs in roughly six minutes. Multi-frame mode emits a 2π camera-orbit sequence ready for ffmpeg.

## Technology stack

C++20 throughout. Apple's `metal-cpp` (2022), Khronos `Vulkan-Hpp`, Emscripten `webgpu_cpp.h` (Dawn), and CUDA C++ are first-class C++ bindings; this lets the shared physics headers compile directly inside every backend's compute kernel.

CMake 3.27+ drives a single build graph spanning desktop, web, and CUDA targets, gated by the `SINGULARITY_BACKEND_*` and `SINGULARITY_BUILD_WEB` options.

SDL3 handles windowing, input, and Vulkan surface creation. The Vulkan backend never links SDL itself; the app shell hands the backend a `vulkan_create_surface` callback so the dependency stays one-way.

Dear ImGui drives the desktop settings panel via the Metal, Vulkan, and SDL3 platform backends. The web demo uses a DOM overlay panel instead. The vendored `imgui_impl_wgpu` is incompatible with current `emdawnwebgpu` (the upstream backend errors on the Emscripten target either way), and DOM widgets are the web-native pattern in any case.

`stb_image_write` produces PNG output for the CLI rendering modes and the screenshot export path.

Catch2 v3 covers the C++ unit tests (around 70 cases, 3,800 assertions). The Python verification harness uses pytest, NumPy, SciPy, SymPy, Pillow, and ImageHash for closed-form analytic GR cross-checks plus golden-image regression.

GitHub Actions runs the matrix CI: lint (clang-format, clang-tidy, ruff, black), ASan + UBSan, gcov coverage to Codecov, build-test on macOS-14, windows-2022, and ubuntu-latest with all four backend variants, build-app producing `.app`, `.exe`, and `.msi` installer artefacts, and build-web compiling the wasm bundle and running a headless Chromium smoke test.

The web target uses Emscripten with `--use-port=emdawnwebgpu`. The geodesic, bloom, and blit kernels ship as WGSL preloaded via `--preload-file`. Bundle size is around 52 KB wasm plus 28 KB WGSL.

WiX (CPack-driven) packages the Windows `.msi` installer. Jekyll on GitHub Pages serves this site, with a hand-rolled layout and KaTeX from CDN for math.

## Physics validation

The verification harness in `verification/` enforces correctness against closed-form general relativity:

- The Schwarzschild Christoffel symbols are re-derived in SymPy from scratch and asserted equal to the hand-coded constants.
- Photon-sphere closure at $r = 1.5\,r_s$ is checked to 0.5 % over 40 impact parameters.
- Weak-field deflection matches Eddington's $4GM/(bc^2)$ to 1 % over 12 impact parameters.
- Kerr ISCO radii (prograde and retrograde) match closed-form expressions to 0.1 % for $a/M \in \{0,\,0.5,\,0.94,\,0.998\}$.
- Energy $E$, axial angular momentum $L_z$, and Carter's constant $\mathcal{Q}$ are conserved to residual $< 10^{-6}$ over 10,000 integrator steps.
- A 50,000-step ISCO conservation test samples drift every 50 steps and asserts the running maximum stays under $5 \times 10^{-5}$. This catches a linear-in-N bleed that the shorter test would miss.

Singularity is not a magnetohydrodynamic simulator. The accretion disc is a continuous emissive cloud, not individual particles carrying magnetic flux. For full GRMHD see Athena++ or IllinoisGRMHD.

## Roadmap

v1.0 (shipped): three real-time GPU backends, CUDA offline path, embedded WebGPU demo with interactive controls, public documentation site.

v1.1 (in progress): 30-second 4K Kerr camera-orbit video via the CUDA sequential-frame mode, async `capture_frame` body-fill on web, video export from the desktop apps (AVFoundation on macOS, Media Foundation on Windows), strict Metal/Vulkan perceptual-hash equivalence, Apple Developer notarization for the macOS `.dmg`, Windows runtime smoke on real hardware.

v1.2+ backlog: Morris-Thorne wormhole as a third metric, Linux `.AppImage`, binary black hole far-field superposition, real Sgr A* mode (real ephemeris and mass), adaptive mesh refinement near caustics, educational tour mode.

The exhaustive plan is in [`docs/TODO.md`](https://github.com/mal0ware/Singularity/blob/main/docs/TODO.md).

## License

Code (including shaders): [MIT](https://github.com/mal0ware/Singularity/blob/main/LICENSE).
Documentation and rendered images: [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Cite this

If you use Singularity itself in something you publish, BibTeX:

```bibtex
@software{singularity_2026,
  author  = {{mal0ware}},
  title   = {Singularity: a real-time physically-accurate black hole renderer},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/mal0ware/Singularity},
  note    = {Cross-platform Schwarzschild and Kerr geodesic ray-tracer with
             Metal, Vulkan, WebGPU, and CUDA backends sharing one C++ physics
             core}
}
```

For the underlying GR and lensing math, cite the source the implementation derives from: James, Tunzelmann, Franklin, & Thorne, *Class. Quantum Grav.* **32** (2015) 065001 (the *Interstellar* VFX paper), together with the textbooks listed in [Physics]({{ '/physics/' | relative_url }}).

## Acknowledgments

Source material that shaped the project's scope:

- Kavan Dave, *I Coded a Black Hole Simulator in C++*. Defined the MVP scope for Phases 1 through 3.
- ScienceClic, *Could the physics in Interstellar be real?*. Drove the Kerr / Doppler / Carter direction.

Physics references in [Physics]({{ '/physics/' | relative_url }}) cite Misner, Thorne, & Wheeler's *Gravitation*; Sean Carroll's *Spacetime and Geometry*; and Kip Thorne's *The Science of Interstellar*. The Kerr lensing implementation follows James, Tunzelmann, Franklin, & Thorne (2015).
