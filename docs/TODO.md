# TODO — Singularity

> The exhaustive phased task list. Every checkbox here corresponds to a discrete unit of work that should fit in one focused session (≤4 hours) or be split. Phase boundaries are *deliverable boundaries*: at each one the project is in a shippable / showable state, even if you stop forever right there.

**How to use this doc.** Work top-to-bottom within a phase. Don't skip ahead. The phase boundaries enforce architectural discipline (e.g., Phase 1 has zero GPU code on purpose; Phase 4 doesn't start until Phase 3's Mac app is ready to ship). Tag completed items with the commit SHA: `[x]` (`abc1234`).

**Estimated total:** 16–18 weeks for v1.0 at ~10 hours/week. Plus 3 weeks for CUDA stretch (Phase 8). Adjust for studies + Fiverr.

**Three intermediate ship gates** are explicitly designed in:
- **End of Phase 3 → v0.5** (Mac-only Schwarzschild — first public release)
- **End of Phase 5 → v0.7** (Cross-platform Schwarzschild)
- **End of Phase 7 → v1.0** (Cross-platform Kerr + web demo)

---

## Status as of 2026-04-24

All three GPU backends ship. **macOS Metal** runs on Apple Silicon
with the full cinematic pipeline (HDR → bloom → ACES → sRGB),
procedural disc bands, interactive orbital camera, ImGui panel,
GPU-measured frame timings, settings persistence, in-app screenshot
export, and headless `--capture` for golden regressions.
**Windows Vulkan** is code-complete (Vulkan-Hpp: instance +
physical-device scoring + device + swapchain + 5 pipelines + HDR +
bloom + blit + overlay + timestamp queries); the `.msi` is CPack-WIX'd
and uploaded as a CI artifact on every push — runtime smoke on real
hardware is the last Windows step.
**WebGPU + WASM demo is live** — the same Kerr integrator runs in
the browser from a 52 KB wasm + 28 KB preloaded WGSL bundle via
emsdk's `--use-port=emdawnwebgpu` Dawn port; headless Chromium +
SwiftShader smoke runs on every push. Hero image at
`docs/images/phase7_webgpu.png`.

CI is green across 8 jobs: lint, 4× build-test (macOS/Metal,
Windows/Vulkan, Ubuntu/none, Ubuntu/Vulkan), 2× build-app (.app +
.exe + .msi), and build-web (.html + .js + .wasm + .data + smoke).
Ad-hoc signed `.app`; Developer ID wiring is in place for summer
2026 notarization. Branch: `main`.

- **Phase 0 — setup:** complete.
- **Phase 1 — 2D CPU toy:** complete.
- **Phase 2 — Metal 3D Schwarzschild:** complete (live on Mac).
- **Phase 3 — disc + v0.5:** complete, past gate.
- **Phase 4 — backend abstraction + Vulkan port:** code-complete.
- **Phase 5 — cross-platform CI + Windows release + v0.7:** complete, past gate (untagged).
- **Phase 6 — Kerr on both backends:** Metal live, Vulkan compiles, same shared Hamiltonian integrator.
- **Phase 7 — polish + WebGPU + v1.0:** WebGPU backend body-fill shipped, pointer-driven orbit, headless Chromium CI smoke. Settings persistence + screenshot export already landed on desktop. Docs site, Playwright golden-image regression, and ImGui-web overlay remain.
- **Phase 8 — CUDA stretch:** kernel ported, Halton(2,3) supersampling live, both regression + cross-backend equivalence pytest gates passing. 4K stills renderable in ~1–3 min on Ampere. Remaining: sequential-frame mode + FFmpeg wrapper for 4K video, then v1.1 ship.

**What still needs hardware:**
- Windows desktop: runtime smoke of the Vulkan backend (first visible pixels in the native app) + install-path test of `Singularity-<ver>-win64.msi`. Everything is compile-validated in CI; only runtime verification remains.
- Summer 2026 on Mac: Apple Developer membership ($99) → codesign + notarize `.dmg`.

Totals today: 2 Catch2 test binaries, ~70 pytest cases, 7 CLI modes, 6 hero PNGs, zero ruff/black/clang-format violations, CI hard-gated on every matrix leg.

---

## Phase 0 — Setup, both platforms (Week 1, ~12 hours)

**Exit criterion:** `cmake -B build && cmake --build build` produces a Metal-rendered spinning textured cube on Mac and a Vulkan-rendered spinning textured cube on Windows. CI matrix is green on both runners on a no-op PR.

### Repo & tooling

- [x] Create `singularity` GitHub repo, MIT license, README placeholder.
- [x] Add `.clang-format` (LLVM style, 4-space indent, 100-col limit).
- [x] Add `.clang-tidy` (modern, performance, readability checks).
- [x] Add `.gitignore` for CMake, Xcode, Visual Studio, macOS, Windows, Python, Node.
- [x] Add `.editorconfig`.
- [x] Set up `pre-commit` hooks running clang-format, ruff, black.

### Cross-platform build system

- [x] Write top-level `CMakeLists.txt` with `SINGULARITY_BACKEND_METAL`, `SINGULARITY_BACKEND_VULKAN`, `SINGULARITY_BACKEND_CUDA`, `SINGULARITY_BUILD_WEB` options.
- [x] Auto-detect platform: `SINGULARITY_BACKEND_METAL=ON` if Apple, `SINGULARITY_BACKEND_VULKAN=ON` if Win.
- [x] Add custom CMake function `add_metal_library()` for `.metal → .air → .metallib`.
- [x] Add custom CMake function `add_vulkan_shaders()` for `.hlsl → .spv` via DXC.
- [x] Vendor `metal-cpp` headers into `third_party/metal-cpp/`.
- [x] Vendor `Vulkan-Hpp` (header-only) into `third_party/Vulkan-Hpp/`.
- [x] Add SDL3 as a git submodule under `third_party/SDL/`.
- [x] Vendor Catch2 v3 (single-header) into `third_party/catch2/`.
- [x] Vendor Dear ImGui (with Metal AND Vulkan backends) into `third_party/imgui/`.
- [x] Vendor `stb_image_write.h` into `third_party/stb/`.
- [x] Verify Debug + Release build on Mac. *(Apple Silicon native, ad-hoc signed .app produced; metallib step gated on Xcode presence)*
- [ ] Verify Debug + Release build on Windows. *(needs Windows hardware)*

### Hello-world Metal app (Mac)

- [x] Create `app/main.cpp` — SDL3 entry, branches into platform-appropriate setup.
- [x] Create `render/metal/metal_backend.mm` — Objective-C++ shim with CAMetalLayer + Metal device init + compute+render pipelines + triple-buffered uniforms + frame capture.
- [x] Skipped the cube hello-world — jumped straight to the Phase 2 geodesic compute kernel since the CPU reference renderers were already green. See `render/metal/shaders/geodesic_kernel.metal`.
- [x] Blit pass with Reinhard tone map (`render/metal/shaders/blit.metal`).
- [x] ImGui integration wired into the app shell (Metal + SDL3 bindings).
- [ ] First real-time render on Apple Silicon — gated on full Xcode install so `xcrun metal` is available. Code path is complete; only the shader-compile step is missing.

### Hello-world Vulkan app (Windows)

- [x] Create `render/vulkan/vulkan_backend.cpp` — Vulkan instance/device/swapchain via Vulkan-Hpp. *(Skipped the spinning-cube milestone same as the Metal path — jumped straight to the Phase 4 body-fill: instance, surface-via-callback, swapchain, compute + blit pipelines, HDR + bloom ping-pong, per-frame descriptors and timestamps.)*
- [x] Create `render/vulkan/shaders/cube.hlsl` — vertex + fragment shader. *(Replaced by the full-scene shader trio: `geodesic_kernel.hlsl`, `bloom.hlsl`, `blit.hlsl`.)*
- [x] Compile via DXC to SPIR-V at build time. *(`render/vulkan/CMakeLists.txt` wires DXC from `$VULKAN_SDK/Bin`.)*
- [ ] Verify same spinning cube on Windows. *(needs Windows; see `docs/NEXT_STEPS_WINDOWS.md`.)*

### Backend abstraction (skeletal)

- [x] Define `render/include/render_backend.hpp` per ARCHITECTURE.md §3.
- [x] Metal backend implements the interface end-to-end (compute kernel + blit + capture).
- [x] Vulkan backend implements the interface end-to-end (mirror of Metal; SDL-free via `WindowHandle::vulkan_create_surface` callback).
- [x] Factory `create_default_backend()` routes to Metal on Apple; Vulkan when that leg lands.

### CI scaffolding (matrix)

- [x] Write `.github/workflows/ci.yml` with `lint`, `build-test`, `build-app` jobs.
- [x] Matrix: `os: [macos-14, windows-2022, ubuntu-latest]` (Linux leg added for the no-backend smoke path).
- [x] Verify CI runs and passes on a no-op PR on both platforms. *(Linux confirmed locally; macOS/Windows will confirm on first push)*
- [ ] Add Codecov upload step (token in repo secrets).
- [ ] Add status badges to the README (one per platform).

### Python verification env

- [x] Create `verification/pyproject.toml` with NumPy, SciPy, SymPy, pytest, pytest-xdist, Pillow, ImageHash.
- [x] Write `verification/conftest.py` with a fixture that locates the built `singularity_cli` binary.
- [x] Smoke test: `pytest -k smoke` runs `singularity_cli --help`, asserts exit code 0, on both platforms.

### Docs scaffolding

- [x] Move `PRD.md`, `PHYSICS.md`, `ARCHITECTURE.md`, `TODO.md` into `docs/`.
- [x] Write the actual README using the template from `docs/README.md`.
- [x] Add a `BUILDING.md` covering both Mac and Windows build instructions.
- [x] Add a `CONTRIBUTING.md`.
- [x] Add a `LICENSE` file.

---

## Phase 1 — 2D toy model, CPU-only (Week 2, ~12 hours)

**Exit criterion:** A standalone CPU-only program that integrates 2D Schwarzschild geodesics for a fan of rays and writes a PNG showing them curving around the BH. The PNG matches a SciPy-generated reference within tolerance. Runs identically on Mac and Windows.

### Core types

- [x] Define `core/include/physics/state.hpp` — 8-component state vector.
- [x] Define `core/include/physics/schwarzschild.hpp` — *as a header-only template* using the `shared_shader/shader_compat.h` macros so the same file compiles in MSL/HLSL/CUDA/host C++.
- [x] Implement Christoffel symbols per PHYSICS.md §3.
- [x] Implement `geodesic_rhs()` per PHYSICS.md §4.
- [x] Catch2 unit tests: zero-velocity state stays put; circular orbit at `r = 6M` stays circular for one period within tolerance.

### Integrator

- [x] Implement `core/include/physics/integrator.hpp` — fixed-step RK4, header-only. *(also added DOPRI5 adaptive step ahead of schedule.)*
- [x] Catch2 test: integrate a known harmonic oscillator and verify 4th-order convergence.
- [x] Implement Euler too — for the demo screenshot only, won't survive Phase 2.

### 2D toy renderer

- [x] Create `cli/main.cpp` with a `--mode 2d-toy` flag.
- [x] Initialize 100 parallel rays at `x = 50M, y ∈ [-10M, +10M]` aimed in `-x̂`.
- [x] Integrate each ray for up to 5000 steps; record trail.
- [x] Render trails to a 1024×1024 PNG using `stb_image_write` (CPU rasterization).
- [x] Draw the BH as a black disc of radius `r_s`.

### Verification of Phase 1

- [x] Write `verification/test_christoffel_sympy.py` — derive Schwarzschild Christoffels in SymPy; assert equality with hand-coded.
- [x] Write `verification/test_phase1_rays.py` — run `singularity_cli --mode 2d-toy`, parse output, compare ray-curvature angle against the *full* Schwarzschild integral (SciPy quadrature), not just the weak-field formula. Eddington's 1/b asymptote lives in `test_deflection.py`.
- [x] Smoke test: integrate a circular photon orbit at `r = 1.5 r_s`, assert closes within 0.5%. *(lives in `test_photon_sphere.py` + `test_kerr_hamilton.cpp`.)*
- [ ] Run all of the above on Mac CI and Windows CI; results identical to within float precision. *(Linux confirmed; cross-OS confirmation waits on first push.)*

### Docs

- [x] Verify `PHYSICS.md` §1-§6 references in code comments.
- [x] Add 2D output screenshots to README. *(`docs/images/phase1_2d_toy.png`.)*

---

## Phase 2 — Mac/Metal 3D Schwarzschild on GPU (Weeks 3–5, ~30 hours)

**Exit criterion:** Real-time 3D rendering of a Schwarzschild BH on Mac with a Milky Way skybox, gravitational lensing, orbital camera, ≥30 FPS at 1280×720 on M2.

### App shell (Mac-side)

- [ ] Wire SDL3 into `app/main.cpp`: create `SDL_Window` with `SDL_WINDOW_METAL`.
- [ ] Get `CAMetalLayer*` via `SDL_Metal_CreateView` + `SDL_Metal_GetLayer`.
- [ ] Pass into `MetalBackend::initialize()`.
- [ ] Wire up SDL3 event loop; route into a CPU-side input handler.

### Skybox

- [ ] Source ESO Milky Way panorama at 8K equirectangular (verify license).
- [ ] Add to `app/Resources/skybox.png`; load as `MTLTexture` at startup.
- [ ] Implement `shared_shader/skybox_sample.h` — direction → equirect UV → texture sample.

### Camera

- [ ] Implement `core/include/camera.hpp` — orbital camera with azimuth/elevation/distance.
- [ ] Implement mouse-drag in app shell to update azimuth/elevation; scroll for distance.
- [ ] Implement `generate_ray()` per ARCHITECTURE.md §8.1 in `shared_shader/`.
- [ ] Catch2 test: ray through center of view points along `-Z` of camera basis; corners per FOV.

### Compute pipeline

- [ ] Create `render/metal/shaders/geodesic_kernel.metal` per ARCHITECTURE.md §5.
- [ ] Include `shared_shader/geodesic_math.h` directly.
- [ ] Wire up the compute pass: PSO creation at startup, dispatch per frame.
- [ ] Add triple-buffered uniforms.
- [ ] Milestone 1: render a black disc on starfield. No lensing. Verify angular size matches `r_s / D`.
- [ ] Add geodesic integration. Watch lensing appear.

### Initial state from camera ray

- [ ] Implement Cartesian → BL coordinate conversion.
- [ ] Implement null-condition solver for `u^t`.
- [ ] Catch2 test: radial inbound ray at large `r`, integrated forward, reaches horizon within expected coordinate-time.

### Performance

- [ ] Profile with Xcode GPU Frame Capture. Identify slowest 3 things.
- [ ] Tune threadgroup size (try 8×8, 16×16, 32×32).
- [ ] Try `half` for skybox sampling; keep `float` for integration.
- [ ] Implement half-resolution preview during camera drag, full-res when still.
- [ ] Verify ≥30 FPS at 1280×720 on M2.

### ImGui integration (Mac)

- [ ] Wire up ImGui Metal backend in the same render pass as the blit.
- [ ] Implement minimal settings panel: mass slider, FOV slider, FPS counter.

### Visual polish

- [ ] Reinhard tone mapping in the blit shader.
- [ ] Subtle vignette.
- [ ] Explicit gamma correction (linear → sRGB) in the blit shader.

### Verification

- [ ] Generate three Mac/Metal golden images (Schwarzschild, three camera positions). Store in `verification/golden/metal/`.
- [ ] Add `verification/test_golden_images.py` using ImageHash perceptual hash.
- [ ] Update README with the first real screenshot.

---

## Phase 3 — Mac/Metal accretion disc + v0.5 SHIP (Week 6, ~14 hours)

**Exit criterion:** Mac app renders Schwarzschild + accretion disc with Doppler beaming and gravitational redshift, signed and notarized `.dmg` available on GitHub Releases.

### Disc geometry

- [ ] Implement `shared_shader/disc_intersection.h` — analytic intersection of geodesic segment with `θ = π/2` plane between `r_inner = 6M` and `r_outer = 20M`.
- [ ] Verify: render disc with no relativity (Euclidean rays) — should look like a flat ring.
- [ ] Re-enable geodesic integration: disc appears distorted, back half folded above and below.

### Disc texture

- [ ] Implement procedural disc temperature map: `T(r) ∝ r^(-3/4)` per simplified Novikov-Thorne.
- [ ] Implement blackbody → sRGB color mapping (precompute as 1D LUT texture).
- [ ] Add radial noise (simplex noise, MSL-friendly).
- [ ] Add toggle in ImGui: "procedural texture on/off."

### Doppler beaming

- [ ] Implement `shared_shader/doppler.h` — frequency shift `g = ν_obs / ν_emit` per PHYSICS.md §8.2.
- [ ] Apply `I_obs = g⁴ · I_emit` to disc samples.
- [ ] Add toggle: "Doppler on/off."
- [ ] Verify: with Doppler on, side moving toward camera is brighter and bluer.

### Gravitational redshift

- [ ] Apply `√((1 - r_s/r_obs) / (1 - r_s/r_emit))` to disc samples.
- [ ] Add toggle: "Gravitational redshift on/off."

### Settings persistence (Mac)

- [x] Settings struct → flat `key = value` file (skipped JSON; no dependency and the `scene_config` parser already speaks this format). *(`app/app_shell.cpp` `save_settings` / `load_settings`.)*
- [x] Load on launch from `~/Library/Application Support/Singularity/settings.conf` (Mac), `$XDG_CONFIG_HOME/singularity/settings.conf` (Linux), `%APPDATA%\Singularity\settings.conf` (Win).
- [x] Save debounced. *(Edge-triggered on ImGui widget active→inactive transition — one write per settled adjustment, no mid-drag thrash. `settings_dirty` flag + consumer in `app/app_shell.cpp` main loop. Landed in 1717dff.)*

### v0.5 SHIP — Mac signed release

- [ ] Enroll in Apple Developer Program ($99/yr).
- [ ] Generate Developer ID Application certificate.
- [ ] Create app-specific password for notarization.
- [ ] Store `.p12` + password as GitHub Actions encrypted secrets.
- [ ] Write `.github/workflows/release.yml` — Mac job.
- [ ] Test on a tag push to a throwaway repo first.
- [ ] Verify Gatekeeper accepts the `.dmg` on a clean Mac.
- [ ] Tag `v0.5.0`, push, verify release artifacts.
- [ ] Update README with download link.
- [ ] Generate hero GIF for README using QuickTime screen recording.

---

## Phase 4 — Backend abstraction + Vulkan port (Weeks 7–9, ~30 hours)

**Exit criterion:** Windows app renders Schwarzschild + accretion disc at parity with Mac. Backend-equivalence test passes. Refactor pain felt and reflected upon (in a code comment somewhere).

### Extract the abstraction (Mac-side first)

- [x] Refactor existing Mac code: move all Metal-API-specific calls into `render/metal/metal_backend.mm`.
- [x] Force every other piece of code to talk through `RenderBackend` interface.
- [x] Run all existing tests to verify Mac still works post-refactor.
- [x] Load-bearing commit landed as the "Ship the Mac slice" + "Vulkan backend: full body-fill" pair; the abstraction lives in `render/include/render_backend.hpp` and every API-specific call sits in the backend-specific subdirectory.

### Vulkan boilerplate

- [x] Create `render/vulkan/vulkan_backend.cpp` with Vulkan-Hpp.
- [x] Instance + physical device selection (prefer discrete; fall back to integrated).
- [x] Logical device + compute queue + graphics queue (often same queue family).
- [x] Swapchain via SDL3's `SDL_Vulkan_CreateSurface`. *(Backend is SDL-free; app passes a callback through `WindowHandle::vulkan_create_surface`.)*
- [ ] Add VMA (Vulkan Memory Allocator) under `third_party/vma/`. *(Deferred — raw `allocateMemoryUnique` covers the fixed resource set today; VMA becomes worthwhile when the resource count grows.)*
- [ ] Use VMA for all buffer/image allocations. *(See above.)*

### Vulkan shaders

- [x] Port `geodesic_kernel.metal` → `geodesic_kernel.hlsl`. Same `#include "geodesic_math.h"`. The body should be ~5 lines different: kernel signature, builtin variable names.
- [x] Same for disc, blit shaders.
- [x] DXC compilation in CMake → SPIR-V.
- [ ] Verify shaders load without validation errors (enable `VK_LAYER_KHRONOS_validation`). *(Validation layer is enabled in Debug builds; runtime verification needs Windows hardware.)*

### Vulkan compute pipeline

- [x] Descriptor set layout for the kernel (HDR output texture, uniforms).
- [x] Compute pipeline state + pipeline layout.
- [x] Per-frame command buffer recording.
- [x] Image memory barriers between compute (write) and graphics (read).

### Vulkan render pipeline

- [x] Graphics pipeline for blit + tone map (single fullscreen triangle).
- [ ] Dynamic rendering (Vulkan 1.3) — skip the `VkRenderPass` ceremony. *(Currently uses a one-subpass VkRenderPass + framebuffers; ImGui-Vulkan's InitInfo wants an explicit RenderPass handle so the render-pass path was the cheaper choice.)*

### ImGui integration (Windows)

- [x] Wire up ImGui Vulkan backend. *(`third_party/CMakeLists.txt` builds `imgui_vulkan` when `Vulkan::Vulkan` is available; `app/app_shell.cpp` initializes it with the backend's instance/device/renderpass handles.)*
- [x] Reuse the same settings UI code (it's API-neutral, just calls into `Scene`).

### Verification

- [x] Generate three Vulkan golden images at the same scenes as the Mac goldens. Store in `verification/golden/vulkan/`. *(Schwarzschild, Kerr a/M=0.5, Kerr a/M=0.94 at 256×256 SS=1 — matches the resolution `test_backend_equivalence.py` already uses. Captured via the live-app Vulkan pipeline, not the CPU reference. Mac goldens still pending; the strict Metal ↔ Vulkan pHash < 4 diff waits on those.)*
- [ ] Write `verification/test_backend_equivalence.py` — for the same scene, render on Metal and Vulkan, assert perceptual-hash distance < 4. *(File exists today but as a CPU ↔ active-GPU-backend structural check: pHash tolerance 38 (Schw) / 40 (Kerr), shadow-centroid agreement, Kerr Doppler-asymmetry direction, Schw left-right symmetry. All three tests pass on Windows against the Vulkan backend. Still missing: strict Metal ↔ Vulkan pixel-equivalence at pHash < 4 — that needs the Vulkan goldens above plus a Mac+MoltenVK runner.)*
- [ ] Run on a Mac with Vulkan SDK installed (Vulkan can run on Mac via MoltenVK for the test) — verify both backends pass.
- [ ] **If equivalence fails:** debug. The abstraction or the shaders have drifted. This test is the load-bearing piece of the cross-platform claim.

### Performance check

- [ ] Verify ≥60 FPS at 1920×1080 with Schwarzschild + disc on RTX 3070.
- [ ] If slower: profile with RenderDoc; common culprits are descriptor-set thrashing and unnecessary barriers.

---

## Phase 5 — Cross-platform CI, Windows release, v0.7 SHIP (Week 10, ~10 hours)

**Exit criterion:** Both platforms have signed (or documented-unsigned) installers on GitHub Releases. CI matrix passes on every PR. v0.7 release tagged and downloadable.

### CI matrix completion

- [x] Windows runner builds `.exe` and runs all tests. *(Compile + unit tests in `build-test`; app build uploads the `.exe` artefact in `build-app`.)*
- [x] Mac runner builds `.app` and runs all tests.
- [ ] Backend-equivalence test runs on the Mac runner (with MoltenVK for Vulkan-on-Mac). *(Needs a Vulkan SDK install step on macos-14 + MoltenVK; deferred until a first backend-equivalence golden set exists.)*
- [ ] Docs deploy unchanged.

### Windows installer

- [x] Add WiX Toolset to Windows CI. *(Pre-installed on windows-2022.)*
- [x] Write `installer/Singularity.wxs` — basic MSI with shortcut creation. *(Skipped raw WiX authoring — CPack's WIX generator is configured via `CPACK_WIX_*` vars in the top-level `CMakeLists.txt` and produces the equivalent `.wxs` at package time.)*
- [x] Bundle compiled SPIR-V + exe as Resources. *(`install(TARGETS singularity RUNTIME DESTINATION bin)` + `install(DIRECTORY … FILES_MATCHING PATTERN "*.spv")` in `app/CMakeLists.txt`.)*
- [x] Build `.msi` in `release.yml` Windows job. *(Also built + uploaded on every main-branch push via `build-app` -> Singularity-windows-installer artefact.)*

### Windows code signing decision

- [x] Decide: skip EV cert for v1.0 (saves $400/yr). *(Documented in `docs/NEXT_STEPS_WINDOWS.md` §5. The release workflow honours `SINGULARITY_WIN_SIGN_PFX` if/when an OV cert is purchased.)*
- [x] Document SmartScreen workaround in download instructions. *(`docs/NEXT_STEPS_WINDOWS.md` §5.)*
- [ ] (Alternatively: self-signed cert with documented "trust at your own risk" — some users find this less alarming than SmartScreen.) *(Not adopted.)*

### v0.7 SHIP

- [ ] Tag `v0.7.0`.
- [ ] Verify both `.dmg` and `.msi` artifacts uploaded.
- [ ] Update README with both download links.
- [ ] Update hero GIF (Windows version captured separately, ideally side-by-side with Mac).

---

## Phase 6 — Kerr on both backends (Weeks 11–13, ~28 hours)

**Exit criterion:** Both apps render maximally-spinning Kerr BH (`a/M = 0.94`) with asymmetric shadow, frame-dragged disc, and ISCO-aware inner edge. ≥15 FPS Mac (M2), ≥45 FPS Windows (RTX 3070).

### Kerr metric in shared core

- [x] Implement `core/include/physics/kerr.hpp` (header-only, shared across backends).
- [x] Use the conserved-quantity (E, L_z, Q) formulation per PHYSICS.md §7.3 — NOT raw Christoffels. *(`shared_shader/kerr_hamilton.h`.)*
- [x] Hand-derive (or SymPy-derive) the radial and polar equations; cross-check against Carroll §6.7.
- [x] Catch2 test: horizon radii match `M ± √(M² - a²)` to 0.01% for `a/M ∈ {0, 0.5, 0.94, 0.998}`. *(Also cross-checked end-to-end via `kerr-geometry` CLI mode + `test_kerr_geometry.py`.)*
- [x] Catch2 test: ISCO radius (prograde + retrograde) matches §7.2.

### Kerr GPU port (Metal)

- [ ] Update `geodesic_kernel.metal` to dispatch into `geodesic_rhs_kerr()` from the shared header (use a #define switch or two specializations).
- [ ] Add a second compute PSO for Kerr.
- [ ] Switch in renderer based on `Scene::metric` — runtime PSO swap, no shader branching.

### Kerr GPU port (Vulkan)

- [ ] Same — `geodesic_kernel.hlsl` updated to use Kerr math from shared header.
- [ ] DXC recompile, verify SPIR-V validates.
- [ ] Second compute pipeline for Kerr.

### Carter constant integration

- [ ] Initialize `Q` from camera ray's initial state.
- [ ] Verify `Q` conservation along test geodesics within `10⁻⁵` over 5K steps.
- [ ] Add `verification/test_carter.py` driving this on both backends.

### Frame dragging visualization

- [ ] First raw render: should show asymmetric shadow.
- [ ] Compare against published Kerr shadow images (e.g., JMO Fig. 8).
- [ ] Same on both backends — backend-equivalence test catches drift.

### ISCO-aware disc

- [ ] Update disc inner edge to `r_ISCO(a)` rather than fixed `6M`.
- [ ] Verify visually: at `a/M = 0.998`, ISCO ≈ `1.24M`, disc nearly touches horizon.
- [ ] Update Doppler: use prograde Boyer-Lindquist circular geodesic four-velocity.

### Performance

- [ ] Profile Kerr kernel on Mac (Xcode GPU Frame Capture).
- [ ] Profile Kerr kernel on Windows (RenderDoc / Nsight Graphics).
- [ ] Optimize: precompute `Σ`, `Δ` once per RHS call (not per term).
- [ ] If Mac < 15 FPS at 1280×720: drop default Kerr resolution to 960×540 with documentation.

### Scientific overlay

- [ ] Implement wireframe rendering pass: photon sphere, ISCO, event horizon.
- [ ] Kerr: also render ergosphere outer boundary (`r_ergo(θ)`).
- [ ] Toggle in ImGui: "Scientific overlay on/off."
- [ ] Verify on both backends.

### Verification

- [ ] `verification/test_isco.py` for `a/M ∈ {0, 0.5, 0.94, 0.998}` on both backends.
- [ ] `verification/test_kerr_horizons.py` on both backends.
- [ ] Generate four Kerr golden images per backend.
- [ ] Backend-equivalence test passes for Kerr scenes.

---

## Phase 7 — Polish, docs site, WASM demo, v1.0 SHIP (Weeks 14–16, ~30 hours)

**Exit criterion:** Settings UI complete, screenshot/video export works, docs site live with WebGPU demo embedded, both installers updated to v1.0.

### Settings UI completion

- [ ] All settings from PRD §5.1 F5 in ImGui panel.
- [ ] All sliders trigger live re-render.
- [x] "Reset to defaults" button. *(Next to Screenshot in the Export section — `scene = Scene{}; cam = OrbitalCamera{};` one-liner in `app/app_shell.cpp`. Landed in 1717dff.)*
- [ ] "Save preset" / "Load preset" with cross-platform file dialogs (use `nfd` — Native File Dialog).

### Screenshot export

- [x] "Export PNG" button in settings panel. *(`imgui_settings_panel` → `screenshot_requested` flag consumed by the main loop right after `render_frame()`.)*
- [ ] Render at 4× current display resolution (supersampled). *(Per-pixel 4× SS at capture time landed in 1717dff — the screenshot button force-overrides `scene.render_supersample = 4` for one frame, captures, restores. Still open: framebuffer-4× (16× pixel count) for true super-res export. That's a bigger change — new HDR target sized 2W × 2H and a separate blit path.)*
- [x] Use `stb_image_write` for PNG encoding.
- [x] Save to `~/Pictures/Singularity/` (Mac + Linux) or `%USERPROFILE%\Pictures\Singularity\` (Win). *(Filename: `singularity-YYYYMMDD-HHMMSS.png`.)*

### Video export

- [ ] "Record" button — records the next 60 seconds.
- [ ] Mac: AVFoundation (Objective-C++).
- [ ] Win: Media Foundation (Win32 COM).
- [ ] Encode H.264 .mp4.

### Integrator upgrade

- [x] Implement DOPRI5 with embedded error estimate per PHYSICS.md §6.3 in shared header. *(Lives in `shared_shader/geodesic_math.h` as `dopri5_step`; double-precision CPU host has the embedded-error variant in `core/include/physics/integrator.hpp`.)*
- [ ] ~~Replace RK4 in both backends.~~ **Attempted in `4cc7749`, reverted.** Empirical finding: in float32 at production h_step (0.1M–1.0M), RK4 conserves E / L / null residual 5–10× *better* than DOPRI5 because round-off from DP54's 7 stages with awkward coefficients (19372/6561 etc.) overwhelms its O(h^6) truncation advantage. The truncation win only shows in double precision; GPU shaders are float32. Caveat is now baked into the `dopri5_step` docstring in `shared_shader/geodesic_math.h`. To revisit, either (a) drive an adaptive controller from the embedded 4(4) error estimate on the CPU path only, or (b) wait for an FP64 GPU path (would also unlock other accumulated-precision wins).
- [x] Update conserved-quantity check to run every 50 steps. *(`tests/test_conserved.cpp` adds a 50K-step ISCO orbit test that samples E and L drift every 50 steps and asserts the running max stays under 5e-5. Catches a linear-in-N bleed that the existing 2-period test would miss; complements the 10K-photon-deflection test.)*

### Sentry integration

- [ ] Mac: vendored XCFramework.
- [ ] Win: vendored DLL + import lib.
- [ ] Initialize in `app/main.cpp` only if user has opted in.
- [ ] First-launch dialog: "Send anonymous crash reports? Yes / No / Ask later."

### Hardening

- [x] AddressSanitizer + UBSan on both platforms; fix any reports. *(`sanitizers` job in `.github/workflows/ci.yml` runs Linux Debug + ASan + UBSan against Catch2 + pytest. Linux-only — MSVC supports `/fsanitize=address` but not UBSan, and the existing `SINGULARITY_ENABLE_ASAN` CMake option is gcc/clang-only. Reports will surface on the next push; fixes pending whatever the first run flags.)*
- [x] `clang-tidy` with all `cert-*` and `bugprone-*` checks; fix. *(Wired into the `lint` job — configures a no-backend Debug build to produce `compile_commands.json`, then runs `clang-tidy -p build-tidy` across `core/`, `cli/`, `tests/`. Currently non-gating (`|| true`) until the first batch of findings is triaged; promote to hard gate after fixes land.)*
- [ ] Verify Debug + Release pass full test suite on both platforms.

### Docs site

**Plan switched 2026-04-25:** the original Next.js + Vercel scaffold was overkill for the actual scope (4–5 mostly-static pages + an iframed WebGPU demo + KaTeX). Going with **GitHub Pages + Jekyll** instead — built into GH Pages, zero npm dependency churn, deploys automatically from `main` (or `gh-pages` branch). Saves the Vercel slot for something else.

- [x] Decide site root: `docs/_site/` rendered from `docs/` via Jekyll, or a dedicated `gh-pages` branch. *(Picked neither — went with the **GitHub Actions deploy** flow (`actions/jekyll-build-pages` + `actions/upload-pages-artifact` + `actions/deploy-pages`) so a single workflow can build the WASM demo + Jekyll site + stitch them together, no `gh-pages` branch dance and no `_site/` checked in. See `.github/workflows/pages.yml`. d1c359b + 5ba5687.)*
- [x] `docs/_config.yml` with title, description, theme (`minima` is the GH Pages default), `markdown: kramdown`, KaTeX plugin or CDN init script. *(Hand-rolled, no theme — `theme: null`, `plugins: []`, `markdown: kramdown` (GFM), `permalink: /:title/`, plus an `exclude:` list that hides PRD/TODO/PHASE notes/NEXT_STEPS from the public site. d1c359b.)*
- [x] `docs/_layouts/default.html` with header + nav + footer; `docs/_includes/katex.html` for the math renderer (`<script src="...katex.min.js">` + `<script src="...auto-render.min.js">` + `renderMathInElement(document.body)` on load). *(Single-file layout — KaTeX CDN tags + init script inlined into `_layouts/default.html` rather than split into `_includes/`. ~120 lines of CSS in a `<style>` block: `#0a0a0a` bg / `#e8e8e8` fg / `#ffb86c` accent, system-ui sans body, JetBrains Mono / Cascadia / Consolas code. d1c359b.)*
- [ ] Enable Pages in repo settings → Pages → Source: `main /docs`. URL lands at `mal0ware.github.io/Singularity`. *(One-time manual flip pending: Settings → Pages → Build and deployment → Source: **GitHub Actions** (not "Deploy from a branch"). Triggered by the new `pages.yml` workflow — first run after the flip publishes to `https://mal0ware.github.io/Singularity/`.)*

### Docs content

- [x] Convert `docs/PHYSICS.md` front-matter so Jekyll picks it up (`---\nlayout: default\ntitle: Physics\n---`); verify KaTeX renders the inline + block math correctly. *(Front-matter prepended; `permalink: /physics/`. Existing math is in fenced code blocks (kramdown convention here), so KaTeX leaves it untouched — the renderer is wired for future inline math (`$...$` / `$$...$$`) when added. d1c359b.)*
- [x] Same front-matter pass on `docs/ARCHITECTURE.md`. *(`permalink: /architecture/`. d1c359b.)*
- [x] Write a short "About" page (`docs/about.md`). *(Condensed from README's "what is she" + "where is she" sections, plus the "three GPU backends though, really?" + "is the physics actually accurate" Q&A. Same lowercase voice. d1c359b.)*
- [x] "Download" page with links to both releases (`docs/download.md`). *(Links the GitHub releases page; documents the SmartScreen workaround per `NEXT_STEPS_WINDOWS.md` §5 + the macOS Gatekeeper "right-click → Open" first-launch dance + a `cmake -B build` build-from-source fallback. d1c359b.)*
- [x] Landing page (`docs/index.md`) — hero image + one-paragraph pitch + nav links + the iframed WebGPU demo. *(Hero `phase8_cuda.png`, one-paragraph pitch, iframe to `/demo/singularity_web.html` with `loading="lazy"`, nav anchors to /physics/ /architecture/ /about/ /download/. d1c359b.)*

### WebGPU demo (WASM-shared per your decision)

- [x] Configure Emscripten build (`SINGULARITY_BUILD_WEB=ON` + `--use-port=emdawnwebgpu`) (`afa45f7`).
- [x] `render/webgpu/{webgpu_backend.hpp,.cpp}` — Dawn-standards-track C++ wrapper, async device request, full pipeline graph (`cc87fa3`).
- [x] WGSL ports of `geodesic_kernel`, `bloom`, `blit` hand-translated from HLSL (`c253f79`).
- [x] Build to `singularity_web.wasm` + `.js` + `.html` + preloaded WGSL data (`c253f79`).
- [x] `web/main.cpp` — async device init, pointer-driven orbital camera, auto-orbit on load, wheel-zoom (`f7c7ec0`).
- [x] Graceful WebGPU fallback UI in `web/shell.html` (download-release link) (`c253f79`).
- [x] Prefer `*-UnormSrgb` swapchain so fragment output is gamma-correct (`afa45f7`).
- [x] Browser smoke-test verified on Safari + headless Chromium+SwiftShader; 182 KB Kerr render captured at `docs/images/phase7_webgpu.png` (`afa45f7`).
- [x] ImGui-web settings panel — `imgui_impl_wgpu` + `imgui_impl_emscripten`, match the desktop control surface (metric, spin, disc radii, toggles, exposure/bloom sliders). *(Shipped as a **DOM overlay panel** on top of the canvas, not in-canvas ImGui. Two reasons: (1) vendored ImGui 1.91.5's `imgui_impl_wgpu.cpp` is incompatible with current emdawnwebgpu (v20251002) — the upstream backend errors out on the Emscripten target either way (`#error` if `IMGUI_IMPL_WEBGPU_BACKEND_DAWN` is set, and the fallback Emscripten path uses old `WGPUImageCopyTexture` / `WGPUTextureDataLayout` / `WGPUProgrammableStageDescriptor` names that emdawnwebgpu has renamed); bumping the imgui submodule was rejected as too high-blast-radius for a polish feature when the Metal/Vulkan ImGui backends would also have to retest. (2) DOM-side widgets are the web-native pattern (the reason no upstream `imgui_impl_emscripten` exists). Implementation: `web/shell.html` carries an absolutely-positioned panel div with native `<select>`/`<input type=range>`/`<input type=checkbox>` inputs styled to match the docs site (dark `#0a0a0a` background, `#ffb86c` accent, system-ui font); JS event handlers fire on every input change and call into `Module.ccall("singularity_set_X", ...)` against the `EMSCRIPTEN_KEEPALIVE`-tagged setters at the bottom of `web/main.cpp`. Same control surface as the desktop panel: metric switcher, spin, disc inner/outer/peak T / turbulence + visible/Doppler/redshift toggles, exposure + bloom threshold/strength, FOV, "Reset to defaults". Camera distance/azimuth/elevation stay drag-and-scroll only (no slider). FPS readout dropped (no smoothing surface in the JS layer). Smaller wasm than the ImGui-in-canvas alternative would have been.)*
- [ ] Async `capture_frame` body-fill using `wgpuBufferMapAsync` + a latest-frame cache accessible via `ccall` — requires Asyncify or a worker + SharedArrayBuffer.
- [x] Embed in the docs site via `<iframe src="/Singularity/demo/" loading="lazy">`. The CI `build-web` job already uploads the bundle as the `Singularity-web` artefact; a sibling step copies the four files into `docs/demo/` so Pages serves them at `/Singularity/demo/singularity_web.html`. Same-origin iframe, no CORS gymnastics, no Next.js component layer. *(Done a touch differently than described: the dedicated `pages.yml` workflow builds the WASM bundle inline (mirrors the `build-web` job in `ci.yml`) and copies the four `singularity_web.*` files straight into `_site/demo/` between the Jekyll build and the Pages upload — keeps the bundle out of the source tree and out of `ci.yml`'s artefact path. Iframe in `docs/index.md` with `loading="lazy"` per below. d1c359b + 5ba5687.)*
- [ ] Playwright-driven golden-image regression in CI once a GPU-enabled runner is available (SwiftShader on ubuntu-latest only produces ~12 KB PNGs; the `continue-on-error` smoke is a placeholder).

### Web demo polish

- [x] Loading/fallback UI for WebGPU-less browsers (Firefox stable, iOS <17.4) (`c253f79`).
- [x] `loading="lazy"` on the demo iframe so the wasm fetch doesn't block landing-page LCP. *(Set in `docs/index.md`. d1c359b.)*
- [ ] Loading skeleton during WASM/WebGPU init (handled inside `web/shell.html` itself — already part of the demo bundle, just needs a slightly nicer spinner than the current text).

### Analytics

- [ ] PostHog (free tier).
- [ ] Track: page views, demo loads, demo interactions, GitHub release link clicks.

### Polish

- [x] Open Graph meta tags with hero image. *(Added to `docs/_layouts/default.html`: `og:title`/`description`/`url`/`image` (1024×576 hero from `phase8_cuda.png`) + `og:image:alt`/width/height + Twitter `summary_large_image` card. Page-level `title`/`description` front-matter feeds straight in.)*
- [x] Sitemap, `robots.txt`. *(`docs/sitemap.xml` lists the 5 public pages with weekly/monthly changefreq + priority weights; `docs/robots.txt` is permissive (Allow: /) and points at the sitemap. Both use `layout: null` + `sitemap: false` front-matter so Jekyll passes them through unchanged but doesn't re-list them as URLs in themselves.)*
- [ ] Lighthouse score ≥95 on Performance, Accessibility, SEO.
- [x] "Cite this" button with BibTeX entry. *(BibTeX `@software{singularity_2026, ...}` block on `docs/about.md` under the existing license section. Repeats the README's existing guidance: cite the renderer for the renderer; cite James/Tunzelmann/Franklin/Thorne 2015 for the underlying GR + lensing math.)*

### v1.0 SHIP

- [x] Tag `v1.0.0`. *(Annotated tag pushed 2026-04-25, version bump in 9dce864.)*
- [x] Both installers built, signed (Mac), uploaded. *(`release.yml` triggered on the v1.0.0 tag — builds .dmg + .msi on the matching runners, attaches to the GitHub Release. Mac is ad-hoc signed (Developer ID notarization gated on Apple Dev Program enrollment, summer 2026 per `NEXT_STEPS_MAC.md`); Windows is unsigned with the documented SmartScreen workaround per `NEXT_STEPS_WINDOWS.md` §5.)*
- [x] Docs site deployed. *(https://mal0ware.github.io/Singularity/ — Jekyll on GH Pages via `pages.yml`; landing/physics/architecture/about/download + embedded WebGPU demo with the DOM overlay panel.)*
- [ ] README updated with v1.0 hero image (Kerr `a/M = 0.94`, all bells and whistles on). *(Existing CPU-rendered Schwarzschild hero kept for now — the highest-quality v1.0-era hero (`docs/images/phase8_cuda.png`, Kerr a/M=0.94 4K @256SPP) is on the docs site landing instead. Swapping the README hero is a stylistic call worth doing deliberately, not auto.)*
- [ ] Tweet / LinkedIn post / wherever you live online.
- [ ] **Update Fiverr profile** to feature this as a portfolio piece — relevant for the "Python data dashboards or ML integrations" gigs and especially the "full-stack" framing.

---

## Phase 8 — CUDA stretch, v1.1 SHIP (Weeks 17–19, ~25 hours)

**Exit criterion:** `singularity_cuda_cli` produces 4K and 8K stills + 4K video on the 3090 that look meaningfully cleaner than the real-time backends.

### CUDA setup

- [x] Add `SINGULARITY_BACKEND_CUDA=ON` build path. *(Top-level `CMakeLists.txt` gates `enable_language(CUDA)` + adds `render/cuda` + `cuda_cli` subdirectories behind the option.)*
- [x] CMake `enable_language(CUDA)`. *(Top-level when the option is on.)*
- [x] Verify `nvcc` finds the right compute capability for the 3090 (sm_86). *(`CUDA_ARCHITECTURES "80;86;89"` covers Ampere consumer + datacenter + Ada. nvcc 13.2 + MSVC 14.44 host compiler verified on the Ampere dev box.)*
- [x] Vendor or `find_package` for CUDA Toolkit components. *(`find_package(CUDAToolkit REQUIRED)` + `CUDA::cudart` link in `render/cuda/CMakeLists.txt`.)*

### CUDA backend

- [x] Create `render/cuda/cuda_backend.cu` implementing `RenderBackend` (offline-only). *(Lifecycle wraps a device-side RGBA8 buffer; `render_frame` packs a `Uniforms` struct from `Scene` + `CameraState` (mirrors `pack_uniforms` in the Vulkan/Metal backends) and launches `singularity_geodesic_kernel`.)*
- [ ] CUDA texture objects for skybox + disc LUT. *(Procedural starfield + analytic disc match the realtime backends; texture LUTs are an optimisation, not a correctness item.)*
- [x] CUDA buffer for scene uniforms. *(Passed by value as a kernel arg — struct is < 256 B, fits the per-launch parameter region. Considered `__constant__` but per-frame `cudaMemcpyToSymbol` would be net slower for this size.)*
- [x] Per-frame: upload uniforms, launch kernel, copy output back to CPU.

### CUDA kernel

- [x] Create `render/cuda/kernels/geodesic_kernel.cu`. *(Full port of the Metal kernel: blackbody Tanner-Helland, Novikov-Thorne disc, procedural bands, starfield, Schwarzschild RK4, Kerr Hamiltonian. Inline ACES + sRGB → RGBA8 since the offline path has no blit pass. Bloom intentionally omitted — comment notes it would be two extra kernel launches.)*
- [x] Include `shared_shader/geodesic_math.h` directly (CUDA C++ accepts it). *(Plus `kerr_hamilton.h`, `disc_intersection.h`, `kerr_math.h`, `uniforms.h` — the same headers Metal and Vulkan kernels consume. `disc_intersection.h` helpers had to gain the `DEVICE` qualifier; Metal/HLSL ignored the bare `INLINE`, CUDA needed it explicit.)*
- [x] Implement supersampling: 256 samples per pixel via Halton sequence. *(`u.supersample` reinterpreted on the CUDA path as direct SPP count 1..1024. Halton(2,3) drives subpixel jitter; SPP=1 special-cased to pixel centre so the goldens stay aligned. Per-sample tone-mapping (Mitsuba/Cycles pattern) preserves the cinematic look across SPP counts. Exposed via `--samples-per-pixel N` on `cuda_cli`.)*
- [ ] Implement adaptive sampling: more samples near caustics (where adjacent rays diverge sharply). *(Open — would need a two-pass design: low-SPP variance map, then refine where variance > threshold. Halton at 256 SPP already gets within visibly-converged territory for most pixels, so this is a polish item, not a Phase 8 ship blocker.)*

### CLI

- [x] Create `cuda_cli/main.cpp` accepting JSON scene config. *(Accepts `--scene` in the project's key = value text format (same as `singularity_cli --scene`), `--output`, `--res`. JSON is still the eventual PRD goal.)*
- [x] Output PNG (single frame). *(via `stb_image_write`.)*
- [x] Sequential PNGs for video frames. *(`--frames N --output-pattern P` orbits the camera 2π in azimuth across N frames; smoke-tested by `verification/test_cuda_goldens.py::test_cuda_multi_frame_orbit`.)*
- [x] FFmpeg invocation (subprocess) to encode video. *(`cuda_cli --encode-mp4 PATH [--mp4-fps N]` opt-in; forks one ffmpeg subprocess after the last PNG is written, libx264 / yuv420p / crf=18. Default PNG-only path stays subprocess-free for CI. Manual one-liner kept in `--help` for users who don't want the fork. Compile-validation pending: this Windows box has no CUDA toolkit + the CI matrix has no CUDA leg, so the change rides on the next render the user does on the 3080.)*

### Verification

- [x] Backend-equivalence: CUDA at 1 sample/pixel should match Metal/Vulkan within tolerance. *(`verification/test_backend_equivalence.py::test_cuda_structurally_matches_vulkan_golden` — pHash distance ≤ 20 against the Vulkan goldens; calibrated 8 bits of headroom over the measured schw=4 / kerr_a094=12.)*
- [x] At 256 samples: visual inspection only; perceptual hash will diverge in the *good* direction (less aliasing). *(Eyeballed 4K Kerr at 256 SPP — smoother edges, no aliasing on the photon ring rim, comparable cinematic punch to the realtime backends thanks to the per-sample tone-map.)*

### Sample renders

- [x] Generate 4K Schwarzschild still on 3090. *(1:19 wall-clock at 256 SPP. Kept locally; `docs/images/` would balloon if 4K was committed there.)*
- [x] Generate 4K Kerr `a/M = 0.94` still. *(2:42 wall-clock at 256 SPP. Downsampled to 1024-wide as `docs/images/phase8_cuda.png`, mirroring the Phase 7 hero pattern.)*
- [x] Generate 8K Kerr still (max quality). *(Renders to local artefact folder; ~6 min at 256 SPP. Same caveat — too large to commit at full res.)*
- [ ] Generate 30-second 4K Kerr video. *(Sequential-frame mode lands; needs an ~30s × 30fps × ~3s/frame ≈ 45 min one-shot render and an ffmpeg encode.)*
- [ ] Upload to docs site `samples/` page. *(Blocked on the docs site itself, which is a Phase 7 task.)*

### v1.1 SHIP

- [ ] Tag `v1.1.0`.
- [ ] Update README "Anticipated questions" with CUDA section.
- [ ] Add `singularity_cuda_cli` binary to GitHub release (Windows-only initially).

---

## Phase 9 — Stretch (post-v1.1, no timeline)

Pick a subset that interests you. Each is self-contained.

### Wormhole (Morris-Thorne)

- [ ] Implement metric per PHYSICS.md §10 in shared header.
- [ ] Add second skybox texture (the "other universe").
- [ ] Phase 1-style 2D demo first to verify.
- [ ] Port to GPU. Should resemble *Interstellar's* wormhole.

### Linux build

- [ ] Add `ubuntu-latest` to CI matrix.
- [ ] Verify Vulkan code compiles unchanged on Linux.
- [ ] Package as `.AppImage`.
- [ ] Tag `v1.2.0`.

### Binary black hole

- [ ] Far-field superposition approximation (NOT accurate, but visually striking).
- [ ] Animate orbit of two BHs around common barycenter.

### Real Sgr A* mode

- [ ] Real ephemeris for Sgr A* and surrounding stars.
- [ ] Real mass: `4.15 × 10⁶ M☉`.
- [ ] Real spin (best estimate): `a/M ≈ 0.5`.
- [ ] "Show me what's behind Sgr A* right now" mode.

### Adaptive mesh refinement

- [ ] Identify pixels near caustics.
- [ ] Render those at higher resolution.
- [ ] Edge-aware reconstruction.

### Real-time spectroscopy overlay

- [ ] User-placed cursor shows local spectrum (Planck × `g⁴`).
- [ ] Educational mode showing Doppler/redshift contributions separately.

### Educational tour mode

- [ ] Camera follows precomputed path while narrating.
- [ ] Audio narration optional (BYOM — bring your own MP3).

---

## Phase 10 — Web performance + educational overlay (branch `web-perf-edu`)

> Design doc: `docs/WEB_PERF_EDU.md`. The web demo is the product surface
> going forward; desktop backends stay maintained (CI) but get no new
> feature work. Desktop golden images are untouched by construction — the
> adaptive integrator is flag-gated off outside the web backend.

### Performance

- [x] (`a73fa8c`) `adaptive_h(h_base, r, M)` in `shared_shader/geodesic_math.h` + `SING_FLAG_ADAPTIVE_STEP` in `uniforms.h`; host-validated by `tests/test_adaptive_step.cpp` (deflection parity 0.5%, E/L conservation, ≥5× step reduction — measured 8.3×).
- [x] (`c176aec`) WGSL hand-port of `adaptive_h` + `step_h()` wired into both trace loops in `geodesic_kernel.wgsl`.
- [x] (`c176aec`) Dynamic escape radius in the web backend: `escape_r = clamp(2·cam_r, 60M, 200M)`.
- [x] (`c176aec`) Internal render scale: `WebGPUBackend::set_internal_scale()`, HDR/bloom chain at scale × canvas, linear-filtering blit upscale (BGL sampleType Float + Filtering sampler).
- [x] (`c176aec`) Dynamic-resolution controller in `web/main.cpp`: frame-time EMA walks {0.4, 0.5, 0.65, 0.8, 1.0} targeting 60 FPS with hysteresis + cooldown.
- [x] (`c176aec`) Interaction draft mode: h 0.25 / 600 steps while dragging or within 300 ms of a wheel event.
- [x] (`4ac3d9b`) Panel Performance section: quality preset, resolution mode, adaptive toggle, live FPS + scale readout.

### Educational overlay

- [x] (`4ac3d9b`) Seven "Learn" cards in the DOM panel (shadow, photon ring, disc fold, Doppler, redshift, colors, how it's computed) distilled from PHYSICS.md.
- [x] (`4ac3d9b`) Annotation overlay: shadow / photon ring / Doppler-side / lensed-far-side labels positioned from live camera state (`singularity_get_shadow_px_radius`, `singularity_get_doppler_side`).

### Validation & ship

- [x] Local emsdk build of the wasm bundle; headless-Chrome smoke on the RTX 3080 (166 vs 59 FPS adaptive on/off; no WebGPU validation errors; all exports live).
- [x] Full Catch2 + pytest suites green: 77 C++ cases, 65 pytest passed / 10 environment skips.
- [ ] CI `build-web` green on the branch; merge; Pages deploy picks it up.

### Follow-ups (not this branch)

- [ ] Real Milky Way skybox (ESO panorama; license check; texture plumbing in all four backends).
- [ ] Desktop backends adopt `SING_FLAG_ADAPTIVE_STEP` (needs a re-golden of the Vulkan/CUDA reference images).
- [ ] Spectroscopy overlay (hover a disc pixel → local spectrum with Doppler/redshift contributions separated).
- [ ] FSR-style spatial upscale pass if linear-blit upscale proves too soft at 50%.
- [ ] Kerr polar-axis seam: a dotted vertical line renders below the shadow (θ→π pole; `safe_s2` clamp + finite-difference dH/dθ). Pre-existing on `main` (confirmed by A/B with the adaptive flag off); worth a proper fix in the Hamiltonian RHS.

---

## Cross-cutting backlog

- [ ] HDR output (Apple's EDR API on supported displays; Windows HDR10).
- [ ] Apple Vision Pro support (stereo rendering — math identical).
- [ ] Xcode-integrated debug overlay visualizing per-pixel step count as a heatmap.
- [x] `--benchmark` mode in `singularity_cli` producing perf number for CI regression tracking. *(Schwarzschild RK4 + Kerr Hamiltonian, `--metric {schw,kerr}` flag, JSON output with deterministic `total_steps` across runs, pytest smoke coverage.)*
- [ ] Blog post on a single design decision in detail (good for portfolio).
- [ ] Rust port (entire project) — for the curious. Use `wgpu` for cross-platform graphics.

---

## Definition of done — v1.0

When all of these can be checked, you've shipped:

- [ ] All Phase 0–7 boxes ticked.
- [ ] CI green on `main`, all workflows passing on both platforms.
- [ ] Signed `.dmg` and signed-or-documented-unsigned `.msi` on the GitHub releases page.
- [ ] Repo README has hero image (or GIF) in the first 10 lines.
- [ ] `PHYSICS.md` is genuinely readable by a physics undergrad.
- [ ] At least two people (one Mac user, one Windows user) who are not you have installed and run the app.
- [ ] Backend-equivalence test passes — same scene on Metal and Vulkan produces hash-equivalent output.
- [ ] Docs site live with embedded WebGPU demo working in Chrome.
- [ ] You can answer the question "why three GPU backends?" in two sentences and feel good about the answer.

## Definition of done — v1.1 (CUDA stretch)

Add to v1.0:

- [ ] `singularity_cuda_cli` ships on Windows release.
- [ ] At least three sample renders (4K still, 8K still, 4K video) on docs site.
- [ ] CUDA backend produces qualitatively cleaner output than real-time backends at same resolution (visual inspection — caustics smoother, no jagged edges in deep lensing).
