# Product Requirements Document — Singularity

**Project codename:** Singularity
**Owner:** Mal
**Last updated:** April 2026
**Status:** Draft v2.0 (cross-platform revision)
**Companion docs:** `PHYSICS.md`, `ARCHITECTURE.md`, `TODO.md`

---

## 1. Vision

Build a real-time, physically-accurate simulator of Schwarzschild and Kerr black holes that ships as a native macOS app (Metal), a native Windows app (Vulkan), and a browser demo (WebGPU + WebAssembly), with a stretch CUDA path for offline high-resolution rendering on NVIDIA hardware. Demonstrate, through code and a public docs site, a graduate-physics-level grasp of general relativity *and* the systems-engineering chops to maintain one C++ physics core across three GPU APIs and three distribution channels.

The deliverable is two signed installers (`.dmg` + `.msi`), one public docs site with embedded interactive demo, optional CUDA offline-renderer binary, and one GitHub repository whose commit history, test suite, and CI pipeline read like the work of a junior graphics engineer rather than a student project.

## 2. Goals

### 2.1 Primary goals (must-have for v1.0)

1. **Native macOS application** rendering Schwarzschild and Kerr black holes in real time at ≥30 FPS at 1280×720 on M2-class hardware, with orbital camera, configurable mass and spin, and a relativistic accretion disc.
2. **Native Windows application** with feature parity to macOS, hitting ≥60 FPS at 1920×1080 on RTX 3070-class hardware.
3. **A backend abstraction layer** in C++ that lets one physics core drive both Metal and Vulkan implementations without per-backend code in the physics layer.
4. **A public docs site** with the physics derivations as MDX, KaTeX-rendered math, and an embedded WebGPU + WebAssembly 2D solver that shares the C++ physics core via Emscripten.
5. **A test and verification harness** that proves the simulator's geodesics agree with analytical solutions to stated tolerances, on both backends.
6. **Reproducible build pipelines** producing signed `.dmg` (notarized) and Windows `.msi` (signed if EV cert acquired, unsigned with documented SmartScreen warning otherwise) from a tagged git commit.

### 2.2 Secondary goals (should-have)

7. **CUDA offline renderer** (`singularity_cuda_cli`) producing 4K and 8K supersampled stills and 4K video on NVIDIA hardware, leveraging the perf headroom of an RTX 3090 / 4080+ for sample counts the real-time backends can't hit.
8. Wormhole rendering (Morris-Thorne metric, Phase 9 stretch) on at least one backend.
9. Screenshot and video export from inside both real-time apps.
10. A short technical writeup on the docs site that could be linked from a resume.

### 2.3 Non-goals (explicit)

- **Not** a CRUD app — no user accounts, no Postgres, no Redis, no Stripe. The prior conversations about SaaS infrastructure do not apply here.
- **Not** Linux-supported on day one. The Vulkan backend should compile on Linux without changes; testing and packaging it is a Phase 8 stretch.
- **Not** a physics engine for arbitrary spacetimes. Only Schwarzschild, Kerr, and (stretch) Morris-Thorne wormhole.
- **Not** scientifically novel. This is a high-quality re-implementation of well-understood physics.
- **Not** a particle-based accretion disc simulator. The disc is rendered as a continuous emissive medium with analytically-prescribed temperature and orbital velocity. Particle physics would be a different project.

### 2.4 Intermediate "ship-able" milestones

The plan is staged so that if scope or time forces an early stop, what's been built is still defensible:

- **v0.5 — Mac-only Schwarzschild (end of Phase 3).** A complete signed macOS app rendering Schwarzschild + accretion disc. If everything else falls through, this alone is a strong portfolio piece.
- **v0.7 — Cross-platform Schwarzschild (end of Phase 5).** Both backends working, both platforms shipping. Kerr deferred.
- **v1.0 — Full cross-platform Kerr + web demo (end of Phase 7).** All primary goals met.
- **v1.1 — CUDA stretch (end of Phase 8).** Secondary goal #7.
- **v1.2 — Wormholes / additional spacetimes (Phase 9+).**

## 3. Success criteria

| # | Criterion | How we measure |
|---|---|---|
| S1 | Real-time Schwarzschild on Mac | ≥30 FPS at 1280×720 on M2 base with disc enabled |
| S2 | Real-time Schwarzschild on Windows | ≥60 FPS at 1920×1080 on RTX 3070 with disc |
| S3 | Real-time Kerr on Mac | ≥15 FPS at 1280×720 on M2; ≥30 FPS on M3 Pro |
| S4 | Real-time Kerr on Windows | ≥45 FPS at 1920×1080 on RTX 3070 |
| S5 | Physical accuracy | Photon sphere within 0.5% of `1.5 r_s`; weak-field deflection within 1% of `4GM/(bc²)`; checked on both backends |
| S6 | Backend-equivalence | A given scene renders to identical perceptual hash (within tolerance 4) on Metal and Vulkan |
| S7 | Distributable builds | One `make release` (or equivalent) produces signed/notarized `.dmg` and signed `.msi` (or unsigned with documented SmartScreen path) |
| S8 | Documented physics | `PHYSICS.md` and the docs site cover metric → Christoffel → geodesic → integrator → observation, reviewable by a physics undergrad |
| S9 | Test coverage | C++ physics core ≥80% line coverage; ten or more golden-image tests per backend |
| S10 | Portfolio impact | Project is the lead item on the GitHub profile within two weeks of v1.0 |

## 4. Personas & use cases

### 4.1 The recruiter / engineer reviewing the GitHub repo

- Lands on the README, sees a screenshot or autoplaying GIF.
- Notices the multi-backend architecture in the doc index, infers the abstraction effort.
- Skims `PHYSICS.md`, sees actual derivations rather than copy-paste.
- Clicks the docs site, plays with the embedded demo for 30 seconds, closes the tab.
- **Implies:** README leads with a visual; backend abstraction is visible from the first navigation pass; docs site demo loads under 3 seconds.

### 4.2 You, three months from now

- Returns to add the CUDA backend or wormhole support.
- Needs to find where ray generation happens, where Christoffel symbols live, why integration is RK4 not Euler.
- **Implies:** Code references `PHYSICS.md` section numbers in comments; backend interface is documented in `ARCHITECTURE.md`; adding a new backend is a recipe, not an archaeology project.

### 4.3 The curious physics-literate visitor

- Wants to understand why the *Interstellar* accretion disc looks symmetric (it shouldn't).
- Reads docs, follows the Doppler-beaming derivation, toggles the "Doppler off" checkbox in the demo.
- **Implies:** Every visual feature has a corresponding "off" toggle in the UI for pedagogy.

### 4.4 The graphics engineer comparing your two backends

- Curious how a metal-cpp idiom maps to its Vulkan-Hpp equivalent.
- Wants to see the abstraction interface; appreciates if it's not over-engineered.
- **Implies:** `RenderBackend` is a small, well-commented header. No factory factories.

## 5. Functional requirements

### 5.1 Simulator core (C++ shared across backends)

- **F1.** Render a Schwarzschild black hole with configurable mass `M` (in solar masses) and ray-marched gravitational lensing of a celestial-sphere skybox.
- **F2.** Render a Kerr black hole with configurable spin `a/M ∈ [0, 0.998]`, exhibiting the asymmetric shadow and frame dragging.
- **F3.** Render a thin accretion disc with configurable inner and outer radii (defaults to ISCO and `20M`), with three independent toggles:
  - **F3a.** Doppler beaming on/off
  - **F3b.** Gravitational redshift on/off
  - **F3c.** Procedural temperature gradient on/off
- **F4.** Camera with orbital controls (mouse-drag rotate, scroll zoom, optional WASD pan) and configurable field of view.
- **F5.** Settings panel exposing: black hole mass, spin, observer distance, FOV, disc inner/outer radius, disc tilt, integrator step count, integrator step size, render backend (where multiple are compiled in).
- **F6.** Screenshot export (`.png`) and video export (`.mp4`, ≤60 seconds) from the running app.
- **F7.** A "scientific mode" overlay drawing photon sphere, ISCO, event horizon, and ergosphere (Kerr only) as wireframe rings on top of the rendered scene.
- **F8.** Stretch (Phase 9): Morris-Thorne wormhole mode.

### 5.2 Per-platform requirements

- **F9.** macOS app shell using SDL3 + Dear ImGui, native Metal backend via metal-cpp, target macOS 14+.
- **F10.** Windows app shell using SDL3 + Dear ImGui, native Vulkan backend via Vulkan-Hpp, target Windows 11+.
- **F11.** A user can switch the active render backend at runtime if more than one is compiled into the same binary (default: only the platform-native backend ships in the released installers; cross-backend builds are a developer-only convenience).

### 5.3 Verification harness (Python, cross-platform)

- **F12.** Compute analytical photon sphere radius for given `M`, compare to numerically integrated circular photon orbit (run on both backends), assert error <0.5%.
- **F13.** Compute weak-field deflection angle for a ray with impact parameter `b ≫ r_s`, compare to `4GM/(bc²)`, assert error <1%.
- **F14.** For the Kerr case, verify ISCO radius matches the closed-form expression as a function of `a/M`.
- **F15.** Generate ten "golden images" per backend — small (256×256) reference renders — that the test suite can diff against perceptual-hash tolerance.
- **F16.** Backend-equivalence test: same scene must produce hash-equivalent output on Metal and Vulkan (tolerance 4).

### 5.4 Documentation site (Next.js / TypeScript)

- **F17.** Landing page with hero animation, project description, "Try the demo" and "Read the docs" CTAs, GitHub repo link, download links for both installers.
- **F18.** Docs section rendered from MDX, mirroring `PHYSICS.md` and `ARCHITECTURE.md` with KaTeX math rendering.
- **F19.** Interactive 2D demo (the Phase 1 simulator, compiled to WASM via Emscripten, rendered with WebGPU) with the same Doppler/lensing/disc toggles as the native apps.
- **F20.** Hosted on Vercel; auto-deploys from `main`; preview deploys on pull requests.

### 5.5 CUDA stretch backend

- **F21.** `singularity_cuda_cli` — headless binary accepting a scene-config JSON and rendering at arbitrary resolution (default 4K, max 8K).
- **F22.** Sample budget per pixel configurable up to 256 (vs ~1 in the real-time backends), enabling supersampled antialiasing and smoother caustics.
- **F23.** Output to `.png` (stills) or sequential frames + ffmpeg-encoded `.mp4` (video).

### 5.6 App shell & UX (both platforms)

- **F24.** Settings persist in platform-appropriate locations:
  - macOS: `~/Library/Application Support/Singularity/settings.json`
  - Windows: `%APPDATA%\Singularity\settings.json`
- **F25.** Crash reporting via Sentry's native SDK on both platforms (opt-in on first launch).
- **F26.** Dark mode only — this is a black hole simulator; light mode would be perverse.

## 6. Non-functional requirements

| # | Category | Requirement |
|---|---|---|
| N1 | Performance — Mac | ≥30 FPS at 1280×720 with Schwarzschild + disc on M2 base; ≥60 FPS at 1920×1080 on M3 Pro+ |
| N2 | Performance — Mac Kerr | ≥15 FPS at 1280×720 with maximally-spinning Kerr + disc on M2 base |
| N3 | Performance — Win | ≥60 FPS at 1920×1080 Schwarzschild + disc on RTX 3070; ≥120 FPS on RTX 3090 |
| N4 | Performance — Win Kerr | ≥45 FPS at 1920×1080 Kerr + disc on RTX 3070 |
| N5 | Memory | <512 MB resident at idle, <1 GB peak (both platforms) |
| N6 | Startup | Cold launch to first rendered frame: <2s (both platforms); warm launch: <500ms |
| N7 | Code quality | C++20, `-Wall -Wextra -Wpedantic`, `clang-tidy` clean, `clang-format`-formatted, no warnings in CI on either platform |
| N8 | Test coverage | ≥80% line coverage on the platform-agnostic C++ physics core |
| N9 | Build reproducibility | Tagged git commit → identical installer byte-for-byte on the same machine (modulo signing timestamps) |
| N10 | License | MIT for code, CC-BY-SA for docs and renders |
| N11 | Accessibility | All controls keyboard-navigable; WCAG AA contrast on the docs site |
| N12 | Privacy | No telemetry by default; Sentry crash reporting requires explicit opt-in |
| N13 | Backend equivalence | Identical (within perceptual-hash tolerance 4) output between Metal and Vulkan on the same scene |

## 7. Tech stack — with honest justification

### 7.1 Included

| Layer | Tech | Why it earns its place |
|---|---|---|
| **Physics core** | C++20 | Matches source material; matches existing skill; right level for GPU interop on all three backends |
| **Mac graphics** | Metal + metal-cpp | Native macOS API; metal-cpp (Apple, 2022) avoids Objective-C++; better dev tools (Xcode GPU Frame Capture) than Vulkan-on-Mac via MoltenVK |
| **Mac shaders** | MSL (Metal Shading Language) | Required for Metal; C++14-flavored, low cognitive load |
| **Win graphics** | Vulkan + Vulkan-Hpp | Cross-industry standard; better generic graphics signal than DX12; Vulkan-Hpp gives RAII and exception-safe resource handling |
| **Win shaders** | HLSL → DXC → SPIR-V | HLSL is closer to C++ than GLSL; DXC's SPIR-V path lets HLSL drive Vulkan |
| **CUDA backend (stretch)** | CUDA C++ + nvcc | Industry standard for NVIDIA-specific compute; valuable for ML/scientific-compute roles, complementing existing data-pipeline experience |
| **Window/event** | SDL3 | Cross-platform window + input; native Metal layer support on Mac, Vulkan surface support on Windows; stable since 2024 |
| **UI** | Dear ImGui | Cross-platform overlay UI that integrates directly into Metal and Vulkan render passes |
| **Build** | CMake 3.27+ | Industry standard cross-platform C++ build; integrates with Xcode and Visual Studio |
| **Tests** | Catch2 v3 (C++) + pytest (Python) | Header-only, modern; pytest standard for Python verification |
| **Verification math** | Python 3.11 + NumPy + SciPy + SymPy | Right tool for analytical re-derivation of Christoffel symbols and reference integration |
| **CI** | GitHub Actions | Free for public repos; both `macos-14` and `windows-2022` runners available; matrix builds |
| **Crash reporting** | Sentry native SDK (both platforms) | Real production-engineering signal |
| **Docs site** | Next.js 15 + TypeScript + Tailwind + MDX + KaTeX | Standard 2026 docs-site stack |
| **Web demo** | WebGPU + WGSL, with C++ physics core via Emscripten → WASM | The cross-stack move: same physics, three render targets (Metal, Vulkan, WebGPU) |
| **Hosting** | Vercel | Free tier covers a portfolio docs site indefinitely |

### 7.2 Considered and rejected

| Tech | Why it was rejected |
|---|---|
| **Postgres / SQLite / any DB** | No user data; settings are a single JSON file |
| **Redis** | No caching, no pub/sub — nothing to put in a cache |
| **WebSockets** | Single-process desktop app; no client/server boundary |
| **Stripe** | Not a SaaS |
| **Docker** | Native graphics apps don't run in Linux containers; the docs site doesn't need it |
| **Kubernetes / Terraform** | One static site on Vercel; nothing to orchestrate |
| **OpenGL** | Deprecated on macOS since 10.14; would teach a dead API |
| **MoltenVK only (no native Metal)** | 10-20% perf overhead; "native Metal backend" is its own resume bullet |
| **DirectX 12** | Signal mostly relevant to MS/Xbox; Vulkan covers a wider job market |
| **Swift / SwiftUI** | Mac-only; metal-cpp removes the historical reason to use Swift for Metal interop |
| **Native UI per platform (AppKit + WinUI)** | Two UI codebases for marginal polish gain at v1.0 |
| **Slang shading language** | Would let one source compile to all backends, but adds a learning curve and replaces three resume bullets (MSL, HLSL, CUDA) with one (Slang) — wrong tradeoff |
| **Rust + wgpu** | Loses CUDA path entirely; adds a new language in a project meant to deepen graphics knowledge, not learn a language |
| **Blockchain** | (To be explicit, since you asked last time.) No. |

### 7.3 The honest meta-point

A recruiter at a graphics-leaning role will be *more* impressed that you didn't reach for Postgres. The "modern stack" surface area for a native graphics simulator is: GPU APIs, shading languages, build systems, CI matrices, code signing, and cross-platform abstractions. We have all of those, legitimately.

## 8. Architecture overview

(Detailed in `ARCHITECTURE.md`; one-paragraph summary here.)

The simulator is a C++ app shell built on SDL3 for windowing and Dear ImGui for UI. It owns one instance of an abstract `RenderBackend`, instantiated as either `MetalBackend` or `VulkanBackend` at startup based on platform and build configuration. Each backend owns its API's device + command queue + pipeline states, but the per-frame logic — camera ray generation, integrator parameters, scene state — flows through the backend interface in API-neutral terms (`Texture*`, `ComputeKernel*`, `RenderTarget*`). The physics math (Christoffel symbols, geodesic RHS, conserved quantities) lives in shared C++ headers that compile inside MSL kernels (Mac), HLSL kernels (Vulkan via DXC→SPIR-V), and CUDA kernels (stretch backend) without modification. The Python verification harness invokes a headless backend through a small CLI binary. The web demo compiles the physics headers to WASM via Emscripten and dispatches them from a TypeScript + WebGPU host.

## 9. Milestones

| Phase | Weeks | Goal | Deliverable |
|---|---|---|---|
| 0 — Setup | 1 | Cross-platform repo, CMake, CI on both platforms, hello-triangle on each backend | Spinning textured cube, on Mac and Windows |
| 1 — 2D toy model | 2 | CPU-only 2D Schwarzschild geodesic | 2D PNG of curved light rays |
| 2 — Mac/Metal 3D Schwarzschild | 3-5 | Real-time Metal compute kernel for lensing | Mac-only video of starfield distorting around BH |
| 3 — Mac/Metal accretion disc + **v0.5 ship** | 6 | Disc + Doppler + redshift; signed Mac .dmg | First public release: macOS-only Schwarzschild |
| 4 — Backend abstraction + Vulkan port | 7-9 | Extract `RenderBackend` interface; port Mac code to Vulkan; both backends pass golden-image tests | Windows binary at Schwarzschild parity |
| 5 — Cross-platform CI + **v0.7 ship** | 10 | Win signed installer; matrix CI; cross-platform release process | Both installers, Schwarzschild only |
| 6 — Kerr on both backends | 11-13 | Add Kerr metric to shared core; one specialization per backend; tests pass on both | Both apps render Kerr |
| 7 — Polish + docs site + WASM demo + **v1.0 ship** | 14-16 | Settings persistence, screenshot/video export, docs site live, web demo embedded | v1.0 |
| 8 — CUDA stretch | 17-19 | `singularity_cuda_cli` for offline 4K/8K rendering on the 3090 | CUDA binary + sample 8K stills |
| 9 — Wormhole + Linux + further stretches | post-v1.1 | Morris-Thorne wormhole; Linux build; ephemeris-driven Sgr A* mode | TBD |

**Total realistic timeline for v1.0:** 16-18 weeks at ~10 hours/week. Add ~3 weeks for CUDA stretch. Studies + Fiverr will move these dates; the v0.5 and v0.7 ship gates exist precisely so an early stop still produces something defensible.

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Backend abstraction designed wrong; Vulkan port reveals leaky interface | High | High | Extract abstraction *after* Metal works (not before); be willing to refactor the interface twice |
| Metal and Vulkan produce subtly different output | Medium | Medium | Backend-equivalence test (F16) catches this; tolerance is perceptual-hash 4, not pixel-exact |
| HLSL→DXC→SPIR-V toolchain rough on Windows CI | Medium | Medium | Pin DXC to a known-good version; document in `BUILDING.md` |
| Numerical instability of geodesic integrator near horizon | High | Medium | Switch to conserved-quantity (Hamiltonian) formulation in Phase 6; tortoise coordinate near horizon |
| Code signing friction on either platform (Apple Dev Program $99/yr; Windows EV cert $400/yr) | Medium | Low | Ship unsigned for personal use until v0.5 (Mac) and v0.7 (Win); buy Apple cert at Phase 3, defer EV cert indefinitely |
| Scope creep into Kerr-Newman, Reissner-Nordström | High | Medium | Hard "no" until v1.0 ships; add to `STRETCH.md` |
| Time dilation / redshift formulas implemented wrong, looks fine but is wrong | Medium | Medium | Python verification harness catches it; never ship a feature without a test |
| metal-cpp API surface small / docs sparse | Medium | Low | Apple's [LearnMetalCPP](https://developer.apple.com/metal/cpp/) sample is the reference; budget extra time in Phase 0 |
| WebGPU browser support gaps in 2026 | Low | Low | Provide WebGL2 fallback render path or accept Chrome/Edge-only |
| 12+ week scope while juggling Fiverr + studies | High | High | v0.5, v0.7, v1.0 ship gates are independently shippable; treat each phase boundary as "if I stop here, do I still have something good?" |
| CUDA backend never gets started because v1.0 takes longer than expected | Medium | Low | CUDA is explicitly a stretch goal; v1.0 success doesn't require it |

## 11. Open questions

These need resolution before or during the relevant phase. None block starting Phase 0.

1. **Skybox texture license.** The ESO Milky Way panorama (CC-BY 4.0) is the obvious choice; verify current attribution.
2. **Apple Developer Program membership timing.** Defer purchase until end of Phase 3 (week 6) to avoid paying for unused months.
3. **Windows signing strategy.** Skip the EV cert for v1.0. Document SmartScreen warning workaround in download instructions. Revisit only if user feedback demands it.
4. **Spin-axis convention for Kerr.** Pick one (`a > 0` co-rotating with `+z` recommended) at start of Phase 6 and document in `PHYSICS.md`.
5. **WASM demo scope.** Phase 1's 2D model is the obvious target. Could the 3D Schwarzschild kernel also run in the browser? Probably yes given WebGPU compute, but bandwidth and battery costs may make it inappropriate as a default. Decide at start of Phase 7.
6. **Should the Linux build be a Phase 9 stretch or a Phase 5 freebie?** The Vulkan code should compile on Linux as-is. The cost is mostly CI runner setup and one tested package format (`.AppImage`?). Reassess at end of Phase 5.

## 12. Appendix — what success looks like

**v0.5 success:** A signed macOS `.dmg` on the GitHub releases page that, when installed on a clean Mac, opens to a real-time render of a Schwarzschild black hole with accretion disc and orbital camera. README has a hero GIF.

**v1.0 success:** The README's hero image is a Kerr black hole with `a/M = 0.94`, viewed from 15° above the equatorial plane, disc from ISCO to `20M`, Doppler beaming on, gravitational redshift on, FOV 60°, 1920×1080. Below the hero, there are download links for both installers. Both installers, when installed on a clean Mac and a clean Windows machine respectively, produce visually-equivalent output. The docs site is live with the embedded WebGPU demo working in Chrome and Safari Tech Preview. If the hero image looks like Gargantua but with the asymmetry *Interstellar* deliberately omitted, you've shipped.

**v1.1 success (CUDA):** A `samples/` folder on the docs site containing 4K and 8K supersampled stills generated by the CUDA renderer that look qualitatively cleaner than what the real-time backends produce — proof that the stretch backend earns its place and that the 3090's compute headroom is being used.
