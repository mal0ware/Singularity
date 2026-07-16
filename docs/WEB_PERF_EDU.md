# Web performance + educational overlay — design

> Spec for the `web-perf-edu` branch (2026-07-16). Goal: the WebGPU demo is
> the project's front door and it renders every frame at fixed 960×540 with a
> fixed-step integrator and a 200M escape radius — sky rays burn ~1400 RK4
> steps crossing nearly-flat space. This branch makes the demo hold an
> interactive framerate on modest GPUs and adds an educational layer that
> explains the physics on screen. Desktop backends are intentionally
> untouched (their golden images stay byte-stable); the web demo is the
> product surface going forward.

## Performance

Four independent levers, applied web-first:

1. **Adaptive step size** (`shared_shader/geodesic_math.h`, flag-gated).
   The affine step grows linearly with radius: `h(r) = h_base · clamp(r / 6M,
   1, 40)`. Near the photon sphere (r ≲ 6M) integration is unchanged; at the
   disc's outer regions the step matches the desktop Draft preset; in the far
   field steps reach 40× base. Gated behind a new uniform flag
   `SING_FLAG_ADAPTIVE_STEP` so desktop backends (flag off) render
   byte-identical goldens. Validated on the host build by
   `tests/test_adaptive_step.cpp`: deflection angle of an adaptive-stepped
   photon must agree with a fine fixed-step reference within 0.5%, and E/L
   conservation must hold. The WGSL kernel gets the same helper hand-ported
   (it has no preprocessor; drift risk documented in PHASE7_WEBGPU.md).

2. **Dynamic escape radius** (web backend only). `escape_r = clamp(2 ·
   cam_r, 60M, 200M)` instead of a fixed 200M. Residual deflection beyond
   ~2× the camera radius shifts the (procedural) starfield direction by
   well under a degree.

3. **Internal render scale** (web backend). The geodesic/HDR/bloom chain
   renders at `scale × canvas` resolution; the existing fullscreen blit
   upscales via normalized-UV sampling. The blit sampler switches from
   Nearest/NonFiltering to Linear/Filtering (rgba16float is filterable in
   core WebGPU). The surface/canvas stays at native size, so text and UI
   stay crisp. New method `WebGPUBackend::set_internal_scale(float)`.

4. **Dynamic resolution + interaction quality states** (web app).
   A frame-time EMA controller steps the internal scale through
   {0.4, 0.5, 0.65, 0.8, 1.0} targeting ~60 FPS with hysteresis.
   While the pointer drags (or within 250ms of a wheel event) the scene
   drops to draft integration (h 0.25, 600 steps); idle returns to the
   panel's quality preset. Panel gains a Performance section: quality
   preset (Draft/Balanced/Quality), resolution mode (Auto/100/75/50),
   adaptive-integrator toggle, live FPS + scale readout.

Non-goals this branch: DLSS/FSR (revisit only if the above is
insufficient), frame accumulation (the disc animates, accumulation would
smear), desktop kernel adoption of adaptive stepping (needs re-goldening —
follow-up), real Milky Way skybox (follow-up; licensing + texture plumbing
across all backends).

## Educational layer (web demo)

1. **"What am I looking at?" cards** — collapsible `<details>` sections in
   the DOM panel: the shadow, the photon ring, why the disc appears folded
   over the hole, Doppler beaming, gravitational redshift, what the colors
   mean, and how the renderer works (per-pixel geodesic integration). Copy
   distilled from docs/PHYSICS.md; plain HTML, no new dependencies.
2. **Per-control hints** — one-line explanations under the physics toggles.
3. **Annotation overlay** — optional DOM labels positioned over the canvas:
   shadow center, photon-ring radius, the Doppler-brightened (approaching)
   side, and the lensed far side of the disc. Positions computed from
   exported camera state (`singularity_get_shadow_px_radius()`,
   `singularity_get_doppler_side()`); refreshed on a 250ms timer.

## Validation

- `tests/test_adaptive_step.cpp` (host build of the exact shared header).
- Full existing Catch2 + pytest suites must stay green — desktop goldens
  are untouched by construction (flag off, escape_r unchanged off-web).
- Web bundle built with emsdk; manual smoke in a WebGPU browser: drag,
  zoom, quality/resolution switches, annotations on/off, FPS readout.
- CI `build-web` job remains the merge gate for the wasm bundle.
