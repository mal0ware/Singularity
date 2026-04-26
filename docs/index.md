---
layout: default
title: Singularity
permalink: /
description: Real-time, physically accurate black-hole renderer with Metal, Vulkan, WebGPU, and CUDA backends sharing one C++ physics core.
---

# Singularity

A real-time, physically accurate black-hole renderer. Each pixel of each frame integrates a null geodesic backwards through curved spacetime: a photon launched from the camera, traced through the Schwarzschild or Kerr metric, until it falls through the event horizon, intersects the accretion disc, or escapes to the celestial sphere. The full geodesic equation runs in the compute kernel. No precomputed lensing tables, no screen-space approximations.

Four GPU backends share a single C++ physics core: Metal on macOS, Vulkan on Windows and Linux, WebGPU in the browser, CUDA for offline supersampled stills.

![Kerr black hole at a/M = 0.94 rendered offline by the CUDA backend at 4K, 256 samples per pixel via Halton(2,3) subpixel jitter, downsampled for the docs hero. Asymmetric photon ring from frame dragging; Doppler-bright on the approaching side; gravitationally redshifted on the receding side.]({{ '/images/phase8_cuda.png' | relative_url }})

## Live demo

<div class="demo-frame">
  <iframe src="{{ '/demo/singularity_web.html' | relative_url }}" loading="lazy" width="100%" height="640" title="Singularity WebGPU demo"></iframe>
</div>

The demo runs the same Hamiltonian Kerr integrator as the desktop backends, compiled to WebAssembly via Emscripten and dispatched against the browser's WebGPU device. Drag the canvas to orbit; scroll to zoom; use the panel on the right to switch metric, change spin, and tune disc and cinematics parameters live.

Requires a WebGPU-capable browser: Chrome or Edge 113+, Safari 17.4+, or Firefox Nightly with `dom.webgpu.enabled`.

## Documentation

- [Physics]({{ '/physics/' | relative_url }}). Schwarzschild and Kerr metrics, Christoffel symbols, the geodesic equation, conserved quantities, Carter's constant, the Novikov-Thorne disc model, Doppler beaming and gravitational redshift, validation strategy with SymPy receipts.
- [Architecture]({{ '/architecture/' | relative_url }}). Backend abstraction, the four GPU pipelines (Metal, Vulkan, WebGPU, CUDA), shader sharing across MSL/HLSL/WGSL/CUDA, ray generation, integrator design, build system, CI/CD, distribution, testing strategy.
- [About]({{ '/about/' | relative_url }}). Project overview, technology stack rationale, physics validation summary, license, citation, acknowledgments.
- [Download]({{ '/download/' | relative_url }}). Pre-built `.dmg` and `.msi` installers, build-from-source instructions, signing notes.

Source: [github.com/mal0ware/Singularity](https://github.com/mal0ware/Singularity).
