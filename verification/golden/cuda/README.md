# CUDA golden renders

Reference images of `singularity_cuda_cli` output at a small,
version-control-friendly resolution. Used by
`verification/test_cuda_goldens.py` (CUDA-vs-CUDA regression, tight
tolerance) and the cross-backend arm of
`verification/test_backend_equivalence.py` (CUDA-vs-Vulkan structural
match, relaxed tolerance).

## Scenes

| File | Metric | Spin `a/M` | Notes |
|---|---|---|---|
| `schw.png` | Schwarzschild | 0 | Symmetric shadow + photon ring + lensed disc halo. |
| `kerr_a05.png` | Kerr | 0.5 | Moderate frame dragging; visible asymmetry onset. |
| `kerr_a094.png` | Kerr | 0.94 | Near-extremal — flat-sided D-shaped shadow, Doppler-brighter prograde side. |

All three render at `--camera-distance 30 --camera-elevation 0.15
--fov-y-deg 60` (the same camera the Vulkan goldens use). Captured
via `singularity_cuda_cli --output ... --res 256x256`, which packs an
ACES + sRGB encode directly into RGBA8 — no separate blit pass and no
bloom. That makes the CUDA goldens slightly brighter than the Vulkan
ones at the same scene.

## Regenerate

From the repo root, with a CUDA build at `build-cuda/`:

```sh
BIN=./build-cuda/cuda_cli/singularity_cuda_cli.exe
$BIN --output verification/golden/cuda/schw.png       --res 256x256
$BIN --output verification/golden/cuda/kerr_a05.png   --res 256x256 --scene scenes/kerr_a05.scene
$BIN --output verification/golden/cuda/kerr_a094.png  --res 256x256 --scene scenes/kerr_a094.scene
```

Need a CUDA-equipped Windows box with the CUDA Toolkit + MSVC host
compiler in the PATH. See `docs/PHASE8_CUDA.md` for the toolchain
setup.

## When to regenerate

- Anything that changes the CUDA kernel's tone-map / exposure / sRGB encode.
- Camera-default change in `cuda_cli/main.cpp`.
- Scene-default change in `render/include/render_backend.hpp` (`Scene` defaults).
- Integrator swap (RK4 → DOPRI5, etc.).

When that happens, rerun the three commands above and commit the new
PNGs alongside the code change so this baseline and the kernel stay
in sync.
