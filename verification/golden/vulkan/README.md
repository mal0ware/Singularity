# Vulkan golden renders

Reference images of `singularity` (Vulkan backend) output at a small,
version-control-friendly resolution. Used by `verification/test_backend_equivalence.py`
once the paired Metal side of the same scenes is committed.

**These are not the Metal ↔ Vulkan equivalence baseline on their own**
— that test needs both sides committed and diffs them at pHash <
tolerance. This directory holds the Vulkan half. `verification/golden/metal/`
gets the Mac half when a Mac session produces matching captures.

## Scenes

| File | Metric | Spin `a/M` | Notes |
|---|---|---|---|
| `schw.png` | Schwarzschild | 0 | Symmetric shadow + photon ring + lensed disc halo. |
| `kerr_a05.png` | Kerr | 0.5 | Moderate frame dragging; visible asymmetry onset. |
| `kerr_a094.png` | Kerr | 0.94 | Near-extremal — flat-sided D-shaped shadow, Doppler-brighter prograde side. |

All three render at camera distance = 30 M, elevation = 0.15 rad, FOV
defaults. Captured via the live-app headless `--capture` path (not via
`singularity_cli --mode cpu-render`) so they exercise the real Vulkan
compute + blit + HDR + bloom + ACES pipeline.

## Regenerate

From the repo root, with a Release build in `build/`:

```powershell
# Windows
.\build\app\singularity.exe --capture=verification/golden/vulkan/schw.png       --capture-ss=1 --res=256x256
.\build\app\singularity.exe --capture=verification/golden/vulkan/kerr_a05.png   --capture-ss=1 --res=256x256 --kerr --spin=0.5
.\build\app\singularity.exe --capture=verification/golden/vulkan/kerr_a094.png  --capture-ss=1 --res=256x256 --kerr --spin=0.94
```

```sh
# macOS / Linux (with MoltenVK or native Vulkan and SINGULARITY_BACKEND_VULKAN=ON):
./build/app/singularity --capture=verification/golden/vulkan/schw.png       --capture-ss=1 --res=256x256
./build/app/singularity --capture=verification/golden/vulkan/kerr_a05.png   --capture-ss=1 --res=256x256 --kerr --spin=0.5
./build/app/singularity --capture=verification/golden/vulkan/kerr_a094.png  --capture-ss=1 --res=256x256 --kerr --spin=0.94
```

256×256 + SS=1 is the same resolution + supersample
`verification/test_backend_equivalence.py` already uses, so the
eventual strict equivalence check (pHash distance < 4, per Phase 4
exit criterion) can diff directly against these PNGs without
re-rendering.

## When to regenerate

- Shader change that alters tone-map, bloom, or ACES constants.
- Camera-default change (distance, elevation, FOV).
- Scene-default change (disc radii, Doppler / redshift toggles).
- Anything else that shifts the cinematic pipeline output intentionally.

When that happens: rerun the three commands above and commit the new
PNGs alongside the code change so the golden dataset and the code
stay in sync. Perceptual-hash tolerance on the equivalence test gives
some headroom, but large ACES / bloom tweaks will blow past it.

## Provenance + compiler variance

These PNGs were captured on Windows 11 + NVIDIA RTX with the app
built under **MinGW-w64 GCC 16** (portable toolchain, no MSVC
locally). The top-level `CMakeLists.txt` ships `-ffast-math` on
GCC/Clang and `/fp:fast` on MSVC, so host-side float computation
(camera basis, uniform marshalling) can reorder across compilers.
Shader output is pure-GPU and identical given identical uniforms,
so the cross-compiler delta is small in practice — but if an
MSVC-built binary trips `test_vulkan_goldens.py` with a pHash
distance of 5–8, suspect FP reordering before a real regression
and regenerate the goldens from the MSVC binary to set a new
baseline.
