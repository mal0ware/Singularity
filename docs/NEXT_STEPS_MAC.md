# Singularity — macOS next steps

Status right now: the C++ / Objective-C++ Metal backend, compute kernel,
blit pass, SDL3 + ImGui app shell, CMake bundling, and ad-hoc signing are
all wired up and compile cleanly on Apple Silicon. What's missing is the
Metal shader compiler (`xcrun metal`) — it ships only with full Xcode, not
with Command Line Tools. Until Xcode is installed, the build emits a clear
warning at configure time and produces a runnable-but-broken app that fails
at startup with `default.metallib not found`.

## 1. Finish the Mac path (no cost, one Xcode install)

```bash
# 1. Install Xcode from the App Store (one-time, ~15 GB download).
#    When it's installed, switch xcode-select away from Command Line Tools:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept

# 2. Sanity check — `xcrun metal` should now resolve to a real binary:
xcrun -f metal

# 3. Clean rebuild. The Metal shader compiler probe is cached, so blow it
#    away alongside the build dir:
cd ~/githubstuff/Singularity
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 4. Run:
open build/app/singularity.app
```

You should see a window with a Schwarzschild black hole: central shadow,
bright accretion-disc ring, gravitational lensing lifting the far side of
the disc over the top. ImGui panel on the right lets you drag the camera,
zoom, toggle Doppler / redshift, switch to Kerr (WIP), and tune disc radii.

Keyboard + mouse:
- **Left-drag** — orbit the black hole
- **Scroll** — zoom in/out
- **Esc** — quit

## 2. Run the existing physics tests to confirm nothing regressed

```bash
ctest --test-dir build --output-on-failure
pytest verification/
```

You should see 66 Catch2 cases pass (3,785 assertions) and 58 pytest cases
pass. Those are the ground-truth physics checks — SymPy Christoffel re-
derivation, photon-sphere closure, Eddington 1/b asymptote, Kerr geometry
against published values, blackbody / sRGB pipeline against an independent
implementation.

## 3. When you buy the Apple Developer account this summer

The signing hook is already in place. To switch from ad-hoc to real
notarizable signing, you'll add two pieces:

### a. Build with your Developer ID

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSINGULARITY_MACOS_CODESIGN_IDENTITY="Developer ID Application: <Your Name> (TEAMID)"
cmake --build build -j
```

The `app/CMakeLists.txt` detects the non-empty identity and switches from
`codesign -s -` (ad-hoc) to `codesign -s "<id>" --options runtime
--entitlements app/entitlements.plist --timestamp`. Hardened runtime is
on; entitlements allow Metal's JIT and library validation is disabled (the
two things Metal + ImGui need to boot).

### b. Notarize + staple

After the signed build succeeds:

```bash
# 1. Zip the .app because notarytool needs an archive:
ditto -c -k --keepParent build/app/singularity.app build/Singularity.zip

# 2. Submit to Apple (the first time, set up a keychain profile):
xcrun notarytool store-credentials "ALTOOL_PROFILE" \
    --apple-id "you@example.com" \
    --team-id   "TEAMID" \
    --password  "app-specific-password"

xcrun notarytool submit build/Singularity.zip \
    --keychain-profile "ALTOOL_PROFILE" \
    --wait

# 3. Staple so the notarisation works offline:
xcrun stapler staple build/app/singularity.app
```

### c. Package into a .dmg for distribution (optional)

```bash
brew install create-dmg
create-dmg \
    --volname "Singularity" \
    --window-size 500 300 \
    --icon-size 96 \
    --app-drop-link 350 150 \
    build/Singularity.dmg \
    build/app/singularity.app
```

Release the resulting `.dmg` on the GitHub Releases page. The
`release.yml` workflow referenced in `docs/ARCHITECTURE.md §10.2` is the
place to codify this once the identity is in repo secrets.

## 4. Phases not achievable from this Mac alone

| Phase | What's missing | Where it gets done |
|---|---|---|
| 5 — Vulkan backend + Windows `.msi` | Windows + Visual Studio + WiX | GitHub Actions `windows-2022` runner handles the build; signing needs an EV cert (~$400/yr from Sectigo) if Windows SmartScreen matters. Unsigned `.msi` works today, users just click past a warning. |
| 7 — WebGPU + WASM browser demo | Emscripten (doable on this Mac) | `emcmake cmake -B build-wasm -DSINGULARITY_BUILD_WEB=ON` — not wired yet; next big doable task after Metal. |
| 8 — CUDA offline renderer | NVIDIA GPU | Not on any modern Mac — needs a Windows or Linux box with a 30/40-series. |

## 5. Validation: golden-image diff CPU vs Metal

Once the Metal build is running, the cross-reference to confirm the GPU
math matches CPU:

```bash
# CPU reference (already works):
build/cli/singularity_cli cpu-render --resolution 512x512 \
    -o /tmp/cpu_ref.png

# Metal capture (add a --capture flag to the app in a later patch, or
# write a tiny cli/singularity_render_cli that uses the Metal backend
# headlessly via capture_frame() and writes a PNG).
```

The backend already exposes `RenderBackend::capture_frame()` which does
an HDR→sRGB blit into a shared-storage texture and reads it back to the
CPU. Plumbing this into a small `--capture out.png --scene foo.conf`
CLI would give you the perceptual-hash golden-image test that
`docs/ARCHITECTURE.md §12.3` calls for.

## What got committed today

- `render/metal/metal_backend.hpp` + `metal_backend.mm` — concrete backend
  (device/queue/layer/PSO/triple-buffered uniforms/compute+blit pipeline/
  frame capture).
- `render/metal/shaders/geodesic_kernel.metal` — Schwarzschild ray tracer
  consuming the same `shared_shader/geodesic_math.h` and
  `shared_shader/disc_intersection.h` the CPU path uses.
- `render/metal/shaders/blit.metal` — fullscreen triangle + Reinhard tone
  map.
- `render/metal/shaders/uniforms.h` — host↔GPU struct shared by `.mm` and
  `.metal`.
- `render/src/create_default_backend.cpp` — the factory.
- `app/main.cpp` + `app/app_shell.{hpp,cpp}` — SDL3 + ImGui + orbital
  camera + settings panel.
- `app/Info.plist.in` + `app/entitlements.plist` — bundle metadata.
- `third_party/CMakeLists.txt` — SDL3 + ImGui + ImGui-Metal wiring.
- `third_party/{SDL,imgui,metal-cpp}` — vendored.
- Top-level CMakeLists — auto-select Metal on Apple, auto-enable the app
  when any backend is on, arm64 default via `sysctl` (Rosetta-safe),
  graceful degradation when `xcrun metal` is missing.
- `.github/workflows/ci.yml` — macOS artifact upload, metallib
  sanity check on the macos-14 runner.
