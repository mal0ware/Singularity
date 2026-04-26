# Building Singularity

> Cross-platform build instructions for the native macOS (Metal) and Windows (Vulkan) apps, plus the optional CUDA offline renderer, the Python verification harness, and the web docs site.

**Minimum tool versions:**

| Tool | Min version | Notes |
|---|---|---|
| CMake | 3.27 | Required for `find_package(Vulkan)` behavior + Metal language support |
| C++ compiler | Apple Clang 15 / MSVC 19.39 (VS 2022 17.9) | C++20 support |
| Python | 3.11 | Verification harness only |
| Node.js | 20 | Web docs site only |

---

## macOS

### Prerequisites

1. **Xcode 15 or later** (App Store or the direct command-line tools download).
   ```sh
   xcode-select --install
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```
   The Metal compiler (`xcrun -sdk macosx metal`) ships with Xcode.

2. **Homebrew + CMake.**
   ```sh
   brew install cmake ninja
   ```

3. **(Optional) Vulkan SDK** — only needed if you want to build the Vulkan backend on Mac via MoltenVK for the cross-backend equivalence test.
   Download from <https://vulkan.lunarg.com/sdk/home#mac>.

### Configure and build

For a quick smoke test of the current Mac build (and a recipe for moving from
ad-hoc signing to a notarized Developer ID build later), see
[`docs/NEXT_STEPS_MAC.md`](docs/NEXT_STEPS_MAC.md).

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
open build/app/singularity.app
```

The Metal backend is enabled automatically on Apple platforms (`SINGULARITY_BACKEND_METAL=ON` by default when `APPLE`).

### Running the physics verification

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e verification/
pytest verification/
```

### Building the web demo (optional)

Requires Emscripten. See [the docs site section](#web-docs-site--wasm-demo) below.

---

## Linux

Linux builds drive the portable C++ core plus the Vulkan backend (same path
the Windows build uses); all physics tests, the headless CLI, and the Python
verification harness run unchanged. The interactive desktop app awaits
Phase 2 / Phase 4 GPU work.

### Prerequisites (Linux)

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build g++-14 python3-venv
```

For the Vulkan backend (when it lands — Phase 4):

```sh
sudo apt-get install -y vulkan-sdk libvulkan-dev vulkan-validationlayers
# Or install the official LunarG SDK for a newer version:
#   https://vulkan.lunarg.com/sdk/home#linux
```

### Configure and build (Linux)

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cli/singularity_cli --help
```

### Running the test suites (Linux)

```sh
# C++ Catch2 tests (physics, integrators, disc, scene config, …)
(cd build && ctest --output-on-failure)

# Python verification harness (SymPy / SciPy / Planck-CIE cross-checks)
python3 -m venv verification/.venv
source verification/.venv/bin/activate
pip install -e verification
SINGULARITY_CLI=$PWD/build/cli/singularity_cli pytest verification/ -v
```

### CLI sample renders

```sh
./build/cli/singularity_cli --mode 2d-toy           --output phase1.png
./build/cli/singularity_cli --mode kerr-2d-toy --spin 0.9 --output kerr.png
./build/cli/singularity_cli --mode disc-preview --spin 0.9 --output disc.png
./build/cli/singularity_cli --mode photon-orbit --spin 0.5 --h-step 0.001 \
    --output orbit.csv
./build/cli/singularity_cli --mode kerr-geometry --spin 0.94 --output geom.json
```

### WSL2 notes

WSL2 is fine for the CPU physics + CLI + tests. The Vulkan backend *can*
run via WSLg but the present path goes through a compositor hop, so FPS
numbers you measure there won't match native — reserve GPU profiling for a
real Linux / Windows host. The Metal backend is macOS-only regardless.

---

## Windows

### Prerequisites

1. **Visual Studio 2022 17.9+** with the "Desktop development with C++" workload.
2. **CMake** (installer from <https://cmake.org/download/>, or `winget install Kitware.CMake`).
3. **Vulkan SDK** from LunarG — required for headers, loader, validation layers, and the `dxc` compiler used to build shaders from HLSL.
   - Download: <https://vulkan.lunarg.com/sdk/home#windows>
   - Installer sets `VULKAN_SDK` env var, which `find_package(Vulkan)` picks up.
4. **(Optional) WiX Toolset v5** for building `.msi` installers. Only needed for release builds.

### Configure and build

From a "Developer PowerShell for VS 2022" prompt:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\singularity.exe
```

Or with Ninja (faster):

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\singularity.exe
```

The Vulkan backend is enabled automatically on Windows (`SINGULARITY_BACKEND_VULKAN=ON` by default when `WIN32`).

### Validation layers

For development, install the Vulkan SDK's validation layers. The build defaults to enabling `VK_LAYER_KHRONOS_validation` in Debug configurations; Release configurations skip it for perf.

### Python verification harness

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e verification/
pytest verification/
```

---

## CUDA backend (optional stretch — Phase 8)

Requires an NVIDIA GPU with a recent driver and the CUDA Toolkit 12.x+. Windows-only initially; Linux support is a Phase 9 stretch.

```powershell
cmake -B build -G Ninja -DSINGULARITY_BACKEND_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\singularity_cuda_cli.exe --scene scenes/kerr_4k.json --out out.png
```

The CUDA backend does **not** drive the interactive desktop app — it is a separate headless binary for offline high-sample-count rendering. See `docs/ARCHITECTURE.md` §7.

---

## Build options

All options default based on host platform; override via `-D<OPTION>=ON|OFF`.

| Option | Default | Effect |
|---|---|---|
| `SINGULARITY_BACKEND_METAL` | `ON` on Apple, `OFF` elsewhere | Build Metal backend (requires Apple Clang + metal compiler) |
| `SINGULARITY_BACKEND_VULKAN` | `ON` on Windows, `OFF` elsewhere | Build Vulkan backend (requires Vulkan SDK + DXC) |
| `SINGULARITY_BACKEND_CUDA` | `OFF` | Build CUDA offline-renderer backend (requires CUDA Toolkit) |
| `SINGULARITY_BUILD_WEB` | `OFF` | Build WASM target via Emscripten (see below) |
| `SINGULARITY_BUILD_TESTS` | `ON` if `BUILD_TESTING` | Build Catch2 unit tests |
| `SINGULARITY_BUILD_CLI` | `ON` | Build `singularity_cli` (headless, for Python verification) |
| `SINGULARITY_ENABLE_ASAN` | `OFF` | AddressSanitizer + UBSan (Debug builds) |

Build types: standard CMake `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel`.

---

## Web docs site & WASM demo

Built from the `web/` subdirectory. Requires Node.js 20+ and Emscripten.

```sh
# One-time Emscripten setup (if you don't already have emsdk):
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh

# Build WASM from project root:
emcmake cmake -B build-wasm -DSINGULARITY_BUILD_WEB=ON
cmake --build build-wasm

# Build and run the docs site:
cd web
npm install
npm run dev
# opens http://localhost:3000
```

---

## Release builds

Release workflow is driven by GitHub Actions (`.github/workflows/release.yml`) on tag push. Local release builds:

### macOS signed `.dmg`

Requires Apple Developer Program membership ($99/yr) and a Developer ID Application certificate installed in Keychain.

```sh
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target singularity_dmg
# Output: build-release/Singularity-x.y.z.dmg (signed + notarized)
```

### Windows `.msi`

Requires WiX Toolset.

```powershell
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target singularity_msi
# Output: build-release\Singularity-x.y.z.msi (unsigned by default; see docs/PRD.md §10 for signing)
```

---

## Common problems

**`find_package(Vulkan)` fails on Windows.**
The Vulkan SDK installer sets `VULKAN_SDK`, but a reboot or new shell is required for it to propagate. Check `echo $env:VULKAN_SDK` in PowerShell.

**Metal shaders fail to compile with "cannot find -lmetal".**
You need the full Xcode install, not just command-line tools. Run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

**DXC not found when compiling HLSL → SPIR-V.**
The Vulkan SDK bundles `dxc`. If CMake can't find it, set `DXC_EXECUTABLE` manually: `-DDXC_EXECUTABLE=$env:VULKAN_SDK/Bin/dxc.exe`.

**"Third-party sources missing" at configure time.**
`third_party/` is populated by git submodules and a bootstrap script. From a fresh clone:
```sh
git submodule update --init --recursive
```
For dependencies vendored directly (Catch2 single-header, stb, metal-cpp), see `third_party/README.md` for fetch instructions.
