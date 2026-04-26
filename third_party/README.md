# third_party/

Vendored and submoduled dependencies. Populated by `git submodule update --init --recursive` plus the direct downloads listed below.

| Dir | Kind | Source | Version / tag | License |
|---|---|---|---|---|
| [`SDL/`](./SDL/) | submodule | <https://github.com/libsdl-org/SDL.git> | `release-3.2.x` | Zlib |
| [`imgui/`](./imgui/) | submodule | <https://github.com/ocornut/imgui.git> (with `docking` branch) | `v1.91+` | MIT |
| [`Vulkan-Hpp/`](./Vulkan-Hpp/) | submodule | <https://github.com/KhronosGroup/Vulkan-Hpp.git> | matches Vulkan SDK 1.3.x | Apache-2.0 |
| [`vma/`](./vma/) | submodule | <https://github.com/GPUOpen-LibrariesAndSAMples/VulkanMemoryAllocator.git> | `v3.x` | MIT |
| [`metal-cpp/`](./metal-cpp/) | direct download | <https://developer.apple.com/metal/cpp/> | `metal3.1` | Apache-2.0 |
| [`catch2/`](./catch2/) | single-header | <https://github.com/catchorg/Catch2/releases> — `catch_amalgamated.hpp` + `.cpp` | `v3.x` | BSL-1.0 |
| [`stb/`](./stb/) | single-header | <https://github.com/nothings/stb> — `stb_image.h`, `stb_image_write.h` | HEAD | MIT / public domain |
| [`json/`](./json/) | submodule | <https://github.com/nlohmann/json.git> | `v3.11+` | MIT |

## Bootstrap from scratch

```sh
# From project root — submodules first:
git submodule update --init --recursive

# Direct downloads not available as git submodules:
mkdir -p third_party/metal-cpp third_party/stb third_party/catch2

# metal-cpp (Apple redistributable headers):
curl -L https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip \
  -o /tmp/metal-cpp.zip && unzip /tmp/metal-cpp.zip -d third_party/

# stb (pick the headers we use):
for h in stb_image.h stb_image_write.h; do
  curl -L "https://raw.githubusercontent.com/nothings/stb/master/$h" \
    -o "third_party/stb/$h"
done

# Catch2 v3 single-header amalgamation:
curl -L https://github.com/catchorg/Catch2/releases/latest/download/catch_amalgamated.hpp \
  -o third_party/catch2/catch_amalgamated.hpp
curl -L https://github.com/catchorg/Catch2/releases/latest/download/catch_amalgamated.cpp \
  -o third_party/catch2/catch_amalgamated.cpp
```

Once `third_party/` is populated, `cmake -B build` should configure cleanly on both macOS and Windows.

## Licenses

All vendored code retains its upstream license. The aggregated Singularity source (everything _not_ under `third_party/`) is MIT; see [`../LICENSE`](../LICENSE).
