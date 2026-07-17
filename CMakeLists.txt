cmake_minimum_required(VERSION 3.27)

if(APPLE)
    project(singularity
        VERSION 1.0.0
        DESCRIPTION "Real-time physically-accurate black hole simulator"
        HOMEPAGE_URL "https://github.com/mal0ware/Singularity"
        LANGUAGES CXX OBJCXX
    )
else()
    project(singularity
        VERSION 1.0.0
        DESCRIPTION "Real-time physically-accurate black hole simulator"
        HOMEPAGE_URL "https://github.com/mal0ware/Singularity"
        LANGUAGES CXX
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# On macOS, when CMake itself runs under Rosetta its autodetected host arch
# is x86_64 even on Apple Silicon hardware — which quietly produces an
# x86_64 binary that Rosetta re-translates at runtime. Ask uname(1) which
# chip this actually is and default CMAKE_OSX_ARCHITECTURES accordingly.
# Users packaging releases should pass -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
# to get a universal binary.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    # Without this block, cmake running under Rosetta (x86_64) quietly emits
    # an x86_64 binary on Apple Silicon. `uname -m` and even `arch` lie
    # under Rosetta — they report the process's translated arch, not the
    # real CPU. `sysctl hw.optional.arm64` bypasses the translation and
    # reports the true hardware. Users wanting a universal release binary
    # can pass -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" explicitly.
    execute_process(
        COMMAND sysctl -n hw.optional.arm64
        OUTPUT_VARIABLE _is_arm64
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_is_arm64 STREQUAL "1")
        set(_native_arch "arm64")
    else()
        set(_native_arch "x86_64")
    endif()
    set(CMAKE_OSX_ARCHITECTURES ${_native_arch} CACHE STRING
        "macOS target architectures (default: native hardware)" FORCE)
    message(STATUS "Defaulting CMAKE_OSX_ARCHITECTURES to ${_native_arch}")
endif()

# ----------------------------------------------------------------------------
# Options — see docs/ARCHITECTURE.md §9 and BUILDING.md
# ----------------------------------------------------------------------------
option(SINGULARITY_BACKEND_METAL  "Build Metal backend"  ${APPLE})
option(SINGULARITY_BACKEND_VULKAN "Build Vulkan backend" ${WIN32})
option(SINGULARITY_BACKEND_CUDA   "Build CUDA backend (offline renderer)" OFF)
option(SINGULARITY_BUILD_WEB      "Build WASM target via Emscripten" OFF)
option(SINGULARITY_BUILD_TESTS    "Build Catch2 unit tests" ON)
option(SINGULARITY_BUILD_CLI      "Build singularity_cli headless driver" ON)
# Auto-enable the desktop app when a real-time backend is compiled in — the
# app has no purpose without one, and it keeps the default on-Apple build
# matching the ARCHITECTURE.md vision without the user needing extra flags.
if(SINGULARITY_BACKEND_METAL OR SINGULARITY_BACKEND_VULKAN)
    option(SINGULARITY_BUILD_APP  "Build desktop app shell (requires SDL3 + ImGui)" ON)
else()
    option(SINGULARITY_BUILD_APP  "Build desktop app shell (requires SDL3 + ImGui)" OFF)
endif()
option(SINGULARITY_ENABLE_ASAN    "Enable AddressSanitizer + UBSan (Debug)" OFF)
# CI guard: a requested backend whose dependencies are missing normally
# degrades to a warning + stub build (friendly for local hacking). Under this
# option it is a hard configure error instead, so CI can never go green while
# silently building stubs — which is exactly how a Vulkan SDK layout change
# slipped past the matrix once.
option(SINGULARITY_REQUIRE_BACKEND "Fail configure if a requested GPU backend's dependencies are missing" OFF)

# ----------------------------------------------------------------------------
# Platform-aware sanity checks
# ----------------------------------------------------------------------------
if(SINGULARITY_BACKEND_METAL AND NOT APPLE)
    message(FATAL_ERROR
        "SINGULARITY_BACKEND_METAL requires an Apple platform.")
endif()

if(SINGULARITY_BUILD_WEB AND NOT EMSCRIPTEN)
    message(FATAL_ERROR
        "SINGULARITY_BUILD_WEB requires Emscripten. "
        "Invoke CMake via `emcmake cmake ...`.")
endif()

# ----------------------------------------------------------------------------
# Warning / optimization flags
# ----------------------------------------------------------------------------
# CUDA TUs go through nvcc, which forwards unknown host-compiler flags only
# when wrapped in -Xcompiler. Putting raw `/W4` etc. into add_compile_options
# would make nvcc treat them as input files. Guard the language-specific
# warning/optimization flags to CXX (and C) so the CUDA backend's .cu files
# build with whatever defaults nvcc + the host compiler agree on, and let
# render/cuda/CMakeLists.txt set CUDA-specific flags via target properties.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                       $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                       $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                       $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/fp:fast>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                       $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                       $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                       $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++98-compat>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-ffast-math>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0>)
    add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g>)
    if(SINGULARITY_ENABLE_ASAN)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# MinGW: statically link the GCC runtime so a locally-cpack'd .msi runs without
# libgcc_s_seh-1.dll / libstdc++-6.dll / libwinpthread-1.dll next to the exe.
# MSVC builds (GHA CI) rely on the ubiquitous VC++ Redist and do not need this.
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

# ----------------------------------------------------------------------------
# Helpers: shader compilation functions
# ----------------------------------------------------------------------------
include(SingularityShaders)

# ----------------------------------------------------------------------------
# Third-party dependencies
#
# Populated either as git submodules or fetched via the bootstrap script.
# See third_party/README.md. We skip third_party entirely if it hasn't been
# populated yet, so a fresh clone can at least run `cmake -B build` without
# exploding — missing targets surface clearly at `cmake --build` time.
# ----------------------------------------------------------------------------
set(SINGULARITY_DEPS_OK TRUE)

macro(singularity_require_dep path label)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${path}")
        message(WARNING
            "${label} not found at ${path}. See BUILDING.md / third_party/README.md. "
            "Configure will succeed but build will fail until the dependency is fetched.")
        set(SINGULARITY_DEPS_OK FALSE)
    endif()
endmacro()

# Only require what the current build configuration actually needs.
if(SINGULARITY_BUILD_CLI)
    singularity_require_dep("third_party/stb/stb_image_write.h" "stb_image_write")
endif()

if(SINGULARITY_BUILD_TESTS)
    singularity_require_dep("third_party/catch2/catch_amalgamated.hpp" "Catch2 v3 single-header")
endif()

if(SINGULARITY_BUILD_APP)
    # Desktop app needs SDL3 (windowing + input) and Dear ImGui (overlay).
    # The WebGPU demo target (SINGULARITY_BUILD_WEB) drives its own event
    # loop via emscripten_set_main_loop and runs overlay-less for the
    # first pass — no SDL, no ImGui — so those deps aren't required here.
    singularity_require_dep("third_party/SDL/CMakeLists.txt"    "SDL3 submodule")
    singularity_require_dep("third_party/imgui/imgui.h"         "Dear ImGui")
endif()

if(SINGULARITY_BACKEND_METAL)
    singularity_require_dep("third_party/metal-cpp/Metal/Metal.hpp" "metal-cpp")
endif()

if(SINGULARITY_BACKEND_VULKAN)
    find_package(Vulkan 1.3 QUIET)
    if(NOT Vulkan_FOUND)
        # Fallback: if the Vulkan SDK isn't installed but the loader
        # (libvulkan.so.1, shipped as part of libvulkan1 on stock Ubuntu /
        # most distros) and the vendored Vulkan-Headers submodule are both
        # present, synthesise a Vulkan::Vulkan target against them. Lets
        # developers build the Vulkan backend without chasing `libvulkan-dev`
        # specifically; the full LunarG SDK is still required for DXC
        # (shader compilation) but the backend C++ can compile + link.
        set(_vk_vendored_hdr
            "${CMAKE_SOURCE_DIR}/third_party/Vulkan-Hpp/Vulkan-Headers/include")
        set(_vk_loader NOTFOUND)
        # Try the common loader locations + filenames. find_library strips
        # the versioned suffix, so we add find_file for .so.1 explicitly.
        find_library(_vk_loader NAMES vulkan vulkan-1
                     PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64)
        if(NOT _vk_loader)
            find_file(_vk_loader NAMES libvulkan.so.1 vulkan-1.dll
                      PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64)
        endif()
        if(EXISTS "${_vk_vendored_hdr}/vulkan/vulkan.h" AND _vk_loader)
            add_library(Vulkan::Vulkan UNKNOWN IMPORTED GLOBAL)
            set_target_properties(Vulkan::Vulkan PROPERTIES
                IMPORTED_LOCATION "${_vk_loader}"
                INTERFACE_INCLUDE_DIRECTORIES "${_vk_vendored_hdr}")
            set(Vulkan_FOUND TRUE)
            message(STATUS
                "Vulkan SDK not installed — using vendored headers "
                "(${_vk_vendored_hdr}) + loader (${_vk_loader})")
        else()
            if(SINGULARITY_REQUIRE_BACKEND)
                message(FATAL_ERROR
                    "SINGULARITY_REQUIRE_BACKEND is ON and the Vulkan SDK was "
                    "not found (find_package(Vulkan) failed and the vendored-"
                    "headers + system-loader fallback did not resolve). "
                    "Refusing to silently build the Vulkan backend as stubs. "
                    "Install https://vulkan.lunarg.com/sdk/home and set "
                    "VULKAN_SDK.")
            endif()
            message(WARNING
                "Vulkan SDK not found via find_package(Vulkan), and the "
                "vendored Vulkan-Headers submodule + system loader fallback "
                "did not resolve either. The Vulkan backend will compile as "
                "stubs without linking vulkan-1. Install "
                "https://vulkan.lunarg.com/sdk/home and set VULKAN_SDK to "
                "wire the real runtime path.")
        endif()
    endif()
    singularity_require_dep("third_party/Vulkan-Hpp/vulkan/vulkan.hpp" "Vulkan-Hpp")
endif()

if(SINGULARITY_BACKEND_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(SINGULARITY_DEPS_OK AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/CMakeLists.txt")
    add_subdirectory(third_party)
endif()

# ----------------------------------------------------------------------------
# Core library: physics + CPU-side helpers (API-neutral)
# ----------------------------------------------------------------------------
add_subdirectory(core)

# ----------------------------------------------------------------------------
# Renderer: abstract interface + backend implementations
# ----------------------------------------------------------------------------
add_subdirectory(render)

# ----------------------------------------------------------------------------
# Executable targets
# ----------------------------------------------------------------------------
if(SINGULARITY_BUILD_APP AND NOT SINGULARITY_BUILD_WEB)
    add_subdirectory(app)
endif()

if(SINGULARITY_BUILD_CLI AND NOT SINGULARITY_BUILD_WEB)
    add_subdirectory(cli)
endif()

if(SINGULARITY_BACKEND_CUDA)
    add_subdirectory(cuda_cli)
endif()

# ----------------------------------------------------------------------------
# Web target — Emscripten produces singularity_web.html (+ .js + .wasm).
# Only built when SINGULARITY_BUILD_WEB=ON, which already implies the
# EMSCRIPTEN toolchain (fatal_error in the Phase 0 sanity checks). The
# web entry lives in web/main.cpp and the WebGPU backend subdirectory
# was added under render/.
# ----------------------------------------------------------------------------
if(SINGULARITY_BUILD_WEB)
    add_subdirectory(web)
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(SINGULARITY_BUILD_TESTS AND SINGULARITY_DEPS_OK)
    enable_testing()
    add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Packaging — CPack
#
# Windows: WIX generator produces Singularity-<version>-win64.msi. Needs the
# WiX 3.x toolset on PATH (windows-2022 GitHub runners have it pre-installed;
# locally: `choco install wixtoolset`). The .msi installs the exe + compiled
# SPIR-V under Program Files\Singularity\, creates a Start-Menu shortcut,
# and wires uninstall via Programs-and-Features.
# macOS: DragNDrop → .dmg around the existing codesigned .app bundle.
# Linux: TGZ developer tarball; no consumer shipping path planned.
# ----------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "Singularity")
set(CPACK_PACKAGE_VENDOR "mal0ware")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Real-time physically-accurate black hole simulator")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/mal0ware/Singularity")
set(CPACK_PACKAGE_CONTACT "mal0ss.network@gmail.com")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Singularity")
# WiX only accepts .txt / .rtf licenses; our repo ships `LICENSE` with no
# extension (common on GitHub so the license detector works). Copy it into
# the build dir with a `.txt` extension purely for packaging.
configure_file("${CMAKE_SOURCE_DIR}/LICENSE"
               "${CMAKE_BINARY_DIR}/LICENSE.txt"
               COPYONLY)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/LICENSE.txt")
set(CPACK_PACKAGE_EXECUTABLES "singularity;Singularity")

if(WIN32)
    set(CPACK_GENERATOR "WIX")
    # Stable upgrade identity — same UUID for every version forever, so
    # Windows Installer can upgrade prior installs in place. Generated once
    # via uuidgen -r on 2026-04-23. DO NOT REGENERATE.
    set(CPACK_WIX_UPGRADE_GUID "34525920-53BD-4CFB-94DD-EA3B1D25800C")
    set(CPACK_WIX_PROGRAM_MENU_FOLDER "Singularity")
    set(CPACK_WIX_ROOT_FEATURE_TITLE "Singularity")
    set(CPACK_WIX_PROPERTY_ARPURLINFOABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_WIX_PROPERTY_ARPHELPLINK
        "https://github.com/mal0ware/Singularity/issues")
    # -sval skips ICE validation warnings; the MIT-license .txt is bigger
    # than the MSI warning threshold but still under the hard limit.
    set(CPACK_WIX_LIGHT_EXTRA_FLAGS "-sval")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "Singularity")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

# ----------------------------------------------------------------------------
# Configuration summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Singularity ${PROJECT_VERSION} ===")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  Metal backend:      ${SINGULARITY_BACKEND_METAL}")
message(STATUS "  Vulkan backend:     ${SINGULARITY_BACKEND_VULKAN}")
message(STATUS "  CUDA backend:       ${SINGULARITY_BACKEND_CUDA}")
message(STATUS "  Web (WASM) build:   ${SINGULARITY_BUILD_WEB}")
message(STATUS "  Build CLI:          ${SINGULARITY_BUILD_CLI}")
message(STATUS "  Build tests:        ${SINGULARITY_BUILD_TESTS}")
message(STATUS "  AddressSanitizer:   ${SINGULARITY_ENABLE_ASAN}")
message(STATUS "  Dependencies OK:    ${SINGULARITY_DEPS_OK}")
message(STATUS "")
