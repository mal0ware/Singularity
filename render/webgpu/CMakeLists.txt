# ----------------------------------------------------------------------------
# WebGPU backend — Emscripten-only target. On a native (Metal / Vulkan) build
# this subdirectory is skipped entirely by the top-level CMakeLists.txt gate
# (SINGULARITY_BUILD_WEB implies EMSCRIPTEN — see the fatal_error in the
# Phase 0 sanity checks).
#
# See docs/ARCHITECTURE.md §6 and docs/PHASE7_WEBGPU.md.
#
# Status today: scaffolding. The WGSL shaders in ./shaders/ are hand-ported
# from the HLSL under ../vulkan/shaders/ and should compile through naga /
# the browser's WGSL front-end. The C++ wrapper class links, but the
# WebGPU bodies inside webgpu_backend.cpp are stubbed behind warn_once()
# messages. A session with emsdk + a WebGPU-capable browser fills those in.
# ----------------------------------------------------------------------------

set(WEBGPU_BACKEND_SOURCES
    webgpu_backend.cpp
    webgpu_backend.hpp
)

add_library(singularity_render_webgpu STATIC ${WEBGPU_BACKEND_SOURCES})
target_include_directories(singularity_render_webgpu
    PUBLIC  ${CMAKE_SOURCE_DIR}/render/include
    PRIVATE ${CMAKE_SOURCE_DIR}/shared_shader
)

# Expose webgpu.h to the .cpp. Emscripten 5.x replaced -sUSE_WEBGPU=1
# with the emdawnwebgpu port (based on Dawn's standards-track webgpu.h);
# the older flag still parses but emits an incompatibility error. The
# port is fetched on first use and cached under $EMSDK/.emscripten_cache.
target_compile_options(singularity_render_webgpu PRIVATE --use-port=emdawnwebgpu)

add_library(singularity::render_webgpu ALIAS singularity_render_webgpu)

# --- WGSL shader staging -----------------------------------------------------
#
# WGSL has no preprocessor and no ahead-of-time compiler in our toolchain
# (naga-cli is optional; the browser compiles WGSL on module creation).
# So we just stage the .wgsl files into a flat "/shaders" directory inside
# the virtual FS that Emscripten's --preload-file option packages. Runtime
# loads them via fopen("/shaders/geodesic_kernel.wgsl", ...).
#
# SINGULARITY_WGSL_DIR is exported as a compile definition on the consumer
# (web target) so it can locate the staged shaders without a hardcoded
# path.
set(SINGULARITY_WGSL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders" CACHE INTERNAL
    "Directory containing hand-ported WGSL shader sources")

# Optional WGSL validation via naga (if available). Non-fatal — the browser
# is the authoritative validator once the demo runs.
find_program(NAGA_EXECUTABLE naga
    DOC "WGSL validator from the wgpu-rs project (cargo install naga-cli)"
)
if(NAGA_EXECUTABLE)
    foreach(_wgsl geodesic_kernel bloom blit)
        add_custom_command(
            TARGET singularity_render_webgpu POST_BUILD
            COMMAND ${NAGA_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_wgsl}.wgsl
            COMMENT "naga validate ${_wgsl}.wgsl"
            VERBATIM
        )
    endforeach()
endif()
