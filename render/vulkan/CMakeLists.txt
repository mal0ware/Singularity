# ----------------------------------------------------------------------------
# Vulkan backend — Windows primary target, also builds under Linux (with
# Mesa llvmpipe/lavapipe for CI) and can link on macOS via MoltenVK for
# cross-backend equivalence tests down the road.
#
# See docs/ARCHITECTURE.md §6.
#
# Status today (2026-04-23): scaffolding. HLSL shaders compile to SPIR-V,
# the C++ wrapper class links, but the Vulkan body inside vulkan_backend.cpp
# is stubbed behind warn_once() messages. The Windows session fills those in.
# ----------------------------------------------------------------------------

set(VULKAN_BACKEND_SOURCES
    vulkan_backend.cpp
    vulkan_backend.hpp
)

add_library(singularity_render_vulkan STATIC ${VULKAN_BACKEND_SOURCES})
# Vulkan-Headers must come before any system SDK include that Vulkan::Vulkan
# brings in as an INTERFACE_INCLUDE_DIRECTORY. Vulkan-Hpp (v1.4.309 in the
# submodule) static_asserts VK_HEADER_VERSION; if the SDK on the host is
# older (Windows CI pins 1.3.280) the C vulkan.h from the SDK wins the
# header lookup and the assert fires. Prepending the vendored
# Vulkan-Headers/include pins both sides to the same version.
target_include_directories(singularity_render_vulkan
    PUBLIC  ${CMAKE_SOURCE_DIR}/render/include
    PRIVATE ${CMAKE_SOURCE_DIR}/shared_shader
            ${CMAKE_SOURCE_DIR}/third_party/Vulkan-Hpp/Vulkan-Headers/include
            ${CMAKE_SOURCE_DIR}/third_party/Vulkan-Hpp
)
# Link Vulkan only when the loader was actually located. Without it the
# scaffolding still compiles (the Vulkan-Hpp include is `__has_include`-
# gated inside vulkan_backend.cpp), so CI on machines without the SDK
# gets to validate the C++ boundary without linker-failing.
if(TARGET Vulkan::Vulkan)
    target_link_libraries(singularity_render_vulkan PRIVATE Vulkan::Vulkan)
    # Flag the real Vulkan path on — the scaffold's #ifdef gate opens up
    # the Vulkan-Hpp usage. Without this define the backend compiles as
    # no-op stubs (intentional for machines without the SDK).
    target_compile_definitions(singularity_render_vulkan PRIVATE
        SINGULARITY_VULKAN_SDK_AVAILABLE=1)
endif()
add_library(singularity::render_vulkan ALIAS singularity_render_vulkan)

# --- HLSL → SPIR-V ----------------------------------------------------------
#
# Each .hlsl file has one or more named entry points. We compile every
# entry into its own .spv artefact so the Vulkan backend can load them
# by filename without a runtime entry-point parameter.
find_program(DXC_EXECUTABLE dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    DOC "DirectX Shader Compiler (ships with the LunarG Vulkan SDK)"
)

if(NOT DXC_EXECUTABLE)
    message(WARNING
        "DXC not found — shader SPIR-V compilation is skipped. Install the "
        "LunarG Vulkan SDK (https://vulkan.lunarg.com/sdk/home) so "
        "$ENV{VULKAN_SDK}/Bin/dxc is on PATH, then reconfigure. The Vulkan "
        "backend library will still compile (it's scaffolding), and the "
        "Metal path is unaffected.")
else()
    set(_vk_shader_inc "-I${CMAKE_SOURCE_DIR}/shared_shader")
    set(_spv_artefacts "")

    function(singularity_compile_hlsl SRC PROFILE ENTRY OUTNAME)
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/${OUTNAME}.spv")
        add_custom_command(
            OUTPUT ${_spv}
            COMMAND ${DXC_EXECUTABLE}
                    -T ${PROFILE}
                    -E ${ENTRY}
                    -spirv
                    -fspv-target-env=vulkan1.3
                    ${_vk_shader_inc}
                    ${SRC}
                    -Fo ${_spv}
            DEPENDS ${SRC}
            COMMENT "HLSL ${OUTNAME} (${PROFILE}/${ENTRY}) -> ${OUTNAME}.spv"
            VERBATIM
        )
        set(_spv_artefacts ${_spv_artefacts} ${_spv} PARENT_SCOPE)
    endfunction()

    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/geodesic_kernel.hlsl
        cs_6_0 main geodesic_kernel)
    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bloom.hlsl cs_6_0 extract bloom_extract)
    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bloom.hlsl cs_6_0 blur_h  bloom_blur_h)
    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bloom.hlsl cs_6_0 blur_v  bloom_blur_v)
    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/blit.hlsl  vs_6_0 main_vs blit_vs)
    singularity_compile_hlsl(
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/blit.hlsl  ps_6_0 main_ps blit_ps)

    add_custom_target(singularity_spirv ALL DEPENDS ${_spv_artefacts})
    add_dependencies(singularity_render_vulkan singularity_spirv)

    set(SINGULARITY_SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL
        "Directory containing compiled SPIR-V artefacts")

    # Tell the backend where to look for .spv files when neither the
    # SINGULARITY_SPV_DIR env var nor a shaders/ dir next to the exe is
    # populated — this is the dev-build-in-tree default.
    target_compile_definitions(singularity_render_vulkan PRIVATE
        SINGULARITY_SPV_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")
endif()
