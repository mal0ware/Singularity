# ----------------------------------------------------------------------------
# Renderer — the abstract interface lives here; concrete backends live in
# subdirectories gated behind the matching SINGULARITY_BACKEND_* option.
# See docs/ARCHITECTURE.md §3.
#
# The library is STATIC when at least one backend is compiled in (so
# create_default_backend can route to it) and INTERFACE otherwise. The CLI /
# verification path doesn't need the factory — only the app shell does — so
# this keeps the headless build graph tiny.
# ----------------------------------------------------------------------------

set(_render_has_any_backend FALSE)
if(SINGULARITY_BACKEND_METAL OR SINGULARITY_BACKEND_VULKAN OR SINGULARITY_BUILD_WEB)
    set(_render_has_any_backend TRUE)
endif()

if(_render_has_any_backend)
    add_library(singularity_render STATIC
        src/create_default_backend.cpp
    )
    target_include_directories(singularity_render PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(singularity_render PUBLIC singularity::core)

    if(SINGULARITY_BACKEND_METAL)
        target_compile_definitions(singularity_render PUBLIC SINGULARITY_HAS_METAL)
    endif()
    if(SINGULARITY_BACKEND_VULKAN)
        target_compile_definitions(singularity_render PUBLIC SINGULARITY_HAS_VULKAN)
    endif()
    if(SINGULARITY_BUILD_WEB)
        target_compile_definitions(singularity_render PUBLIC SINGULARITY_HAS_WEBGPU)
    endif()
else()
    add_library(singularity_render INTERFACE)
    target_include_directories(singularity_render INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(singularity_render INTERFACE singularity::core)
endif()

add_library(singularity::render ALIAS singularity_render)

if(SINGULARITY_BACKEND_METAL)
    add_subdirectory(metal)
    # Private-link after the subdirectory so the concrete backend symbols are
    # resolvable from create_default_backend.cpp.
    # Public link so the concrete backend symbols are transitively visible
    # from the app shell — static libs don't forward PRIVATE deps to their
    # own consumers, which would force the app to re-list the target and
    # the linker to emit a duplicate-library warning.
    target_link_libraries(singularity_render PUBLIC singularity::render_metal)
endif()

if(SINGULARITY_BACKEND_VULKAN)
    add_subdirectory(vulkan)
    target_link_libraries(singularity_render PUBLIC singularity::render_vulkan)
endif()

if(SINGULARITY_BACKEND_CUDA)
    add_subdirectory(cuda)
endif()

if(SINGULARITY_BUILD_WEB)
    add_subdirectory(webgpu)
    target_link_libraries(singularity_render PUBLIC singularity::render_webgpu)
endif()
