# ----------------------------------------------------------------------------
# singularity_core — platform-agnostic C++ physics + helpers
#
# The physics math itself lives in header-only files under core/include/physics/
# so it can be #included from MSL/HLSL/CUDA/host C++ kernels (see
# docs/ARCHITECTURE.md §4). This library is the CPU-side glue: settings
# loading, scene state, camera math, etc.
# ----------------------------------------------------------------------------

set(SINGULARITY_CORE_SOURCES
    # CPU-side sources will be added as Phase 1+ lands.
)

if(SINGULARITY_CORE_SOURCES)
    add_library(singularity_core STATIC ${SINGULARITY_CORE_SOURCES})
    target_include_directories(singularity_core PUBLIC
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/shared_shader
    )
else()
    # Header-only while the core is still being written in Phase 1.
    add_library(singularity_core INTERFACE)
    target_include_directories(singularity_core INTERFACE
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/shared_shader
    )
endif()

add_library(singularity::core ALIAS singularity_core)
