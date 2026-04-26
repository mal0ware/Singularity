# ----------------------------------------------------------------------------
# CUDA backend — NVIDIA-only, offline renderer for the `singularity_cuda_cli`
# binary (not used by the interactive desktop apps). See docs/ARCHITECTURE.md
# §7 for the rationale.
#
# Status (2026-04-24): skeleton. The `CudaBackend` class compiles against
# the `RenderBackend` interface and its kernel is a solid-color fill that
# round-trips a PNG — enough to validate the CMake + nvcc + cudaMalloc /
# cudaMemcpy surface end-to-end. The actual geodesic port (pixel-per-
# thread ray trace sharing `shared_shader/geodesic_math.h`) is the main
# body of Phase 8 and replaces the stub kernel without touching the
# surrounding plumbing.
# ----------------------------------------------------------------------------

find_package(CUDAToolkit REQUIRED)

set(CUDA_BACKEND_SOURCES
    cuda_backend.cu
    cuda_backend.hpp
    kernels/geodesic_kernel.cu
)

add_library(singularity_render_cuda STATIC ${CUDA_BACKEND_SOURCES})

target_include_directories(singularity_render_cuda
    PUBLIC  ${CMAKE_SOURCE_DIR}/render/include
            ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_SOURCE_DIR}/shared_shader
)

target_link_libraries(singularity_render_cuda
    PUBLIC  CUDA::cudart
    PRIVATE singularity::core
)

# RTX 30/40 generations cover the dev's hardware; override via
# -DCMAKE_CUDA_ARCHITECTURES for older cards (e.g. "75" for Turing, "60"
# for Pascal). 80+86+89 means fatbin includes Ampere consumer, Ampere
# datacenter, and Ada.
set_target_properties(singularity_render_cuda PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(singularity::render_cuda ALIAS singularity_render_cuda)
