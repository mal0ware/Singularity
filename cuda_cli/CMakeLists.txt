# ----------------------------------------------------------------------------
# singularity_cuda_cli — offline high-sample-count renderer, CUDA-only.
# Accepts a scene config (key=value text, same format as singularity_cli's
# --scene) and writes a single PNG. The companion to `singularity` (live app)
# and `singularity_cli` (CPU reference). See docs/TODO.md Phase 8.
# ----------------------------------------------------------------------------

add_executable(singularity_cuda_cli main.cpp)

target_link_libraries(singularity_cuda_cli
    PRIVATE
        singularity::core
        singularity::render_cuda
)

target_include_directories(singularity_cuda_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/stb
    ${CMAKE_SOURCE_DIR}/render/include
)

# stb_image_write.h — same reasoning as in cli/CMakeLists.txt. Silence its
# strict-warning noise in this one translation unit only.
if(NOT MSVC)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS
        "-Wno-missing-field-initializers -Wno-deprecated-declarations")
endif()
