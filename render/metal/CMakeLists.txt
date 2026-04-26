# ----------------------------------------------------------------------------
# Metal backend — Apple platforms only.
# See docs/ARCHITECTURE.md §5.
# ----------------------------------------------------------------------------

set(METAL_BACKEND_SOURCES
    metal_backend.mm
    metal_backend.hpp
)

set(METAL_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/geodesic_kernel.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bloom.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/blit.metal
)

add_library(singularity_render_metal STATIC ${METAL_BACKEND_SOURCES})
# Only link the frameworks — `singularity_render` depends on us, not the
# other way around. We include its headers directly via
# `target_include_directories` below to avoid a circular static-library link.
target_link_libraries(singularity_render_metal PRIVATE
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework Foundation"
    "-framework AppKit"
)
target_include_directories(singularity_render_metal
    PUBLIC  ${CMAKE_SOURCE_DIR}/render/include
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/metal-cpp
            ${CMAKE_SOURCE_DIR}/shared_shader
            ${CMAKE_CURRENT_SOURCE_DIR}
)
# The .mm sources need ARC off (we use manual retain/release patterns via
# dispatch_semaphore and @autoreleasepool) but do need modern Obj-C features.
set_source_files_properties(metal_backend.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc -fobjc-weak"
)
add_library(singularity::render_metal ALIAS singularity_render_metal)

add_metal_library(singularity_metallib
    OUTPUT_NAME default.metallib
    SOURCES ${METAL_SHADERS}
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/shared_shader
                 ${CMAKE_CURRENT_SOURCE_DIR}/shaders
)
add_dependencies(singularity_render_metal singularity_metallib)

# Expose the metallib path so the app shell can stage it into the bundle's
# Resources/ directory via a post-build step (see app/CMakeLists.txt).
get_target_property(_metallib singularity_metallib SINGULARITY_METALLIB_PATH)
set(SINGULARITY_METALLIB_FILE ${_metallib} CACHE INTERNAL
    "Absolute path to the compiled default.metallib")
