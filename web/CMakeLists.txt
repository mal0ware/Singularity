# ----------------------------------------------------------------------------
# Web target — produces singularity_web.html (+ .js + .wasm) via Emscripten.
#
# Added to the top-level graph only when SINGULARITY_BUILD_WEB=ON, which
# is already gated on the EMSCRIPTEN toolchain (the top-level fatal_error
# catches a native invocation early). `emcmake cmake -B build-wasm
# -DSINGULARITY_BUILD_WEB=ON` is the one-step configure.
# ----------------------------------------------------------------------------

add_executable(singularity_web main.cpp)
set_target_properties(singularity_web PROPERTIES SUFFIX ".html")

target_link_libraries(singularity_web PRIVATE
    singularity::render
    singularity::render_webgpu
    singularity::core
)
# web/main.cpp #includes "render/webgpu/webgpu_backend.hpp" directly.
# The render library's PUBLIC include is only render/include; the
# webgpu backend lives one level out from there. Adding the repo root
# here keeps the include path canonical (same style as app/ uses).
target_include_directories(singularity_web PRIVATE ${CMAKE_SOURCE_DIR})

# Emscripten flags. --use-port=emdawnwebgpu enables the webgpu.h surface
# at compile + link (replaces the older -sUSE_WEBGPU=1 in emsdk 5.x).
# -sALLOW_MEMORY_GROWTH=1 lets the wasm heap expand past its initial
# size once the swapchain + HDR + bloom buffers are allocated.
# --preload-file bundles the WGSL shader directory into the virtual FS
# under "/shaders", so fopen("/shaders/geodesic_kernel.wgsl", "rb") works
# unchanged between desktop and web paths.
#
# The EXPORTED_FUNCTIONS / RUNTIME_METHODS lists are kept minimal — we
# only need main() exported and ccall/cwrap available if a future patch
# wants to drive the renderer from JavaScript (e.g. an ImGui-less UI).
set(_emscripten_flags
    --use-port=emdawnwebgpu
    -sWASM=1
    -sALLOW_MEMORY_GROWTH=1
    -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']
    # main + the DOM-overlay panel's C bridge (mirrors the EMSCRIPTEN_KEEPALIVE
    # setters at the bottom of web/main.cpp). Listed explicitly so the linker
    # doesn't strip them as unused — KEEPALIVE alone is fragile under LTO.
    -sEXPORTED_FUNCTIONS=['_main','_singularity_set_metric','_singularity_set_spin','_singularity_set_disc_visible','_singularity_set_disc_inner','_singularity_set_disc_outer','_singularity_set_disc_peak_t','_singularity_set_disc_turbulence','_singularity_set_doppler','_singularity_set_redshift','_singularity_set_exposure','_singularity_set_bloom_threshold','_singularity_set_bloom_strength','_singularity_set_fov','_singularity_reset']
)

target_compile_options(singularity_web PRIVATE ${_emscripten_flags})
target_link_options(singularity_web PRIVATE
    ${_emscripten_flags}
    --preload-file ${SINGULARITY_WGSL_DIR}@/shaders
    --shell-file   ${CMAKE_CURRENT_SOURCE_DIR}/shell.html
)

# Stamp a symlink / copy of the generated .html under a stable name
# ("index.html") so `python3 -m http.server` pointed at the build
# directory serves the demo at `/`.
add_custom_command(TARGET singularity_web POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:singularity_web>
            ${CMAKE_CURRENT_BINARY_DIR}/index.html
    COMMENT "Staging singularity_web.html -> index.html for http.server"
    VERBATIM
)
