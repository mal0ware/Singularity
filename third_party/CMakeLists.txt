# ----------------------------------------------------------------------------
# third_party/CMakeLists.txt — only included when SINGULARITY_DEPS_OK and
# actually needed by the current configuration. Adds:
#   - SDL3                     (needed when SINGULARITY_BUILD_APP or BUILD_WEB)
#   - Dear ImGui core          (needed when SINGULARITY_BUILD_APP)
#   - Dear ImGui Metal backend (needed when SINGULARITY_BUILD_APP + Metal)
#   - Dear ImGui SDL3 backend  (needed when SINGULARITY_BUILD_APP)
# metal-cpp and stb are header-only; they're included via target include
# paths directly from the consuming library, not here.
# ----------------------------------------------------------------------------

if(SINGULARITY_BUILD_APP OR SINGULARITY_BUILD_WEB)
    # --- SDL3 -------------------------------------------------------------
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/SDL/CMakeLists.txt)
        set(SDL_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_STATIC ON  CACHE BOOL "" FORCE)
        set(SDL_TEST OFF   CACHE BOOL "" FORCE)
        add_subdirectory(SDL EXCLUDE_FROM_ALL)
    endif()

    # --- Dear ImGui core --------------------------------------------------
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.h)
        set(_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
        add_library(imgui STATIC
            ${_IMGUI_DIR}/imgui.cpp
            ${_IMGUI_DIR}/imgui_draw.cpp
            ${_IMGUI_DIR}/imgui_tables.cpp
            ${_IMGUI_DIR}/imgui_widgets.cpp
            ${_IMGUI_DIR}/imgui_demo.cpp
        )
        target_include_directories(imgui PUBLIC ${_IMGUI_DIR})
        target_compile_definitions(imgui PUBLIC
            IMGUI_DISABLE_OBSOLETE_FUNCTIONS
        )

        # SDL3 platform backend. The header compiles as C++, the source is
        # plain C++ too — no extra fuss.
        if(TARGET SDL3::SDL3 OR TARGET SDL3-static)
            add_library(imgui_sdl3 STATIC
                ${_IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
            )
            target_link_libraries(imgui_sdl3 PUBLIC imgui)
            if(TARGET SDL3::SDL3)
                target_link_libraries(imgui_sdl3 PUBLIC SDL3::SDL3)
            else()
                target_link_libraries(imgui_sdl3 PUBLIC SDL3-static)
            endif()
            target_include_directories(imgui_sdl3 PUBLIC ${_IMGUI_DIR})
        endif()

        # Metal backend is Objective-C++. Gate on Apple + Metal option so
        # non-Apple platforms don't accidentally try to compile it.
        if(APPLE AND SINGULARITY_BACKEND_METAL)
            add_library(imgui_metal STATIC
                ${_IMGUI_DIR}/backends/imgui_impl_metal.mm
            )
            target_link_libraries(imgui_metal PUBLIC imgui
                PRIVATE "-framework Metal"
                        "-framework QuartzCore"
                        "-framework Foundation")
            target_include_directories(imgui_metal PUBLIC ${_IMGUI_DIR})
            set_source_files_properties(${_IMGUI_DIR}/backends/imgui_impl_metal.mm
                PROPERTIES COMPILE_FLAGS "-fobjc-arc")
        endif()

        # Vulkan backend. Gated on the Vulkan option + find_package(Vulkan)
        # having succeeded (the Vulkan::Vulkan target is what provides both
        # the headers and libvulkan). On machines without the SDK the app
        # wouldn't be functional anyway; we skip imgui_vulkan cleanly.
        if(SINGULARITY_BACKEND_VULKAN AND TARGET Vulkan::Vulkan)
            add_library(imgui_vulkan STATIC
                ${_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
            )
            target_link_libraries(imgui_vulkan
                PUBLIC imgui Vulkan::Vulkan)
            target_include_directories(imgui_vulkan PUBLIC ${_IMGUI_DIR})
        endif()

        # No imgui_wgpu target on the web build. Vendored ImGui 1.91.5's
        # imgui_impl_wgpu.cpp doesn't compile against current
        # emdawnwebgpu (renamed WGPU types). The web demo uses a DOM
        # overlay instead; see web/shell.html.
    endif()
endif()
