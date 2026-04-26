# ----------------------------------------------------------------------------
# singularity — cross-platform desktop application shell
# ----------------------------------------------------------------------------

set(APP_SOURCES
    main.cpp
    app_shell.cpp
    app_shell.hpp
)

add_executable(singularity ${APP_SOURCES})
target_include_directories(singularity PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/stb
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# On Apple, app_shell.cpp pulls in Obj-C headers (CAMetalLayer, ImGui Metal
# backend bridging). Compile it as Objective-C++ via a per-file language
# override so main.cpp and any future vanilla-C++ modules stay plain CXX.
if(APPLE)
    set_source_files_properties(app_shell.cpp PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-fobjc-arc"
    )
endif()

target_link_libraries(singularity
    PRIVATE
        singularity::render
)

# The concrete backend and SDL3 flow in through singularity::render and
# imgui_sdl3 respectively. Direct links to those targets here would cause
# duplicate-library warnings from ld. The only pieces the app needs that
# aren't pulled in transitively are the ImGui backends — link those.
if(TARGET imgui_sdl3)
    target_link_libraries(singularity PRIVATE imgui_sdl3)
elseif(TARGET imgui)
    target_link_libraries(singularity PRIVATE imgui)
endif()

if(TARGET imgui_metal)
    target_link_libraries(singularity PRIVATE imgui_metal)
endif()

if(TARGET imgui_vulkan)
    target_link_libraries(singularity PRIVATE imgui_vulkan)
endif()

# The Vulkan app path uses raw VkCreateDescriptorPool/VkDestroyDescriptorPool
# directly (for the ImGui font/texture pool), so link the loader when present.
# imgui_vulkan already transitively brings Vulkan::Vulkan in via PUBLIC link,
# but we keep this explicit so a future refactor that drops imgui_vulkan
# from the link list doesn't silently break the Vulkan-only path.
if(SINGULARITY_BACKEND_VULKAN AND TARGET Vulkan::Vulkan)
    target_link_libraries(singularity PRIVATE Vulkan::Vulkan)
endif()

# ----------------------------------------------------------------------------
# macOS: build a proper .app bundle with default.metallib in Resources/
# and an Info.plist generated from the CMake template. Ad-hoc codesign by
# default; drop a real Developer ID identity into
# SINGULARITY_MACOS_CODESIGN_IDENTITY to flip to a notarizable build.
# ----------------------------------------------------------------------------
if(APPLE)
    set_target_properties(singularity PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME            "Singularity"
        MACOSX_BUNDLE_GUI_IDENTIFIER         "dev.mal.singularity"
        MACOSX_BUNDLE_BUNDLE_VERSION         "${PROJECT_VERSION}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING   "${PROJECT_VERSION}"
        MACOSX_BUNDLE_INFO_PLIST             ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in
        XCODE_ATTRIBUTE_ENABLE_HARDENED_RUNTIME YES
    )

    # Stage the compiled metallib into Contents/Resources/. Using an
    # OUTPUT-based custom command rather than a POST_BUILD one so CMake
    # tracks the dependency on the metallib source file — POST_BUILD only
    # fires when the executable itself re-links, which means shader-only
    # edits would leave a stale metallib in the bundle.
    if(DEFINED SINGULARITY_METALLIB_FILE AND SINGULARITY_METALLIB_FILE)
        set(_staged_metallib
            "$<TARGET_BUNDLE_CONTENT_DIR:singularity>/Resources/default.metallib")
        # A stamp file forces CMake to actually invoke the command whenever
        # the metallib changes; generator expressions in OUTPUT get eager-
        # evaluated so we can't use the real bundle path there.
        set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/metallib.stamp")
        add_custom_command(
            OUTPUT ${_stamp}
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    "$<TARGET_BUNDLE_CONTENT_DIR:singularity>/Resources"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${SINGULARITY_METALLIB_FILE}"
                    ${_staged_metallib}
            COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
            DEPENDS "${SINGULARITY_METALLIB_FILE}" singularity_metallib
            COMMENT "Staging default.metallib into bundle Resources/"
            VERBATIM
        )
        add_custom_target(singularity_stage_metallib ALL DEPENDS ${_stamp})
        add_dependencies(singularity_stage_metallib singularity)
    endif()

    # Optional Developer ID signing. When SINGULARITY_MACOS_CODESIGN_IDENTITY
    # is set (e.g. "Developer ID Application: Malachi ..."), sign with
    # hardened runtime + entitlements so the bundle is notarizable. Otherwise
    # fall through to ad-hoc signing (`-s -`) so the bundle at least runs on
    # the developer's own machine.
    set(SINGULARITY_MACOS_CODESIGN_IDENTITY "" CACHE STRING
        "Code-sign identity for the macOS bundle (e.g. 'Developer ID Application: ...'). Empty -> ad-hoc.")
    set(SINGULARITY_MACOS_ENTITLEMENTS
        "${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist" CACHE FILEPATH
        "Entitlements used when a real signing identity is supplied.")

    if(SINGULARITY_MACOS_CODESIGN_IDENTITY)
        add_custom_command(TARGET singularity POST_BUILD
            COMMAND codesign --force --deep
                    --sign "${SINGULARITY_MACOS_CODESIGN_IDENTITY}"
                    --options runtime
                    --entitlements "${SINGULARITY_MACOS_ENTITLEMENTS}"
                    --timestamp
                    "$<TARGET_BUNDLE_DIR:singularity>"
            COMMENT "Signing Singularity.app with '${SINGULARITY_MACOS_CODESIGN_IDENTITY}'"
            VERBATIM
        )
    else()
        add_custom_command(TARGET singularity POST_BUILD
            COMMAND codesign --force --deep --sign -
                    "$<TARGET_BUNDLE_DIR:singularity>"
            COMMENT "Ad-hoc signing Singularity.app (no Developer ID supplied)"
            VERBATIM
        )
    endif()
endif()

if(WIN32)
    set_target_properties(singularity PROPERTIES WIN32_EXECUTABLE TRUE)

    # Install rules for the WIX .msi generator. Everything lands under
    # <install_root>\bin — singularity.exe at the root, compiled SPIR-V in
    # bin\shaders\. The backend's resolve_spv_dir() prefers <exe_dir>\shaders
    # so no environment variable is needed for a packaged install.
    install(TARGETS singularity RUNTIME DESTINATION bin)
    # Explicit file list rather than install(DIRECTORY FILES_MATCHING) —
    # MSVC's multi-config build drops *.tlog intermediates and per-config
    # subdirs (CMakeFiles/, Release/, x64/) alongside the .spv output, and
    # install(DIRECTORY) recreates those empty directory structures in the
    # .msi payload. Naming each artefact keeps the installed bin\shaders\
    # tree clean.
    if(TARGET singularity_spirv AND DEFINED SINGULARITY_SPV_DIR)
        install(FILES
            "${SINGULARITY_SPV_DIR}/geodesic_kernel.spv"
            "${SINGULARITY_SPV_DIR}/bloom_extract.spv"
            "${SINGULARITY_SPV_DIR}/bloom_blur_h.spv"
            "${SINGULARITY_SPV_DIR}/bloom_blur_v.spv"
            "${SINGULARITY_SPV_DIR}/blit_vs.spv"
            "${SINGULARITY_SPV_DIR}/blit_ps.spv"
            DESTINATION bin/shaders)
    endif()
endif()

if(APPLE)
    # Install rule for CPack DragNDrop. Lands Singularity.app at the root
    # of the mounted .dmg so drag-to-Applications is the obvious move.
    install(TARGETS singularity BUNDLE DESTINATION .)
endif()
