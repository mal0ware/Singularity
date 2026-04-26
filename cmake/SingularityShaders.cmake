# ----------------------------------------------------------------------------
# Singularity — shader compilation helpers
#
# Provides:
#   add_metal_library(<target> SOURCES <.metal files...> [INCLUDE_DIRS ...])
#       Compiles .metal -> .air -> default.metallib and packages it as a
#       resource of <target>. Requires xcrun from a full Xcode install.
#
#   add_vulkan_shaders(<target> SOURCES <.hlsl files...> [INCLUDE_DIRS ...])
#       Compiles .hlsl -> .spv via DXC (ships with the Vulkan SDK) targeting
#       Vulkan 1.3, and declares them as generated resources for <target>.
#
# Both helpers mirror the patterns in docs/ARCHITECTURE.md §9.2.
# ----------------------------------------------------------------------------

function(add_metal_library TARGET)
    if(NOT APPLE)
        message(FATAL_ERROR "add_metal_library requires an Apple platform.")
    endif()

    cmake_parse_arguments(ARG "" "OUTPUT_NAME" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_metal_library(${TARGET}): SOURCES required.")
    endif()
    if(NOT ARG_OUTPUT_NAME)
        set(ARG_OUTPUT_NAME "default.metallib")
    endif()

    find_program(XCRUN xcrun REQUIRED)

    # Probe for the `metal` compiler once per configure. Command Line Tools
    # alone is *not* enough — the `metal` binary ships only with full Xcode.
    # When missing we emit a stub target that prints a clear actionable error
    # at build time instead of letting `xcrun` splatter its own diagnostic
    # across several parallel shader compiles.
    if(NOT DEFINED CACHE{SINGULARITY_METAL_TOOL_OK})
        execute_process(
            COMMAND ${XCRUN} -f metal
            RESULT_VARIABLE _metal_rc
            OUTPUT_QUIET ERROR_QUIET)
        if(_metal_rc EQUAL 0)
            set(SINGULARITY_METAL_TOOL_OK TRUE CACHE INTERNAL "metal compiler present")
        else()
            set(SINGULARITY_METAL_TOOL_OK FALSE CACHE INTERNAL "metal compiler missing")
        endif()
    endif()

    if(NOT SINGULARITY_METAL_TOOL_OK)
        set(_msg "Metal shader compiler (xcrun metal) not found — install full Xcode from the App Store, then 'sudo xcode-select -s /Applications/Xcode.app/Contents/Developer'. The native backend will still link; default.metallib is not produced, so the app will fail at startup with 'metallib not found' until Xcode is installed and the shaders are rebuilt.")
        message(WARNING "${_msg}")
        # Empty-succeed stub so the rest of the build completes. This lets the
        # user verify the Obj-C++ + C++ paths compile while they wait for the
        # Xcode download to finish, and the app's own startup error makes the
        # missing-library condition obvious and actionable.
        add_custom_target(${TARGET}
            COMMAND ${CMAKE_COMMAND} -E echo
                    "[singularity] metallib skipped — Xcode not installed."
        )
        set_target_properties(${TARGET} PROPERTIES
            SINGULARITY_METALLIB_PATH ""
            SINGULARITY_METAL_MISSING TRUE
        )
        return()
    endif()

    set(_include_flags "")
    foreach(dir ${ARG_INCLUDE_DIRS})
        list(APPEND _include_flags "-I" "${dir}")
    endforeach()

    set(_air_files "")
    foreach(SHADER ${ARG_SOURCES})
        get_filename_component(NAME ${SHADER} NAME_WE)
        get_filename_component(ABS ${SHADER} ABSOLUTE)
        set(_air "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air")
        add_custom_command(
            OUTPUT ${_air}
            COMMAND ${XCRUN} -sdk macosx metal -c ${ABS}
                    -o ${_air}
                    -std=metal3.1
                    -gline-tables-only
                    ${_include_flags}
            DEPENDS ${ABS}
            COMMENT "MSL ${NAME}.metal -> ${NAME}.air"
            VERBATIM
        )
        list(APPEND _air_files ${_air})
    endforeach()

    set(_metallib "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT_NAME}")
    add_custom_command(
        OUTPUT ${_metallib}
        COMMAND ${XCRUN} -sdk macosx metallib ${_air_files} -o ${_metallib}
        DEPENDS ${_air_files}
        COMMENT "Linking ${ARG_OUTPUT_NAME}"
        VERBATIM
    )

    add_custom_target(${TARGET} ALL DEPENDS ${_metallib})
    set_target_properties(${TARGET} PROPERTIES
        SINGULARITY_METALLIB_PATH "${_metallib}"
    )
endfunction()

function(add_vulkan_shaders TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_vulkan_shaders(${TARGET}): SOURCES required.")
    endif()

    if(NOT DEFINED DXC_EXECUTABLE OR DXC_EXECUTABLE STREQUAL "")
        find_program(DXC_EXECUTABLE dxc
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
            DOC "DirectX Shader Compiler (ships with Vulkan SDK)"
        )
    endif()
    if(NOT DXC_EXECUTABLE)
        message(FATAL_ERROR
            "DXC not found. Install the Vulkan SDK or pass "
            "-DDXC_EXECUTABLE=/path/to/dxc.")
    endif()

    set(_include_flags "")
    foreach(dir ${ARG_INCLUDE_DIRS})
        list(APPEND _include_flags "-I${dir}")
    endforeach()

    set(_spv_files "")
    foreach(SHADER ${ARG_SOURCES})
        get_filename_component(NAME ${SHADER} NAME_WE)
        get_filename_component(ABS ${SHADER} ABSOLUTE)
        # Profile inferred from filename: *_kernel.hlsl / cube.hlsl use cs/vs/ps
        # per the shader's entrypoint. For Phase 0 we default to compute; callers
        # can override per-file by passing `PROFILE_<NAME>`.
        set(_profile "cs_6_6")
        set(_entry "main")
        if(DEFINED PROFILE_${NAME})
            set(_profile ${PROFILE_${NAME}})
        endif()
        if(DEFINED ENTRY_${NAME})
            set(_entry ${ENTRY_${NAME}})
        endif()
        set(_spv "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv")
        add_custom_command(
            OUTPUT ${_spv}
            COMMAND ${DXC_EXECUTABLE}
                    -T ${_profile}
                    -E ${_entry}
                    -spirv
                    -fspv-target-env=vulkan1.3
                    ${_include_flags}
                    ${ABS}
                    -Fo ${_spv}
            DEPENDS ${ABS}
            COMMENT "HLSL ${NAME}.hlsl -> ${NAME}.spv (${_profile}/${_entry})"
            VERBATIM
        )
        list(APPEND _spv_files ${_spv})
    endforeach()

    add_custom_target(${TARGET} ALL DEPENDS ${_spv_files})
    set_target_properties(${TARGET} PROPERTIES
        SINGULARITY_SPV_FILES "${_spv_files}"
    )
endfunction()
