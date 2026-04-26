# ----------------------------------------------------------------------------
# C++ unit tests — Catch2 v3 (single-header, vendored in third_party/catch2).
# Runs on the CPU build of the physics core to catch logic bugs before they
# reach any GPU backend.
# ----------------------------------------------------------------------------

set(TEST_SOURCES
    test_schwarzschild.cpp
    test_integrator.cpp
    test_kerr.cpp
    test_conserved.cpp
    test_redshift.cpp
    test_disc.cpp
    test_disc_intersection.cpp
    test_kerr_geodesic.cpp
    test_kerr_hamilton.cpp
    test_scene_config.cpp
)

add_executable(singularity_tests
    ${TEST_SOURCES}
    ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
)

target_link_libraries(singularity_tests PRIVATE singularity::core)

target_include_directories(singularity_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/catch2
)

# Catch2's own amalgamated source trips a few of our strict warnings; we
# don't want to police upstream's style, so silence them for that TU only.
if(NOT MSVC)
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
        PROPERTIES COMPILE_FLAGS "-w"
    )
else()
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/third_party/catch2/catch_amalgamated.cpp
        PROPERTIES COMPILE_FLAGS "/w"
    )
endif()

include(CTest)
add_test(NAME singularity_tests COMMAND singularity_tests)
