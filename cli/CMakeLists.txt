# ----------------------------------------------------------------------------
# singularity_cli — headless driver exercised by the Python verification
# harness. Links only against singularity::core so it builds without any GPU
# backend or window-system dependency.
# ----------------------------------------------------------------------------

set(CLI_SOURCES
    main.cpp
)

add_executable(singularity_cli ${CLI_SOURCES})

target_link_libraries(singularity_cli PRIVATE singularity::core)

target_include_directories(singularity_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# stb_image_write.h triggers legitimate but noisy warnings under our
# strict flags (missing field initializers, one sprintf usage). It is
# included from exactly one TU; silence warnings for that file only.
if(NOT MSVC)
    set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS
        "-Wno-missing-field-initializers -Wno-deprecated-declarations")
endif()

# ctest-level smoke for the benchmark mode: fastest-failing signal if the
# integrator dispatch wiring breaks. Deeper coverage (JSON schema +
# total_steps determinism) lives in verification/test_smoke.py.
include(CTest)
add_test(NAME singularity_cli_benchmark_smoke
    COMMAND singularity_cli --mode benchmark --n-rays 4 --max-steps 100)
