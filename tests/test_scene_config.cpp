// tests/test_scene_config.cpp
//
// Exercises the scene-config text-format loader in
// ``core/include/scene/scene_config.hpp``. Covers the happy path, each
// field type (float / int / bool), comment / blank-line handling, and the
// three error categories (unknown key, malformed line, missing file).

#include <catch_amalgamated.hpp>

#include <sstream>

#include "scene/scene_config.hpp"

using namespace singularity::scene;

TEST_CASE("Scene defaults are sensible", "[scene]") {
    SceneConfig cfg;
    REQUIRE(cfg.mass == 1.0f);
    REQUIRE(cfg.spin == 0.0f);
    REQUIRE(cfg.disc_r_inner < 0.0f);  // -1 sentinel → use ISCO
    REQUIRE(cfg.disc_r_outer == 20.0f);
    REQUIRE(cfg.disc_doppler_on);
    REQUIRE(cfg.disc_redshift_on);
    REQUIRE(cfg.max_steps > 0);
}

TEST_CASE("Scene loader parses a full valid config", "[scene]") {
    std::istringstream in("# Kerr near-extremal\n"
                          "mass         = 1.0\n"
                          "spin         = 0.94\n"
                          "disc_r_inner = 2.04\n"
                          "disc_r_outer = 25.0\n"
                          "disc_doppler_on = true\n"
                          "disc_redshift_on = false\n"
                          "camera_fov_deg = 55.0\n"
                          "camera_distance = 40.0\n"
                          "camera_elevation = 0.3\n"
                          "h_step = 0.05\n"
                          "max_steps = 12000\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::Ok);
    REQUIRE(cfg.mass == 1.0f);
    REQUIRE(cfg.spin == 0.94f);
    REQUIRE(cfg.disc_r_inner == 2.04f);
    REQUIRE(cfg.disc_r_outer == 25.0f);
    REQUIRE(cfg.disc_doppler_on);
    REQUIRE_FALSE(cfg.disc_redshift_on);
    REQUIRE(cfg.camera_fov_deg == 55.0f);
    REQUIRE(cfg.camera_distance == 40.0f);
    REQUIRE(cfg.camera_elevation == 0.3f);
    REQUIRE(cfg.h_step == 0.05f);
    REQUIRE(cfg.max_steps == 12000);
}

TEST_CASE("Scene loader ignores comments and blank lines", "[scene]") {
    std::istringstream in("\n"
                          "   # header comment\n"
                          "mass = 2.5   # after-value comment\n"
                          "\n"
                          "spin = 0.5\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::Ok);
    REQUIRE(cfg.mass == 2.5f);
    REQUIRE(cfg.spin == 0.5f);
}

TEST_CASE("Scene loader accepts several bool spellings", "[scene]") {
    for (const char* yes : {"true", "True", "TRUE", "yes", "on", "1"}) {
        SceneConfig cfg;
        std::istringstream in(std::string("disc_doppler_on = ") + yes + "\n");
        REQUIRE(load_scene_config_from_stream(in, cfg).status == SceneLoadStatus::Ok);
        REQUIRE(cfg.disc_doppler_on);
    }
    for (const char* no : {"false", "no", "off", "0"}) {
        SceneConfig cfg;
        std::istringstream in(std::string("disc_doppler_on = ") + no + "\n");
        REQUIRE(load_scene_config_from_stream(in, cfg).status == SceneLoadStatus::Ok);
        REQUIRE_FALSE(cfg.disc_doppler_on);
    }
}

TEST_CASE("Scene loader reports unknown keys precisely", "[scene]") {
    std::istringstream in("mass = 1.0\n"
                          "weird_key = 42\n"
                          "spin = 0.5\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::UnknownKey);
    REQUIRE(r.line_number == 2);
    REQUIRE(r.error_line == "weird_key");
}

TEST_CASE("Scene loader reports malformed lines", "[scene]") {
    std::istringstream in("mass = 1.0\n"
                          "this line has no equals sign\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::ParseError);
    REQUIRE(r.line_number == 2);
}

TEST_CASE("Scene loader rejects non-numeric values", "[scene]") {
    std::istringstream in("mass = one-point-zero\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::ParseError);
}

TEST_CASE("Scene loader reports missing file", "[scene]") {
    SceneConfig cfg;
    const auto r = load_scene_config_from_file("/does/not/exist/scene.conf", cfg);
    REQUIRE(r.status == SceneLoadStatus::FileNotFound);
}

TEST_CASE("Scene loader preserves unspecified defaults", "[scene]") {
    // Only setting mass should leave everything else at its default.
    std::istringstream in("mass = 10.0\n");
    SceneConfig cfg;
    const auto r = load_scene_config_from_stream(in, cfg);
    REQUIRE(r.status == SceneLoadStatus::Ok);
    REQUIRE(cfg.mass == 10.0f);
    REQUIRE(cfg.spin == 0.0f);
    REQUIRE(cfg.disc_r_outer == 20.0f);  // default held
}
