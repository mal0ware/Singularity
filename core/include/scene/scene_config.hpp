// core/include/scene/scene_config.hpp
//
// Lightweight scene-configuration loader for the CLI. Reads a minimal
// ``key = value`` text format — one parameter per line, plain ASCII, # for
// comments — into a strongly-typed ``SceneConfig`` struct. Deliberately not
// JSON: the rendering CLI has a flat parameter surface (spin, mass, disc
// edges, integrator step, …) and a line-based format lets us stay
// dependency-free while still supporting presets that a user can edit in a
// text editor and diff cleanly in version control.
//
// Example file (any subset of the keys is valid — missing keys keep their
// ``SceneConfig`` defaults):
//
//     # Kerr near-extremal with a bright disc
//     mass          = 1.0
//     spin          = 0.94
//     disc_r_inner  = 2.04     # ISCO for this spin
//     disc_r_outer  = 25.0
//     camera_fov    = 55.0
//     h_step        = 0.05
//
// When Phase 7 starts wiring this into the real renderer the JSON path in
// docs/PRD.md remains the goal — this simple format is a stepping stone.

#ifndef SINGULARITY_SCENE_SCENE_CONFIG_HPP
#define SINGULARITY_SCENE_SCENE_CONFIG_HPP

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace singularity::scene {

struct SceneConfig {
    // Black hole.
    float mass = 1.0f;  // geometrized units; horizon at M+√(M²-a²)
    float spin = 0.0f;  // a/M, clamped to [0, 1]

    // Accretion disc.
    float disc_r_inner = -1.0f;  // M; -1 means ISCO at the scene's spin
    float disc_r_outer = 20.0f;  // M
    bool disc_doppler_on = true;
    bool disc_redshift_on = true;

    // Camera (placeholder — wired into Phase 2 renderer).
    float camera_fov_deg = 60.0f;
    float camera_distance = 30.0f;  // M
    float camera_elevation = 0.0f;  // radians from equatorial

    // Integrator.
    float h_step = 0.1f;  // affine-parameter step, M
    int max_steps = 5000;
};

enum class SceneLoadStatus {
    Ok,
    FileNotFound,
    ParseError,
    UnknownKey,
};

struct SceneLoadResult {
    SceneLoadStatus status = SceneLoadStatus::Ok;
    std::string error_line;  // original line if status != Ok
    int line_number = 0;     // 1-indexed
};

// Internal helpers ---------------------------------------------------------

namespace detail {

inline void trim(std::string& s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

inline bool parse_bool(const std::string& v, bool& out) {
    std::string lower;
    lower.reserve(v.size());
    for (char c : v)
        lower.push_back(char(std::tolower((unsigned char)c)));
    if (lower == "true" || lower == "on" || lower == "yes" || lower == "1") {
        out = true;
        return true;
    }
    if (lower == "false" || lower == "off" || lower == "no" || lower == "0") {
        out = false;
        return true;
    }
    return false;
}

inline bool parse_float(const std::string& v, float& out) {
    if (v.empty())
        return false;
    char* end = nullptr;
    const float parsed = std::strtof(v.c_str(), &end);
    if (end == v.c_str())
        return false;
    // Allow trailing whitespace (already trimmed) but fail on non-space junk.
    while (end && *end) {
        if (!std::isspace((unsigned char)*end))
            return false;
        ++end;
    }
    out = parsed;
    return true;
}

inline bool parse_int(const std::string& v, int& out) {
    if (v.empty())
        return false;
    char* end = nullptr;
    const long parsed = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str())
        return false;
    while (end && *end) {
        if (!std::isspace((unsigned char)*end))
            return false;
        ++end;
    }
    out = int(parsed);
    return true;
}

}  // namespace detail

inline SceneLoadResult load_scene_config_from_stream(std::istream& in, SceneConfig& out) {
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        // Strip comments.
        auto hash = line.find('#');
        if (hash != std::string::npos)
            line.erase(hash);
        detail::trim(line);
        if (line.empty())
            continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) {
            return {SceneLoadStatus::ParseError, line, line_no};
        }
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        detail::trim(key);
        detail::trim(val);
        if (key.empty() || val.empty()) {
            return {SceneLoadStatus::ParseError, line, line_no};
        }

        bool ok = true;
        if (key == "mass")
            ok = detail::parse_float(val, out.mass);
        else if (key == "spin")
            ok = detail::parse_float(val, out.spin);
        else if (key == "disc_r_inner")
            ok = detail::parse_float(val, out.disc_r_inner);
        else if (key == "disc_r_outer")
            ok = detail::parse_float(val, out.disc_r_outer);
        else if (key == "disc_doppler_on")
            ok = detail::parse_bool(val, out.disc_doppler_on);
        else if (key == "disc_redshift_on")
            ok = detail::parse_bool(val, out.disc_redshift_on);
        else if (key == "camera_fov_deg")
            ok = detail::parse_float(val, out.camera_fov_deg);
        else if (key == "camera_distance")
            ok = detail::parse_float(val, out.camera_distance);
        else if (key == "camera_elevation")
            ok = detail::parse_float(val, out.camera_elevation);
        else if (key == "h_step")
            ok = detail::parse_float(val, out.h_step);
        else if (key == "max_steps")
            ok = detail::parse_int(val, out.max_steps);
        else {
            return {SceneLoadStatus::UnknownKey, key, line_no};
        }

        if (!ok) {
            return {SceneLoadStatus::ParseError, line, line_no};
        }
    }
    return {SceneLoadStatus::Ok, {}, line_no};
}

inline SceneLoadResult load_scene_config_from_file(const std::string& path, SceneConfig& out) {
    std::ifstream in(path);
    if (!in) {
        return {SceneLoadStatus::FileNotFound, path, 0};
    }
    return load_scene_config_from_stream(in, out);
}

}  // namespace singularity::scene

#endif  // SINGULARITY_SCENE_SCENE_CONFIG_HPP
