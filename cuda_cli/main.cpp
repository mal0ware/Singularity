// cuda_cli/main.cpp
//
// Entrypoint for singularity_cuda_cli — the offline CUDA renderer. Loads a
// scene config, brings up CudaBackend, renders one or more frames, writes
// PNGs. Multi-frame mode (`--frames N --output-pattern path_%04d.png`)
// orbits the camera by a full 2π in azimuth across the sequence; the PNG
// sequence can either be encoded out-of-band by the user (the documented
// ffmpeg one-liner in `--help`) or, opt-in, by passing `--encode-mp4 PATH`
// to fork a single ffmpeg subprocess after the last PNG is written. The
// encode is opt-in so the default path stays statically-linkable +
// side-effect-free for CI / scripted use.

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "cuda_backend.hpp"
#include "render_backend.hpp"
#include "scene/scene_config.hpp"
#include "stb_image_write.h"

namespace {

void print_help() {
    std::puts("singularity_cuda_cli — offline CUDA renderer\n"
              "\n"
              "Usage:\n"
              "  singularity_cuda_cli [--scene PATH] [--output PATH] [--res WxH]\n"
              "                       [--samples-per-pixel N]\n"
              "                       [--camera-distance M] [--camera-elevation R]\n"
              "                       [--camera-azimuth R] [--fov-y-deg D]\n"
              "                       [--frames N --output-pattern PATTERN]\n"
              "\n"
              "Options:\n"
              "  --scene PATH         Scene config (same format as singularity_cli --scene).\n"
              "  --output PATH        PNG output path (default: cuda_render.png).\n"
              "                       Ignored in multi-frame mode.\n"
              "  --res WxH            Output resolution (default: 1280x720)\n"
              "  --samples-per-pixel N  Halton(2,3) subpixel samples (1..1024, default 1).\n"
              "                         256 is the documented offline-quality headline.\n"
              "  --camera-distance M  Orbital radius in M (default 30)\n"
              "  --camera-elevation R Elevation from equator in rad (default 0.15)\n"
              "  --camera-azimuth R   Starting azimuth in rad (default 0.0)\n"
              "  --fov-y-deg D        Vertical FOV in degrees (default 60)\n"
              "  --frames N           Render N frames orbiting 2π in azimuth (default 1).\n"
              "  --output-pattern P   printf-style path with one %d (or %0Nd) placeholder for the\n"
              "                       frame index, e.g. frames/frame_%04d.png. Required when\n"
              "                       --frames > 1.\n"
              "  --encode-mp4 PATH    After the last frame is written, fork ffmpeg to encode\n"
              "                       the sequence to PATH at 30 fps, libx264, yuv420p, crf=18.\n"
              "                       Requires --frames > 1 and ffmpeg on PATH. Opt-in; the\n"
              "                       default PNG-only path stays subprocess-free.\n"
              "  --mp4-fps N          Encoder framerate (default 30). Only honoured with\n"
              "                       --encode-mp4.\n"
              "\n"
              "Manual encode (equivalent to --encode-mp4):\n"
              "  ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 \\\n"
              "         -pix_fmt yuv420p -crf 18 kerr.mp4\n");
}

// Same orbital-camera basis the live app uses (app/app_shell.cpp::compute_basis):
// row 0 = right, row 1 = up, row 2 = -forward. Centered on origin.
void build_orbital_camera(singularity::CameraState& cam,
                          float distance_M,
                          float elevation_rad,
                          float azimuth_rad,
                          float fov_y_deg) {
    const float ce = std::cos(elevation_rad);
    const float se = std::sin(elevation_rad);
    const float ca = std::cos(azimuth_rad);
    const float sa = std::sin(azimuth_rad);

    cam.position[0] = distance_M * ce * ca;
    cam.position[1] = distance_M * ce * sa;
    cam.position[2] = distance_M * se;

    const float inv = 1.0f / distance_M;
    const float fwd[3] = {-cam.position[0] * inv, -cam.position[1] * inv, -cam.position[2] * inv};

    float rt[3] = {fwd[1], -fwd[0], 0.0f};
    const float rt_len = std::sqrt(rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2]);
    const float inv_r = (rt_len > 1e-6f) ? 1.0f / rt_len : 1.0f;
    rt[0] *= inv_r;
    rt[1] *= inv_r;
    rt[2] *= inv_r;

    const float up[3] = {
        fwd[1] * rt[2] - fwd[2] * rt[1],
        fwd[2] * rt[0] - fwd[0] * rt[2],
        fwd[0] * rt[1] - fwd[1] * rt[0],
    };

    cam.basis[0] = rt[0];
    cam.basis[1] = rt[1];
    cam.basis[2] = rt[2];
    cam.basis[3] = up[0];
    cam.basis[4] = up[1];
    cam.basis[5] = up[2];
    cam.basis[6] = -fwd[0];
    cam.basis[7] = -fwd[1];
    cam.basis[8] = -fwd[2];

    cam.fov_y_radians = fov_y_deg * (3.14159265358979323846f / 180.0f);
}

}  // namespace

int main(int argc, char** argv) {
    std::string scene_path;
    std::string output_path = "cuda_render.png";
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    float camera_distance = 30.0f;
    float camera_elevation = 0.15f;
    float camera_azimuth = 0.0f;
    float fov_y_deg = 60.0f;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t frames = 1;
    std::string output_pattern;
    std::string encode_mp4_path;
    std::uint32_t mp4_fps = 30;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--help" || a == "-h") {
            print_help();
            return 0;
        }
        if (a == "--scene" && i + 1 < argc) {
            scene_path = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--res" && i + 1 < argc) {
            const char* v = argv[++i];
            const char* x = std::strchr(v, 'x');
            if (x == nullptr) {
                std::fprintf(stderr, "singularity_cuda_cli: --res expects WxH, got %s\n", v);
                return 2;
            }
            width = static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
            height = static_cast<std::uint32_t>(std::strtoul(x + 1, nullptr, 10));
            if (width < 4 || height < 4) {
                std::fprintf(stderr, "singularity_cuda_cli: --res too small\n");
                return 2;
            }
        } else if (a == "--camera-distance" && i + 1 < argc) {
            camera_distance = std::strtof(argv[++i], nullptr);
        } else if (a == "--camera-elevation" && i + 1 < argc) {
            camera_elevation = std::strtof(argv[++i], nullptr);
        } else if (a == "--camera-azimuth" && i + 1 < argc) {
            camera_azimuth = std::strtof(argv[++i], nullptr);
        } else if (a == "--fov-y-deg" && i + 1 < argc) {
            fov_y_deg = std::strtof(argv[++i], nullptr);
        } else if (a == "--samples-per-pixel" && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 1ul || v > 1024ul) {
                std::fprintf(stderr,
                             "singularity_cuda_cli: --samples-per-pixel must be in [1, 1024]\n");
                return 2;
            }
            samples_per_pixel = static_cast<std::uint32_t>(v);
        } else if (a == "--frames" && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 1ul || v > 100000ul) {
                std::fprintf(stderr, "singularity_cuda_cli: --frames must be in [1, 100000]\n");
                return 2;
            }
            frames = static_cast<std::uint32_t>(v);
        } else if (a == "--output-pattern" && i + 1 < argc) {
            output_pattern = argv[++i];
        } else if (a == "--encode-mp4" && i + 1 < argc) {
            encode_mp4_path = argv[++i];
        } else if (a == "--mp4-fps" && i + 1 < argc) {
            const unsigned long v = std::strtoul(argv[++i], nullptr, 10);
            if (v < 1ul || v > 240ul) {
                std::fprintf(stderr, "singularity_cuda_cli: --mp4-fps must be in [1, 240]\n");
                return 2;
            }
            mp4_fps = static_cast<std::uint32_t>(v);
        } else {
            std::fprintf(
                stderr, "singularity_cuda_cli: unknown argument: %.*s\n", int(a.size()), a.data());
            print_help();
            return 2;
        }
    }

    singularity::Scene scene{};
    if (!scene_path.empty()) {
        singularity::scene::SceneConfig cfg;
        const auto res = singularity::scene::load_scene_config_from_file(scene_path, cfg);
        using S = singularity::scene::SceneLoadStatus;
        if (res.status != S::Ok) {
            const char* reason = res.status == S::FileNotFound ? "file-not-found"
                                 : res.status == S::UnknownKey ? "unknown-key"
                                                               : "parse-error";
            std::fprintf(stderr,
                         "singularity_cuda_cli: scene load failed (%s, line %d): %s\n",
                         reason,
                         res.line_number,
                         res.error_line.c_str());
            return 2;
        }
        scene.mass_solar = cfg.mass;
        scene.spin_a_over_M = cfg.spin;
        scene.disc_inner_M = cfg.disc_r_inner;
        scene.disc_outer_M = cfg.disc_r_outer;
        scene.disc_doppler_on = cfg.disc_doppler_on;
        scene.disc_redshift_on = cfg.disc_redshift_on;
        scene.metric = (cfg.spin > 0.0f) ? singularity::Scene::MetricType::Kerr
                                         : singularity::Scene::MetricType::Schwarzschild;
    }
    scene.render_supersample = samples_per_pixel;

    if (frames > 1 && output_pattern.empty()) {
        std::fprintf(stderr,
                     "singularity_cuda_cli: --frames > 1 requires --output-pattern with a "
                     "printf-style %%d placeholder, e.g. frames/frame_%%04d.png\n");
        return 2;
    }
    if (frames > 1 && output_pattern.find('%') == std::string::npos) {
        std::fprintf(stderr,
                     "singularity_cuda_cli: --output-pattern must contain a printf placeholder "
                     "(%%d / %%04d)\n");
        return 2;
    }
    if (!encode_mp4_path.empty() && frames < 2) {
        std::fprintf(stderr,
                     "singularity_cuda_cli: --encode-mp4 requires --frames > 1 (a sequence to "
                     "encode)\n");
        return 2;
    }

    singularity::RenderConfig rcfg{};
    rcfg.width = width;
    rcfg.height = height;

    singularity::CudaBackend backend;
    if (!backend.initialize(singularity::WindowHandle{}, rcfg)) {
        std::fprintf(stderr, "singularity_cuda_cli: backend initialize failed\n");
        return 1;
    }

    constexpr float kTwoPi = 6.28318530717958647693f;

    for (std::uint32_t f = 0; f < frames; ++f) {
        // Span 2π across [0, frames) so the last frame is just before
        // wrapping back to frame 0 — produces a seamless looping orbit.
        const float az = camera_azimuth + (frames > 1 ? (kTwoPi * float(f) / float(frames)) : 0.0f);
        singularity::CameraState camera{};
        build_orbital_camera(camera, camera_distance, camera_elevation, az, fov_y_deg);

        backend.render_frame(scene, camera);
        const auto img = backend.capture_frame();
        if (img.pixels_rgba.empty() || img.width == 0 || img.height == 0) {
            std::fprintf(stderr, "singularity_cuda_cli: empty capture (frame %u)\n", f);
            backend.shutdown();
            return 1;
        }

        char path_buf[1024];
        const char* path_cstr;
        if (frames > 1) {
            // printf-style format gates above ensure exactly one %d-style spec.
            std::snprintf(path_buf, sizeof(path_buf), output_pattern.c_str(), int(f));
            path_cstr = path_buf;
        } else {
            path_cstr = output_path.c_str();
        }

        if (!stbi_write_png(path_cstr,
                            int(img.width),
                            int(img.height),
                            4,
                            img.pixels_rgba.data(),
                            int(img.width) * 4)) {
            std::fprintf(stderr, "singularity_cuda_cli: failed to write '%s'\n", path_cstr);
            backend.shutdown();
            return 1;
        }
        if (frames > 1) {
            std::printf("wrote %s (%ux%u, frame %u/%u, CUDA)\n",
                        path_cstr,
                        img.width,
                        img.height,
                        f + 1,
                        frames);
            std::fflush(stdout);
        } else {
            std::printf("wrote %s (%ux%u, CUDA)\n", path_cstr, img.width, img.height);
        }
    }

    backend.shutdown();

    if (!encode_mp4_path.empty()) {
        // Hand the same printf-pattern cuda_cli wrote with through to ffmpeg
        // -- ffmpeg's image-sequence demuxer accepts the same %0Nd syntax.
        // std::system rather than posix_spawn / CreateProcess to keep the
        // platform code minimal; this is opt-in and the user authored the
        // output paths, so command-injection isn't a meaningful threat.
        // Quoting the two paths covers the "sane filename" cases (spaces,
        // punctuation); files containing literal `"` would still break and
        // are out of scope for an offline render driver.
        char cmd[2048];
        const int written = std::snprintf(cmd,
                                          sizeof(cmd),
                                          "ffmpeg -y -hide_banner -loglevel warning "
                                          "-framerate %u -i \"%s\" "
                                          "-c:v libx264 -pix_fmt yuv420p -crf 18 \"%s\"",
                                          mp4_fps,
                                          output_pattern.c_str(),
                                          encode_mp4_path.c_str());
        if (written < 0 || written >= int(sizeof(cmd))) {
            std::fprintf(stderr,
                         "singularity_cuda_cli: ffmpeg command exceeded buffer (paths too long)\n");
            return 1;
        }
        std::printf("encoding: %s\n", cmd);
        std::fflush(stdout);
        const int rc = std::system(cmd);
        if (rc != 0) {
            std::fprintf(stderr,
                         "singularity_cuda_cli: ffmpeg exited %d -- is `ffmpeg` on PATH? "
                         "(see manual one-liner in --help)\n",
                         rc);
            return 1;
        }
        std::printf("wrote %s (%u frames at %u fps, libx264 crf=18 yuv420p)\n",
                    encode_mp4_path.c_str(),
                    frames,
                    mp4_fps);
    }

    return 0;
}
