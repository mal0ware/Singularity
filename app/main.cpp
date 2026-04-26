// app/main.cpp
//
// Platform entry. Dispatches to the windowed shell or the `--capture`
// headless renderer depending on CLI flags.

#include <SDL3/SDL_main.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "app_shell.hpp"

int main(int argc, char** argv) {
    singularity::app::AppShellConfig cfg{};

    // Crude CLI; swap out for a real parser when flags grow.
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            std::printf("Usage: singularity [options]\n"
                        "  --res=WxH              window / capture resolution (default 1280x720)\n"
                        "  --no-vsync             disable vsync on the window path\n"
                        "  --kerr                 start in Kerr mode\n"
                        "  --spin=A               Kerr spin a/M (implies --kerr)\n"
                        "  --capture=OUT.png      headless: render one frame, write PNG, exit\n"
                        "  --capture-ss=N         capture supersampling (1|2|4, default 2)\n"
                        "  --capture-dist=M       camera distance in M (default 30)\n"
                        "  --capture-elev=RAD     camera elevation in rad (default 0.15)\n");
            return 0;
        } else if (std::strcmp(a, "--kerr") == 0) {
            cfg.start_kerr = true;
        } else if (std::strncmp(a, "--spin=", 7) == 0) {
            cfg.start_spin_a_over_M = (float)std::atof(a + 7);
            cfg.capture_spin_a_over_M = cfg.start_spin_a_over_M;
            cfg.start_kerr = true;
        } else if (std::strncmp(a, "--res=", 6) == 0) {
            const char* x = std::strchr(a + 6, 'x');
            if (x) {
                cfg.width = std::atoi(a + 6);
                cfg.height = std::atoi(x + 1);
            }
        } else if (std::strcmp(a, "--no-vsync") == 0) {
            cfg.vsync = false;
        } else if (std::strncmp(a, "--capture=", 10) == 0) {
            cfg.capture_path = std::string(a + 10);
        } else if (std::strncmp(a, "--capture-ss=", 13) == 0) {
            cfg.capture_supersample = std::atoi(a + 13);
        } else if (std::strncmp(a, "--capture-dist=", 15) == 0) {
            cfg.capture_distance_M = (float)std::atof(a + 15);
        } else if (std::strncmp(a, "--capture-elev=", 15) == 0) {
            cfg.capture_elevation_rad = (float)std::atof(a + 15);
        } else {
            std::fprintf(stderr, "singularity: unknown flag '%s' (try --help)\n", a);
            return 2;
        }
    }

    if (!cfg.capture_path.empty()) {
        return singularity::app::run_capture(cfg);
    }
    return singularity::app::run(cfg);
}
