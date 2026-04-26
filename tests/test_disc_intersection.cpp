// tests/test_disc_intersection.cpp
//
// Unit tests for the shared equatorial-plane crossing helper used by both
// cpu-render (Schwarzschild) and kerr-cpu-render sites in cli/main.cpp.
// The helper is a header-only, backend-portable primitive
// (shared_shader/disc_intersection.h) so the Metal/Vulkan/CUDA kernels will
// all call into the exact code exercised here.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "disc_intersection.h"

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfPi = 0.5f * kPi;
}  // namespace

TEST_CASE("no crossing when both samples are on the same side of π/2", "[disc-intersection]") {
    const DiscCrossing above = detect_equatorial_crossing(kHalfPi - 0.3f, kHalfPi - 0.05f);
    REQUIRE_FALSE(above.crossed);

    const DiscCrossing below = detect_equatorial_crossing(kHalfPi + 0.2f, kHalfPi + 0.4f);
    REQUIRE_FALSE(below.crossed);
}

TEST_CASE("straddling π/2 is a crossing and the fraction interpolates linearly",
          "[disc-intersection]") {
    // Symmetric straddle: half-way through the step lands on π/2.
    const float theta_prev = kHalfPi - 0.1f;
    const float theta_curr = kHalfPi + 0.1f;
    const DiscCrossing xing = detect_equatorial_crossing(theta_prev, theta_curr);

    REQUIRE(xing.crossed);
    REQUIRE(xing.frac == Catch::Approx(0.5f).margin(1e-6f));

    // Off-centre straddle: 25 % above, 75 % below → frac = 0.25.
    const DiscCrossing skew = detect_equatorial_crossing(kHalfPi - 0.04f, kHalfPi + 0.12f);
    REQUIRE(skew.crossed);
    REQUIRE(skew.frac == Catch::Approx(0.25f).margin(1e-6f));
}

TEST_CASE("crossing detection is symmetric under reversal", "[disc-intersection]") {
    // A ray going north-to-south and its time-reversed south-to-north
    // partner must both register a crossing with complementary fractions.
    const float a = kHalfPi - 0.2f;
    const float b = kHalfPi + 0.3f;
    const DiscCrossing fwd = detect_equatorial_crossing(a, b);
    const DiscCrossing rev = detect_equatorial_crossing(b, a);
    REQUIRE(fwd.crossed);
    REQUIRE(rev.crossed);
    REQUIRE((fwd.frac + rev.frac) == Catch::Approx(1.0f).margin(1e-6f));
}

TEST_CASE("lerp_scalar recovers endpoints and the midpoint", "[disc-intersection]") {
    REQUIRE(lerp_scalar(10.0f, 20.0f, 0.0f) == Catch::Approx(10.0f));
    REQUIRE(lerp_scalar(10.0f, 20.0f, 1.0f) == Catch::Approx(20.0f));
    REQUIRE(lerp_scalar(10.0f, 20.0f, 0.5f) == Catch::Approx(15.0f));
}

TEST_CASE("annulus test is inclusive at both rims", "[disc-intersection]") {
    constexpr float r_in = 6.0f;
    constexpr float r_out = 20.0f;

    REQUIRE(in_disc_annulus(r_in, r_in, r_out));
    REQUIRE(in_disc_annulus(r_out, r_in, r_out));
    REQUIRE(in_disc_annulus(12.0f, r_in, r_out));

    REQUIRE_FALSE(in_disc_annulus(r_in - 0.01f, r_in, r_out));
    REQUIRE_FALSE(in_disc_annulus(r_out + 0.01f, r_in, r_out));
    REQUIRE_FALSE(in_disc_annulus(0.0f, r_in, r_out));
}

TEST_CASE("end-to-end: interpolated r_cross lands inside the step's r-span",
          "[disc-intersection]") {
    // Emulate a near-equatorial RK4 pair: ray goes from θ = π/2 − 0.15,
    // r = 12.4 M to θ = π/2 + 0.05, r = 11.9 M. The crossing fraction
    // should pick out r ≈ 12.025 M (1/4 of the way from prev to curr).
    const float theta_prev = kHalfPi - 0.15f;
    const float theta_curr = kHalfPi + 0.05f;
    const float r_prev = 12.4f;
    const float r_curr = 11.9f;

    const DiscCrossing xing = detect_equatorial_crossing(theta_prev, theta_curr);
    REQUIRE(xing.crossed);
    const float r_cross = lerp_scalar(r_prev, r_curr, xing.frac);

    // r_cross must be between the two samples, and no further from the
    // prev sample than the width of the step.
    REQUIRE(r_cross <= std::max(r_prev, r_curr));
    REQUIRE(r_cross >= std::min(r_prev, r_curr));
    // frac = 0.15 / 0.20 = 0.75, r_cross = 12.4 + 0.75 * (11.9 − 12.4) = 12.025.
    REQUIRE(r_cross == Catch::Approx(12.025f).margin(1e-5f));
}
