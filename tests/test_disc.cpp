// tests/test_disc.cpp
//
// Exercises the accretion-disc physics in core/include/physics/disc.hpp.
// PHYSICS.md §8.3 specifies the Novikov-Thorne simplified temperature
// profile and the inner-edge suppression; these tests pin the formulas at
// their closed-form limits and confirm monotonicity of the radial profile.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "physics/disc.hpp"

using namespace singularity::physics;

namespace {
constexpr float kM = 1.0f;
constexpr float kISCO = 6.0f * kM;
constexpr float kT_REF = 1.0e7f;  // arbitrary reference K (not verified in tests)
}  // namespace

TEST_CASE("Disc temperature hits T_ref at r = r_ref", "[physics][disc]") {
    REQUIRE_THAT(disc_temperature(kISCO, kISCO, kT_REF), Catch::Matchers::WithinRel(kT_REF, 1e-6f));
    REQUIRE_THAT(disc_temperature(kISCO * 2.0f, kISCO * 2.0f, kT_REF),
                 Catch::Matchers::WithinRel(kT_REF, 1e-6f));
}

TEST_CASE("Disc temperature falls off as r^(-3/4)", "[physics][disc]") {
    // Doubling the radius must divide the temperature by 2^(3/4) = 1.6818.
    const float t1 = disc_temperature(kISCO, kISCO, kT_REF);
    const float t2 = disc_temperature(kISCO * 2.0f, kISCO, kT_REF);
    const float t4 = disc_temperature(kISCO * 4.0f, kISCO, kT_REF);
    REQUIRE_THAT(t1 / t2, Catch::Matchers::WithinRel(std::pow(2.0f, 0.75f), 1e-5f));
    REQUIRE_THAT(t2 / t4, Catch::Matchers::WithinRel(std::pow(2.0f, 0.75f), 1e-5f));
}

TEST_CASE("Disc temperature is strictly monotonic in r", "[physics][disc]") {
    // Scan outward from ISCO — the inner edge of the physical disc. The bare
    // power law has no inner cutoff so we stay outside r_ISCO to avoid the
    // unphysical inward regime.
    float prev = disc_temperature(kISCO, kISCO, kT_REF);
    for (int i = 1; i <= 20; ++i) {
        const float r = kISCO + float(i) * kM;
        const float t = disc_temperature(r, kISCO, kT_REF);
        REQUIRE(t < prev);
        prev = t;
    }
}

TEST_CASE("Inner-edge factor vanishes at ISCO and asymptotes to 1", "[physics][disc]") {
    REQUIRE(disc_inner_edge_factor(kISCO, kISCO) == 0.0f);
    // 1 − √(r_ISCO/r) with r = r_ISCO: zero, as written.
    // At r = 100 r_ISCO: 1 − √(1/100) = 0.9, close to unity.
    REQUIRE_THAT(disc_inner_edge_factor(100.0f * kISCO, kISCO),
                 Catch::Matchers::WithinRel(0.9f, 1e-4f));
    // Monotonic: the factor always increases with r past r_ISCO.
    float prev = 0.0f;
    for (int i = 1; i <= 50; ++i) {
        const float r = kISCO * (1.0f + 0.1f * float(i));
        const float f = disc_inner_edge_factor(r, kISCO);
        REQUIRE(f > prev);
        REQUIRE(f < 1.0f);
        prev = f;
    }
}

TEST_CASE("Inner-edge factor clamps to zero inside r_ISCO", "[physics][disc]") {
    // Defensive: a physical disc cannot extend inward of its ISCO, but the
    // formula's output there would be complex (sqrt of a value > 1 goes
    // negative under 1 − √). We clamp at zero so callers don't receive
    // either NaN or a nonsensical negative emissivity.
    REQUIRE(disc_inner_edge_factor(0.5f * kISCO, kISCO) == 0.0f);
    REQUIRE(disc_inner_edge_factor(0.99f * kISCO, kISCO) == 0.0f);
}

TEST_CASE("Novikov-Thorne temperature peaks off the inner edge", "[physics][disc]") {
    // The combined T(r) = r^(−3/4) · f(r)^(1/4) vanishes at the ISCO (where
    // f = 0) and decays to zero at infinity — so it has a peak in between.
    // Computing dT/dr = 0 for the simplified profile gives r_peak =
    // (49/36) r_ISCO ≈ 1.361 r_ISCO (Page & Thorne 1974). Verify by scan.
    float t_peak = 0.0f;
    float r_peak = 0.0f;
    for (int i = 100; i <= 300; ++i) {  // r/r_ISCO in [1.00, 3.00]
        const float r = kISCO * (float(i) * 0.01f);
        const float t = disc_temperature_nt(r, kISCO, kT_REF, kISCO);
        if (t > t_peak) {
            t_peak = t;
            r_peak = r;
        }
    }
    REQUIRE(r_peak > 1.3f * kISCO);
    REQUIRE(r_peak < 1.4f * kISCO);
}

TEST_CASE("Disc flux scales as T^4 (Stefan-Boltzmann)", "[physics][disc]") {
    // disc_flux_ratio is a thin wrapper on (T/T_ref)^4 — a direct check
    // that the fourth power is computed and not squared/square-rooted.
    for (float ratio : {0.5f, 0.913f, 1.0f, 1.5f, 2.0f}) {
        const float t = ratio * kT_REF;
        REQUIRE_THAT(disc_flux_ratio(t, kT_REF),
                     Catch::Matchers::WithinRel(ratio * ratio * ratio * ratio, 1e-5f));
    }
}

TEST_CASE("Wien's law round-trips", "[physics][disc]") {
    // λ_peak ↔ T is a simple reciprocal; round-tripping should return the
    // same value up to float precision.
    for (float T : {1e3f, 1e4f, 1e5f, 5778.0f /* Sun */, 1e7f}) {
        const float lambda = wien_peak_wavelength_m(T);
        const float T_back = wien_temperature_K(lambda);
        REQUIRE_THAT(T_back, Catch::Matchers::WithinRel(T, 1e-5f));
    }
}

TEST_CASE("Wien's law matches the sun's canonical peak wavelength", "[physics][disc]") {
    // Well-known: the Sun's surface is at ~5778 K, its Planck spectrum peaks
    // at ~502 nm (green-yellow). A drift here would flag either a wrong
    // constant or unit confusion (b is given in m·K, not nm·K).
    const float lambda_sun = wien_peak_wavelength_m(5778.0f);
    REQUIRE_THAT(lambda_sun, Catch::Matchers::WithinRel(5.015e-7f, 1e-3f));
}

TEST_CASE("Blackbody sRGB is red-saturated at cool temperatures", "[physics][disc]") {
    // Cool M-dwarf at 3000 K reads orange-red: R clipped at 1, G well below
    // R, B small. Any channel swap would fail this.
    const sRGBColor c = blackbody_srgb_tanner_helland(3000.0f);
    REQUIRE(c.r > 0.95f);
    REQUIRE(c.g < c.r);
    REQUIRE(c.b < c.g);
    REQUIRE(c.b < 0.5f);
}

TEST_CASE("Blackbody sRGB is approximately white around 6500 K", "[physics][disc]") {
    // D65 daylight whitepoint sits near 6500 K. R, G, B should all be within
    // a small delta of each other.
    const sRGBColor c = blackbody_srgb_tanner_helland(6500.0f);
    const float mean = (c.r + c.g + c.b) / 3.0f;
    REQUIRE(std::fabs(c.r - mean) < 0.15f);
    REQUIRE(std::fabs(c.g - mean) < 0.15f);
    REQUIRE(std::fabs(c.b - mean) < 0.15f);
    // Not grey (sanity — a bug that zeroed all channels would also pass the
    // equal-channel test otherwise).
    REQUIRE(mean > 0.8f);
}

TEST_CASE("Blackbody sRGB is blue-saturated at hot temperatures", "[physics][disc]") {
    // Hot O-star at 25000 K: B clipped at 1, R depleted, G intermediate.
    const sRGBColor c = blackbody_srgb_tanner_helland(25000.0f);
    REQUIRE(c.b > 0.95f);
    REQUIRE(c.r < c.b);
    REQUIRE(c.g < c.b);
    REQUIRE(c.r < c.g);  // blue-white, not blue-green
}

TEST_CASE("Blackbody sRGB clamps to [0, 1] at extreme temperatures", "[physics][disc]") {
    // Fit extrapolation beyond its nominal range must stay in gamut — the
    // disc inner edge can reach 1e5 K for stellar-mass BH scenarios.
    for (float T : {500.0f, 1000.0f, 2000.0f, 6500.0f, 10000.0f, 40000.0f, 100000.0f}) {
        const sRGBColor c = blackbody_srgb_tanner_helland(T);
        REQUIRE(c.r >= 0.0f);
        REQUIRE(c.r <= 1.0f);
        REQUIRE(c.g >= 0.0f);
        REQUIRE(c.g <= 1.0f);
        REQUIRE(c.b >= 0.0f);
        REQUIRE(c.b <= 1.0f);
    }
}

TEST_CASE("Blackbody sRGB is continuous across the t=66 branch", "[physics][disc]") {
    // The piecewise formula switches branches at T/100 = 66. At the seam
    // the two sides must agree — any jump would produce visible banding
    // in the disc LUT.
    const sRGBColor lo = blackbody_srgb_tanner_helland(6599.0f);
    const sRGBColor hi = blackbody_srgb_tanner_helland(6601.0f);
    REQUIRE(std::fabs(lo.r - hi.r) < 0.05f);
    REQUIRE(std::fabs(lo.g - hi.g) < 0.05f);
    REQUIRE(std::fabs(lo.b - hi.b) < 0.05f);
}
