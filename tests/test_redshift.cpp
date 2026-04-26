// tests/test_redshift.cpp
//
// Exercises core/include/physics/redshift.hpp. PHYSICS.md §8 + §11 test
// matrix. The individual formulas are short so the test coverage aims to
// hit every limiting case that would catch a typo: the Schwarzschild
// invariants ``g_grav(r, r) = 1`` and ``g_dop(v=0) = 1``, the ISCO disc
// value quoted in §8.1, and the I ∝ g⁴ identity.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "physics/redshift.hpp"

using namespace singularity::physics;

namespace {
constexpr float kM = 1.0f;
constexpr float kRS = 2.0f * kM;
}  // namespace

TEST_CASE("Gravitational redshift collapses to unity when r_emit == r_obs", "[physics][redshift]") {
    // Photon emitted and observed at the same radius — no potential
    // difference to climb. Any drift from 1.0 would signal a mis-squared
    // ratio or swapped arguments.
    for (float r : {3.0f * kRS, 5.0f * kRS, 10.0f * kRS, 100.0f * kRS}) {
        REQUIRE_THAT(schwarzschild_grav_redshift(kRS, r, r),
                     Catch::Matchers::WithinRel(1.0f, 1e-6f));
    }
}

TEST_CASE("Gravitational redshift at ISCO matches the closed form", "[physics][redshift]") {
    // ISCO in the Schwarzschild limit sits at r = 6M, so r_s/r_emit = 1/3
    // and g_grav→∞ = √(1 − 1/3) = √(2/3) ≈ 0.8165. (PHYSICS.md §8.1 quotes
    // 0.913 = √(5/6) instead — that's a typo born of substituting M for
    // r_s under §2's r_s = 2M convention; the correct value is √(2/3).)
    const float r_emit = 6.0f * kM;
    const float g = schwarzschild_grav_redshift_to_infinity(kRS, r_emit);
    REQUIRE_THAT(g, Catch::Matchers::WithinRel(std::sqrt(2.0f / 3.0f), 1e-4f));
}

TEST_CASE("Gravitational redshift factor is < 1 when climbing outward", "[physics][redshift]") {
    // Light emitted deeper in the well (smaller r) is redshifted when
    // observed farther out. A reversed-sign bug would flip this inequality.
    const float g = schwarzschild_grav_redshift(kRS, 3.0f * kRS, 100.0f * kRS);
    REQUIRE(g < 1.0f);
    REQUIRE(g > 0.5f);  // sanity — ISCO-to-infinity is 0.913
}

TEST_CASE("Gravitational redshift blueshifts light falling inward", "[physics][redshift]") {
    // Symmetric case — swap r_emit and r_obs, get 1/g. Photon climbing *in*
    // to a deeper potential is blueshifted.
    const float r_out = 10.0f * kM;
    const float r_in = 3.0f * kM;
    const float g_out = schwarzschild_grav_redshift(kRS, r_in, r_out);
    const float g_in = schwarzschild_grav_redshift(kRS, r_out, r_in);
    REQUIRE_THAT(g_in, Catch::Matchers::WithinRel(1.0f / g_out, 1e-5f));
    REQUIRE(g_in > 1.0f);
}

TEST_CASE("Doppler factor at v = 0 is unity", "[physics][redshift]") {
    // An emitter at rest in the local static frame has no kinematic shift.
    // A mis-encoded Lorentz factor would perturb this from 1.
    const float g = schwarzschild_doppler_factor(kRS, 6.0f * kM, 0.0f, 1.0f);
    REQUIRE_THAT(g, Catch::Matchers::WithinRel(1.0f, 1e-6f));
}

TEST_CASE("Doppler factor beams toward observer, suppresses away from them",
          "[physics][redshift]") {
    // The half of the disc moving toward the observer is blueshifted
    // (g > 1); the half moving away is redshifted (g < 1). This is the
    // asymmetry PHYSICS.md §8.2 highlights — the "strange" visual feature
    // Nolan's team suppressed in Interstellar.
    const float r_emit = 6.0f * kM;
    const float omega = schwarzschild_keplerian_omega(kM, r_emit);
    const float g_toward = schwarzschild_doppler_factor(kRS, r_emit, omega, +1.0f);
    const float g_away = schwarzschild_doppler_factor(kRS, r_emit, omega, -1.0f);
    REQUIRE(g_toward > 1.0f);
    REQUIRE(g_away < 1.0f);
    // Exact identity: g(+cos) · g(−cos) = (1 − v²) / ((1 − v)(1 + v)) = 1.
    // This is a weak constraint (both symbolic reductions give 1 regardless
    // of v) but catches any sign slip that would make the product non-unity.
    REQUIRE_THAT(g_toward * g_away, Catch::Matchers::WithinRel(1.0f, 1e-4f));
    // More informative: at v = 0.5c the blueshifted branch should be √3
    // and the redshifted branch 1/√3 — verify we're in the right ballpark.
    const float v = r_emit * omega / std::sqrt(1.0f - kRS / r_emit);
    const float expected_toward = std::sqrt(1.0f - v * v) / (1.0f - v);
    const float expected_away = std::sqrt(1.0f - v * v) / (1.0f + v);
    REQUIRE_THAT(g_toward, Catch::Matchers::WithinRel(expected_toward, 1e-4f));
    REQUIRE_THAT(g_away, Catch::Matchers::WithinRel(expected_away, 1e-4f));
}

TEST_CASE("Keplerian Ω recovers the Newtonian limit at large r", "[physics][redshift]") {
    // Ω_K = √(M/r³). Kepler's third law survives into GR for Schwarzschild
    // circular orbits. Verifying for large r where GR corrections are
    // negligible gives a closed-form cross-check.
    for (float r : {100.0f * kM, 1000.0f * kM, 10000.0f * kM}) {
        const float omega = schwarzschild_keplerian_omega(kM, r);
        const float expected = std::sqrt(kM / (r * r * r));
        REQUIRE_THAT(omega, Catch::Matchers::WithinRel(expected, 1e-6f));
    }
}

TEST_CASE("Intensity scales as the fourth power of g", "[physics][redshift]") {
    // Rybicki & Lightman §4.9: I_obs = g⁴ I_emit. Trivial check that the
    // helper is actually computing g⁴ and not g² or g^(1/4).
    for (float g : {0.5f, 0.913f, 1.0f, 1.2f, 2.0f}) {
        REQUIRE_THAT(intensity_scaling(g), Catch::Matchers::WithinRel(g * g * g * g, 1e-6f));
    }
}

TEST_CASE("Combined shift beats bare gravitational shift on the approaching side",
          "[physics][redshift]") {
    // For an observer at infinity, the approaching half of a Keplerian disc
    // at r = 6M has g_combined > g_grav (gravitational redshift partially
    // cancelled by Doppler blueshift); the receding half has g_combined <
    // g_grav (both effects redshift). A sign error in the Doppler combiner
    // would flip one of these.
    const float r_emit = 6.0f * kM;
    const float r_obs = 1e6f * kM;  // ≈ infinity
    const float g_grav = schwarzschild_grav_redshift(kRS, r_emit, r_obs);
    const float g_toward = schwarzschild_combined_shift(kRS, r_emit, r_obs, +1.0f);
    const float g_away = schwarzschild_combined_shift(kRS, r_emit, r_obs, -1.0f);
    REQUIRE(g_toward > g_grav);
    REQUIRE(g_away < g_grav);
}
