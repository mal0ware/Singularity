// tests/test_kerr.cpp
//
// Exercises the Kerr analytic scalars in core/include/physics/kerr.hpp. Each
// Test Case cross-checks a closed-form quantity against either (a) a hand-
// computed value at canonical spins, or (b) an internal consistency relation
// that the implementation must satisfy by derivation.
//
// Canonical spin grid (matches PHYSICS.md §11 test matrix):
//     a/M ∈ {0.0, 0.5, 0.94, 0.998}
// Schwarzschild limit at a = 0 must reproduce every scalar from §§3–5, so
// those values anchor the tests.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "physics/kerr.hpp"
#include "physics/schwarzschild.hpp"

using namespace singularity::physics;

namespace {
constexpr float kM = 1.0f;
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

TEST_CASE("Kerr horizons match r± = M ± √(M² − a²)", "[physics][kerr]") {
    // Published values cross-checked against MTW Chapter 33 and Carroll §6.7.
    struct HorizonCase {
        float a_over_M;
        float r_plus_expected;
        float r_minus_expected;
    };
    const HorizonCase cases[] = {
        {0.000f, 2.0f, 0.0f},  // Schwarzschild limit
        {0.500f, 1.866025f, 0.133975f},
        {0.940f, 1.341204f, 0.658796f},
        {0.998f, 1.063246f, 0.936754f},
    };

    for (const auto& c : cases) {
        const float a = c.a_over_M * kM;
        const float r_plus = kerr_outer_horizon(kM, a);
        const float r_minus = kerr_inner_horizon(kM, a);

        // 0.01% tolerance — PHYSICS.md §11 spec. The closed form is exact in
        // exact arithmetic; the budget only absorbs float32 rounding.
        REQUIRE_THAT(r_plus, Catch::Matchers::WithinRel(c.r_plus_expected, 1e-4f));
        REQUIRE_THAT(r_minus, Catch::Matchers::WithinRel(c.r_minus_expected, 1e-4f));

        // Δ(r) = r² − 2Mr + a² vanishes exactly at both horizons.
        REQUIRE(std::fabs(kerr_delta(kM, a, r_plus)) < 1e-5f);
        REQUIRE(std::fabs(kerr_delta(kM, a, r_minus)) < 1e-5f);
    }
}

TEST_CASE("Extremal Kerr collapses both horizons to M", "[physics][kerr]") {
    // Exact-extremal a = M: r_plus = r_minus = M. The implementation clamps
    // the radicand at zero so no NaN escapes even if float noise would make
    // M² − a² slightly negative.
    const float a = kM;
    REQUIRE_THAT(kerr_outer_horizon(kM, a), Catch::Matchers::WithinRel(kM, 1e-6f));
    REQUIRE_THAT(kerr_inner_horizon(kM, a), Catch::Matchers::WithinRel(kM, 1e-6f));
}

TEST_CASE("Kerr ergosphere envelopes the outer horizon", "[physics][kerr]") {
    // At the poles r_ergo = r_plus (ergosphere touches horizon). At the
    // equator r_ergo = 2M for any spin, because the a² cos²θ term is zero.
    const float a_values[] = {0.0f, 0.5f * kM, 0.94f * kM, 0.998f * kM};
    for (float a : a_values) {
        const float r_plus = kerr_outer_horizon(kM, a);
        const float r_ergo_pole = kerr_ergosphere_outer(kM, a, 0.0f);
        const float r_ergo_eq = kerr_ergosphere_outer(kM, a, 0.5f * kPi);

        REQUIRE_THAT(r_ergo_pole, Catch::Matchers::WithinRel(r_plus, 1e-5f));
        REQUIRE_THAT(r_ergo_eq, Catch::Matchers::WithinRel(2.0f * kM, 1e-5f));

        // Off-axis ergosphere always encloses the horizon: r_ergo(θ) ≥ r_plus.
        for (float theta : {0.25f * kPi, 0.5f * kPi, 0.75f * kPi}) {
            REQUIRE(kerr_ergosphere_outer(kM, a, theta) >= r_plus - 1e-6f);
        }
    }
}

TEST_CASE("Kerr photon sphere recovers 3M at a = 0", "[physics][kerr]") {
    // Schwarzschild limit: both branches collapse to the 3M photon sphere of
    // §5.2. This test protects the trigonometric gymnastics in the closed
    // form — a sign flip inside the arccos would miss this check entirely
    // since arccos(0) = π/2 gives the same cos(π/3) = 1/2 either way.
    const float r_ph_pro = kerr_photon_sphere_prograde(kM, 0.0f);
    const float r_ph_ret = kerr_photon_sphere_retrograde(kM, 0.0f);

    REQUIRE_THAT(r_ph_pro, Catch::Matchers::WithinRel(photon_sphere_radius(kM), 1e-5f));
    REQUIRE_THAT(r_ph_ret, Catch::Matchers::WithinRel(photon_sphere_radius(kM), 1e-5f));
}

TEST_CASE("Kerr photon sphere reaches extremal limits", "[physics][kerr]") {
    // a = M: prograde collapses to M (co-rotates with the horizon itself),
    // retrograde stretches to 4M. Canonical result from Carroll §6.7.
    const float a = kM;
    REQUIRE_THAT(kerr_photon_sphere_prograde(kM, a), Catch::Matchers::WithinRel(kM, 1e-5f));
    REQUIRE_THAT(kerr_photon_sphere_retrograde(kM, a),
                 Catch::Matchers::WithinRel(4.0f * kM, 1e-5f));
}

TEST_CASE("Kerr ISCO recovers Schwarzschild 6M at a = 0", "[physics][kerr]") {
    // Zero-spin limit: both branches must collapse to r_ISCO = 6M.
    const float r_isco_pro = kerr_isco_prograde(kM, 0.0f);
    const float r_isco_ret = kerr_isco_retrograde(kM, 0.0f);
    REQUIRE_THAT(r_isco_pro, Catch::Matchers::WithinRel(isco_timelike(kM), 1e-5f));
    REQUIRE_THAT(r_isco_ret, Catch::Matchers::WithinRel(isco_timelike(kM), 1e-5f));
}

TEST_CASE("Kerr ISCO prograde shrinks, retrograde grows with spin", "[physics][kerr]") {
    // Expected values computed at double precision with the PHYSICS.md §7.2
    // Bardeen-Press-Teukolsky closed form, which the float32 implementation
    // reproduces to better than one ULP at every tabulated spin.
    struct IscoCase {
        float a_over_M;
        float pro_expected;
        float ret_expected;
    };
    const IscoCase cases[] = {
        {0.000f, 6.000000f, 6.000000f},
        {0.500f, 4.233003f, 7.554585f},
        {0.940f, 2.023593f, 8.830752f},
        {0.998f, 1.236971f, 8.994374f},
    };

    for (const auto& c : cases) {
        const float a = c.a_over_M * kM;
        const float r_isco_pro = kerr_isco_prograde(kM, a);
        const float r_isco_ret = kerr_isco_retrograde(kM, a);

        // 0.1% tolerance per PHYSICS.md §11 test matrix.
        REQUIRE_THAT(r_isco_pro, Catch::Matchers::WithinRel(c.pro_expected, 1e-3f));
        REQUIRE_THAT(r_isco_ret, Catch::Matchers::WithinRel(c.ret_expected, 1e-3f));

        // Prograde ISCO always lies outside the photon sphere (physical
        // consistency: a timelike stable orbit sits at r > r_photon).
        REQUIRE(r_isco_pro > kerr_photon_sphere_prograde(kM, a));
        REQUIRE(r_isco_ret > kerr_photon_sphere_retrograde(kM, a));
    }
}

TEST_CASE("Kerr ISCO monotonic in spin on each branch", "[physics][kerr]") {
    // Prograde branch strictly decreases with increasing spin; retrograde
    // strictly increases. A sign error in the Z₁/Z₂ formula would typically
    // violate monotonicity before failing the absolute-value checks above.
    float prev_pro = kerr_isco_prograde(kM, 0.0f);
    float prev_ret = kerr_isco_retrograde(kM, 0.0f);
    for (int i = 1; i <= 10; ++i) {
        const float a = (0.1f * i) * kM;
        const float next_pro = kerr_isco_prograde(kM, a);
        const float next_ret = kerr_isco_retrograde(kM, a);
        REQUIRE(next_pro < prev_pro + 1e-5f);
        REQUIRE(next_ret > prev_ret - 1e-5f);
        prev_pro = next_pro;
        prev_ret = next_ret;
    }
}
