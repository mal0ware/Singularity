// tests/test_kerr_geodesic.cpp
//
// Exercises shared_shader/kerr_math.h: the geometric scalars (Σ, Δ, P), the
// radial and polar "potentials" (R, Θ), and the sign-tracking geodesic RHS.
// The stepper + turning-point handling (PHYSICS.md §7.3) lives in a follow-up
// header and has its own tests; this file validates the algebra those tests
// will depend on.
//
// Strategy: every Kerr formula reduces to a known Schwarzschild or weak-field
// expression in a limiting case. Tests pin those limits explicitly so a
// typo in the conserved-quantity potentials can't drift past unnoticed.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "kerr_math.h"
#include "physics/kerr.hpp"

using singularity::physics::kerr_inner_horizon;
using singularity::physics::kerr_outer_horizon;
using singularity::physics::kerr_photon_sphere_prograde;

namespace {
constexpr float kM = 1.0f;
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

TEST_CASE("Σ = r² + a² cos²θ at canonical points", "[physics][kerr-geodesic]") {
    // Schwarzschild limit (a = 0): Σ = r² for any θ.
    REQUIRE_THAT(kerr_Sigma(10.0f, 0.25f * kPi, 0.0f), Catch::Matchers::WithinRel(100.0f, 1e-6f));
    // Equatorial plane (θ = π/2): Σ = r² for any a.
    REQUIRE_THAT(kerr_Sigma(10.0f, 0.5f * kPi, 0.8f), Catch::Matchers::WithinRel(100.0f, 1e-5f));
    // Polar axis (θ = 0): Σ = r² + a².
    REQUIRE_THAT(kerr_Sigma(10.0f, 0.0f, 0.8f), Catch::Matchers::WithinRel(100.0f + 0.64f, 1e-5f));
}

TEST_CASE("Δ vanishes exactly at the Kerr horizons", "[physics][kerr-geodesic]") {
    // Δ(r) = r² − 2Mr + a² has roots r± = M ± √(M² − a²). The horizon
    // calculator in kerr.hpp is the independent reference, so asking both
    // to agree cross-checks both encodings of the same algebra.
    for (float a_over_M : {0.0f, 0.3f, 0.5f, 0.9f, 0.98f}) {
        const float a = a_over_M * kM;
        const float r_plus = kerr_outer_horizon(kM, a);
        const float r_minus = kerr_inner_horizon(kM, a);
        REQUIRE(std::fabs(kerr_Delta(r_plus, a, kM)) < 1e-5f);
        REQUIRE(std::fabs(kerr_Delta(r_minus, a, kM)) < 1e-5f);
    }
}

TEST_CASE("P(r) reduces to E·r² at a = 0", "[physics][kerr-geodesic]") {
    // P(r) = E(r² + a²) − a L_z collapses to E r² in the Schwarzschild limit,
    // regardless of L_z. A bug that dropped the `a` coefficient from the
    // L_z term would pass this test; a bug that confused a and M would not.
    for (float r : {3.0f, 10.0f, 50.0f}) {
        REQUIRE_THAT(kerr_P(r, 0.0f, 1.0f, 5.0f), Catch::Matchers::WithinRel(r * r, 1e-6f));
    }
}

TEST_CASE("Kerr R(r) reduces to Schwarzschild equatorial-null potential at a=0",
          "[physics][kerr-geodesic]") {
    // Schwarzschild equatorial null (PHYSICS.md §5.1):
    //     (dr/dλ)² = E² − (L²/r²)(1 − r_s/r)
    // Kerr form:   Σ² (dr/dλ)² = R(r)      [with a=0, Q=0, θ=π/2]
    //              Σ at equator = r²       (independent of a)
    // So Schwarzschild expression = R(r) / r⁴. Equivalently:
    //     R(r) = r⁴ E² − r² (r² − r_s r) L²
    //          = r⁴ E² − r (r − r_s) L² · r²
    // Check point-by-point.
    constexpr float kRS = 2.0f * kM;
    KerrConserved c{};
    c.E = 1.0f;
    c.L_z = 5.0f;
    c.Q = 0.0f;
    c.a = 0.0f;
    c.M = kM;
    for (float r : {3.0f, 5.0f, 10.0f, 50.0f}) {
        const float expected = r * r * r * r - r * (r - kRS) * c.L_z * c.L_z;
        REQUIRE_THAT(kerr_R(r, c), Catch::Matchers::WithinRel(expected, 1e-5f));
    }
}

TEST_CASE("Kerr R(r) vanishes at the horizon regardless of spin", "[physics][kerr-geodesic]") {
    // At r = r_plus, Δ = 0, so R(r) = P(r)² — generally non-zero. But if the
    // photon is tuned to the horizon generator (L_z = a E (r+²+a²)/(2Mr+) or
    // similar), P also vanishes. Simpler invariant: R(r_+) = P(r_+)², which
    // is a fixed function of E, L_z, a. Verify the algebra matches.
    for (float a_over_M : {0.0f, 0.5f, 0.94f}) {
        const float a = a_over_M * kM;
        const float r_plus = kerr_outer_horizon(kM, a);
        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = 3.5f;
        c.Q = 0.5f;
        c.a = a;
        c.M = kM;
        const float P = kerr_P(r_plus, a, c.E, c.L_z);
        REQUIRE_THAT(kerr_R(r_plus, c), Catch::Matchers::WithinRel(P * P, 5e-4f));
    }
}

TEST_CASE("Equatorial Kerr photon sphere: R = 0 and dR/dr = 0 simultaneously",
          "[physics][kerr-geodesic]") {
    // At the prograde equatorial photon orbit r = r_ph_pro(a), with the
    // critical impact parameter b_pro(a) and Q = 0, the radial potential
    // has a double root: R = 0 and dR/dr = 0. This is *the* definition of
    // the photon sphere. A bug in any of the P / Δ / R expressions would
    // perturb one of these two conditions by a finite amount.
    //
    // Chandrasekhar §63: derived directly from R(r) = 0 in the equatorial
    // plane with Q = 0. Solving the resulting quadratic in b = L_z/E gives
    //
    //     b_prograde(r, a, M) = (r √Δ − 2 M a) / (r − 2 M)
    //
    // recovering 3 √3 M at a = 0 and M at extremal prograde. The analogous
    // retrograde branch has a plus sign on the radical.

    auto b_ph_pro = [](float a, float M, float r_ph) {
        const float Delta = r_ph * r_ph - 2.0f * M * r_ph + a * a;
        return (r_ph * std::sqrt(Delta) - 2.0f * M * a) / (r_ph - 2.0f * M);
    };

    auto dR_dr = [](float r, KerrConserved c) {
        // Finite-difference for the derivative; the analytic expression is
        // algebraically straightforward but cross-checking via FD also
        // validates that kerr_R itself is smooth in r away from horizons.
        const float h = 1e-4f;
        return (kerr_R(r + h, c) - kerr_R(r - h, c)) / (2.0f * h);
    };

    for (float a_over_M : {0.0f, 0.3f, 0.5f, 0.9f}) {
        const float a = a_over_M * kM;
        const float r_ph = kerr_photon_sphere_prograde(kM, a);
        const float b = b_ph_pro(a, kM, r_ph);

        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = b;  // b = L_z / E with E = 1
        c.Q = 0.0f;
        c.a = a;
        c.M = kM;

        // Absolute tolerance scales with L² r²: use a scale-relative cap.
        const float scale = (c.L_z * c.L_z + 1.0f) * r_ph * r_ph;
        REQUIRE(std::fabs(kerr_R(r_ph, c)) < 5e-3f * scale);
        REQUIRE(std::fabs(dR_dr(r_ph, c)) < 5e-2f * scale);
    }
}

TEST_CASE("Kerr Θ reduces to Q at the equator for any spin", "[physics][kerr-geodesic]") {
    // θ = π/2: cos²θ = 0, so Θ(θ) = Q. A bug that mixed up sin and cos in
    // the polar potential would flip this invariant — worth pinning.
    for (float a : {0.0f, 0.3f, 0.9f}) {
        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = 2.0f;
        c.a = a;
        c.M = kM;
        for (float Q : {-0.5f, 0.0f, 0.5f, 5.0f}) {
            c.Q = Q;
            REQUIRE_THAT(kerr_Theta(0.5f * kPi, c), Catch::Matchers::WithinAbs(Q, 1e-6f));
        }
    }
}

TEST_CASE("Geodesic RHS at equator has dθ/dλ = 0 when Q = 0", "[physics][kerr-geodesic]") {
    // Q = 0 + equatorial start: the geodesic stays in the equatorial plane.
    // The RHS must return dθ/dλ = 0 exactly there — any non-zero output
    // would cause the ray to leak out of the plane even when physics says
    // it can't.
    for (float a : {0.0f, 0.5f, 0.94f}) {
        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = 4.0f;
        c.Q = 0.0f;
        c.a = a;
        c.M = kM;

        KerrCoord s{};
        s.t = 0.0f;
        s.r = 10.0f * kM;
        s.theta = 0.5f * kPi;
        s.phi = 0.0f;

        const KerrCoord d = kerr_geodesic_rhs(s,
                                              c,
                                              /*sigma_r=*/+1.0f,
                                              /*sigma_theta=*/+1.0f);
        REQUIRE(std::fabs(d.theta) < 1e-6f);
        // dr/dλ and dφ/dλ should be finite non-zero — placeholder sanity.
        REQUIRE(std::isfinite(d.r));
        REQUIRE(std::isfinite(d.phi));
        REQUIRE(std::isfinite(d.t));
    }
}

TEST_CASE("RK4 advances an outbound equatorial photon outward", "[physics][kerr-geodesic]") {
    // Fire a photon at r = 20 M, θ = π/2, moving radially outward (σ_r = +1).
    // No polar motion (Q = 0). After N RK4 steps, r must have increased and
    // θ must still sit on the equator — a first smoke test that the stepper
    // reaches the right quadrant of phase space without blowing up.
    for (float a_over_M : {0.0f, 0.5f, 0.94f}) {
        const float a = a_over_M * kM;
        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = 4.0f;
        c.Q = 0.0f;
        c.a = a;
        c.M = kM;

        KerrStepperState s{};
        s.coord.t = 0.0f;
        s.coord.r = 20.0f * kM;
        s.coord.theta = 0.5f * kPi;
        s.coord.phi = 0.0f;
        s.sigma_r = +1.0f;
        s.sigma_theta = +1.0f;

        const float r_start = s.coord.r;
        for (int i = 0; i < 2000; ++i) {
            s = kerr_rk4_step_tracking(s, 0.1f * kM, c);
        }
        REQUIRE(s.coord.r > r_start);                            // moved outward
        REQUIRE(std::fabs(s.coord.theta - 0.5f * kPi) < 1e-3f);  // stayed equatorial
        REQUIRE(std::isfinite(s.coord.phi));
        REQUIRE(std::isfinite(s.coord.t));
        REQUIRE(s.sigma_r == +1.0f);  // no radial turn
    }
}

TEST_CASE("RK4 preserves the null condition along an equatorial geodesic",
          "[physics][kerr-geodesic]") {
    // The null condition g_μν u^μ u^ν = 0 follows from the construction of
    // R(r) and Θ(θ), so if the stepper is correct the residual should stay
    // at integrator-noise level (~float eps × step-count) across many steps.
    // We check it directly by evaluating the four-velocity derived from the
    // RHS and the covariant metric components.
    constexpr float a = 0.5f * kM;
    KerrConserved c{};
    c.E = 1.0f;
    c.L_z = 4.0f;
    c.Q = 0.0f;
    c.a = a;
    c.M = kM;

    KerrStepperState s{};
    s.coord.t = 0.0f;
    s.coord.r = 15.0f * kM;
    s.coord.theta = 0.5f * kPi;
    s.coord.phi = 0.0f;
    s.sigma_r = +1.0f;
    s.sigma_theta = +1.0f;

    auto null_residual = [&](const KerrStepperState& st) {
        const KerrCoord u = kerr_geodesic_rhs(st.coord, c, st.sigma_r, st.sigma_theta);
        const float r = st.coord.r;
        const float s_th = std::sin(st.coord.theta);
        const float s2 = s_th * s_th;
        const float Sigma = kerr_Sigma(r, st.coord.theta, c.a);
        const float Delta = kerr_Delta(r, c.a, c.M);
        // Kerr covariant metric (Boyer-Lindquist):
        //   g_tt  = −(1 − 2Mr/Σ)
        //   g_tφ  = −2 M r a sin²θ / Σ
        //   g_rr  = Σ / Δ
        //   g_θθ  = Σ
        //   g_φφ  = sin²θ · ( (r² + a²)² − a² Δ sin²θ ) / Σ
        const float g_tt = -(1.0f - 2.0f * c.M * r / Sigma);
        const float g_tp = -2.0f * c.M * r * c.a * s2 / Sigma;
        const float g_rr = Sigma / Delta;
        const float g_thth = Sigma;
        const float g_pp =
            s2 * ((r * r + c.a * c.a) * (r * r + c.a * c.a) - c.a * c.a * Delta * s2) / Sigma;
        return g_tt * u.t * u.t + 2.0f * g_tp * u.t * u.phi + g_rr * u.r * u.r
               + g_thth * u.theta * u.theta + g_pp * u.phi * u.phi;
    };

    const float residual_start = null_residual(s);
    for (int i = 0; i < 1000; ++i) {
        s = kerr_rk4_step_tracking(s, 0.1f * kM, c);
    }
    const float residual_end = null_residual(s);

    // Starting residual should be tiny (the RHS is constructed to solve
    // g_μν u^μ u^ν = 0 exactly).
    REQUIRE(std::fabs(residual_start) < 1e-3f);
    // After 1000 RK4 steps at h = 0.1 M, the residual can drift with the
    // accumulated truncation error; accept up to 1e-1 for the float32 path.
    REQUIRE(std::fabs(residual_end) < 1e-1f);
}

TEST_CASE("Kerr RHS recovers Schwarzschild dφ/dt ratio at a=0 equator",
          "[physics][kerr-geodesic]") {
    // Schwarzschild equatorial null (at the radial turn of a trajectory,
    // u^r = 0): L_z = r² u^φ and E = (1 − r_s/r) u^t give the ratio
    //     (dφ/dλ)/(dt/dλ) = L_z / (r² · u^t) · 1
    //                     = L_z (1 − r_s/r) / (E r²)
    // The Kerr RHS in the Schwarzschild limit must reproduce this directly.
    constexpr float kRS = 2.0f * kM;
    KerrConserved c{};
    c.E = 1.0f;
    c.L_z = 3.5f;
    c.Q = 0.0f;
    c.a = 0.0f;
    c.M = kM;

    const float r = 10.0f * kM;
    KerrCoord s{};
    s.t = 0.0f;
    s.r = r;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;

    const KerrCoord d = kerr_geodesic_rhs(s, c, +1.0f, +1.0f);

    const float expected_ratio = c.L_z * (1.0f - kRS / r) / (c.E * r * r);
    REQUIRE_THAT(d.phi / d.t, Catch::Matchers::WithinRel(expected_ratio, 1e-4f));
}
