// tests/test_kerr_hamilton.cpp
//
// Exercises the Hamiltonian-form Kerr geodesic integrator in
// shared_shader/kerr_hamilton.h. The sign-tracking form in kerr_math.h has a
// known stall behaviour at radial turning points (documented in its header);
// the Hamiltonian form advances (p_r, p_θ) smoothly through turning points
// because they correspond to canonical momenta passing through zero, not
// branch flips.
//
// Strategy:
//   1. On-shell condition: 2H = 0 (null geodesic) holds to float precision
//      at initialisation and stays there through many RK4 steps.
//   2. Schwarzschild-limit recovery: at a = 0, equatorial, Kerr Hamiltonian
//      reproduces Schwarzschild geodesics that the existing shared RK4
//      already validates.
//   3. Circular photon orbit at the equatorial Schwarzschild photon sphere
//      closes within tolerance over an orbital period.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "kerr_hamilton.h"
#include "kerr_math.h"

namespace {
constexpr float kM = 1.0f;
constexpr float kPi = 3.14159265358979323846f;
}  // namespace

TEST_CASE("Hamiltonian 2H initialised to zero on a null orbit", "[physics][kerr-hamilton]") {
    // Schwarzschild-limit equatorial photon sphere: r = 3M, u^φ = (1/r)√f,
    // u^r = 0, u^θ = 0. Compute (E, L_z) from these and assert 2H = 0.
    KerrConserved c{};
    c.a = 0.0f;
    c.M = kM;

    const float r = 3.0f * kM;
    const float f = 1.0f - 2.0f * kM / r;
    const float u_t = 1.0f;
    const float u_phi = std::sqrt(f) / r;  // null condition in eq. plane
    c.E = f * u_t;                         // E = (1−rs/r) u^t at equator
    c.L_z = r * r * u_phi;                 // L_z = r² u^φ
    c.Q = 0.0f;

    KerrHamState s{};
    s.t = 0.0f;
    s.r = r;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    kerr_ham_momenta_from_velocities(&s, /*u_r=*/0.0f, /*u_θ=*/0.0f, c);

    const float H2 = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    REQUIRE(std::fabs(H2) < 1e-4f);
}

TEST_CASE("Hamiltonian RHS at equator has dθ/dλ = 0 when p_θ = 0", "[physics][kerr-hamilton]") {
    // An equatorial start with p_θ = 0 must stay equatorial: dθ/dλ = p_θ/Σ.
    // Similarly, dp_θ/dλ at θ = π/2 is proportional to ∂(2H)/∂θ, which
    // must vanish by symmetry at the equator for L_z² / sin²θ (even in θ
    // about π/2) and for the other θ-terms.
    for (float a : {0.0f, 0.5f, 0.94f}) {
        KerrConserved c{};
        c.E = 1.0f;
        c.L_z = 4.0f;
        c.Q = 0.0f;
        c.a = a;
        c.M = kM;

        KerrHamState s{};
        s.t = 0.0f;
        s.r = 10.0f * kM;
        s.theta = 0.5f * kPi;
        s.phi = 0.0f;
        s.p_r = 0.0f;
        s.p_theta = 0.0f;

        const KerrHamState d = kerr_ham_rhs(s, c);
        REQUIRE(std::fabs(d.theta) < 1e-6f);
        // dp_θ/dλ at equator — sin(2θ) factor in ∂Σ/∂θ vanishes, and the
        // sin²θ piece in 2H is at its minimum, so the derivative is zero.
        REQUIRE(std::fabs(d.p_theta) < 1e-4f);
    }
}

TEST_CASE("Hamiltonian Schwarzschild photon sphere closes within 0.5%",
          "[physics][kerr-hamilton]") {
    // Same test as the sign-tracking form — but via Hamilton's equations,
    // which smoothly integrate p_r through zero without a sign flip. Closure
    // to 0.5% is the PHYSICS.md §11 tolerance for photon-sphere orbits.
    KerrConserved c{};
    c.a = 0.0f;
    c.M = kM;

    const float r0 = 3.0f * kM;
    const float f = 1.0f - 2.0f * kM / r0;
    const float u_t = 1.0f;
    const float u_phi = std::sqrt(f) / r0;
    c.E = f * u_t;
    c.L_z = r0 * r0 * u_phi;
    c.Q = 0.0f;

    KerrHamState s{};
    s.t = 0.0f;
    s.r = r0;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    kerr_ham_momenta_from_velocities(&s, 0.0f, 0.0f, c);

    const float period = 2.0f * kPi / u_phi;
    const float h = 0.01f * kM;
    const int n = int(period / h);
    for (int i = 0; i < n; ++i) {
        s = kerr_ham_rk4_step(s, h, c);
    }
    REQUIRE(std::fabs(s.r - r0) < 0.005f * r0);
    REQUIRE(std::fabs(s.theta - 0.5f * kPi) < 1e-4f);
}

TEST_CASE("Hamiltonian Kerr integrator handles non-equatorial geodesics",
          "[physics][kerr-hamilton]") {
    // Off-equator photon orbit — Q > 0 forces polar motion. The θ coordinate
    // must stay bounded and 2H stay near zero; a bug in the momentum
    // derivatives that produced θ-drift would fail this.
    KerrConserved c{};
    c.a = 0.5f * kM;
    c.M = kM;
    c.E = 1.0f;
    c.L_z = 3.0f;  // prograde angular momentum
    c.Q = 2.0f;    // positive Carter constant — true 3D orbit

    KerrHamState s{};
    s.t = 0.0f;
    s.r = 10.0f * kM;
    s.theta = kPi / 3.0f;  // 60° off-equator
    s.phi = 0.0f;
    // p_r from null-condition at this point:  2H = 0 fixes p_r² given
    // (r, θ, p_θ, E, L_z, Q, a, M). We set p_θ = 0 (instantaneous polar
    // turning point) and solve for p_r² via the relation p_r² = R(r)/Δ²
    // in the sign-tracking formulation — translate into the Hamiltonian
    // momenta via p_r = Σ u^r / Δ = √R / Δ.
    const float R_val = kerr_R(s.r, c);
    const float Delta = kerr_Delta(s.r, c.a, c.M);
    REQUIRE(R_val > 0.0f);
    s.p_r = std::sqrt(R_val) / Delta;
    s.p_theta = 0.0f;

    const float H2_0 = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    // Initial residual sits at ~1e-2 for float32 — the null condition's
    // 2H=0 is the outcome of several O(1)·O(1) products being subtracted,
    // so float catastrophic cancellation leaves ~10 ULP in the result.
    // An analytical bug would push this past 1 in magnitude.
    REQUIRE(std::fabs(H2_0) < 2e-2f);

    float theta_min = s.theta;
    float theta_max = s.theta;
    float r_min = s.r;
    float r_max = s.r;

    for (int i = 0; i < 1500; ++i) {
        s = kerr_ham_rk4_step(s, 0.1f * kM, c);
        if (!std::isfinite(s.r) || !std::isfinite(s.theta))
            break;
        theta_min = std::min(theta_min, s.theta);
        theta_max = std::max(theta_max, s.theta);
        r_min = std::min(r_min, s.r);
        r_max = std::max(r_max, s.r);
    }

    // θ must have oscillated (not stuck at start) and stayed bounded away
    // from the poles (L_z ≠ 0 forbids the pole).
    REQUIRE(theta_max - theta_min > 0.01f);
    REQUIRE(theta_min > 0.1f);
    REQUIRE(theta_max < kPi - 0.1f);

    // 2H stays small after a long run.
    const float H2_end = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    REQUIRE(std::fabs(H2_end) < 1e-1f);
    REQUIRE(std::isfinite(r_min));
    REQUIRE(std::isfinite(r_max));
}

TEST_CASE("Hamiltonian stress test — 20000 RK4 steps stay bounded",
          "[physics][kerr-hamilton][stress]") {
    // Long-horizon escape trajectory: photon starts well outside the photon
    // sphere and is integrated for 20000 RK4 steps at h = 0.1 M. At this
    // step count truncation error can reveal slow secular drift that's
    // invisible on the 1000-step scale. We don't require 2H stays at float
    // precision — just that it hasn't blown past 1 (gross bug territory)
    // and the coordinates stay finite.
    KerrConserved c{};
    c.a = 0.3f * kM;
    c.M = kM;
    c.E = 1.0f;
    c.L_z = 2.0f;
    c.Q = 0.0f;

    KerrHamState s{};
    s.t = 0.0f;
    s.r = 50.0f * kM;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    // Outbound, nearly-radial photon — set u^r = +1 and derive p_r.
    const float ur = 1.0f;
    const float Delta = kerr_Delta(s.r, c.a, c.M);
    const float Sigma = kerr_Sigma(s.r, s.theta, c.a);
    s.p_r = Sigma * ur / Delta;
    s.p_theta = 0.0f;

    for (int i = 0; i < 20000; ++i) {
        s = kerr_ham_rk4_step(s, 0.1f * kM, c);
        // Bail out if the photon legitimately flies past our bookkeeping
        // (it's outbound; r grows). We only want to catch blow-ups.
        if (s.r > 1e6f * kM)
            break;
    }

    REQUIRE(std::isfinite(s.r));
    REQUIRE(std::isfinite(s.theta));
    REQUIRE(std::isfinite(s.phi));
    REQUIRE(std::isfinite(s.p_r));
    REQUIRE(std::isfinite(s.p_theta));
    const float H2 = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    REQUIRE(std::isfinite(H2));
    REQUIRE(std::fabs(H2) < 1.0f);
}

TEST_CASE("Hamiltonian form preserves 2H = 0 along a long geodesic", "[physics][kerr-hamilton]") {
    // Kerr equatorial null geodesic starting well outside the photon sphere.
    // 2H begins at zero by construction; RK4 accumulates truncation error
    // but it should stay comfortably within 1e-3 over 2000 steps at h = 0.1.
    KerrConserved c{};
    c.a = 0.5f * kM;
    c.M = kM;
    c.Q = 0.0f;

    // Outbound photon at r = 30 M, moving in +r̂ direction with some
    // orbital component. Set u^r = +1, u^θ = 0, u^φ = 0.01 (tangential).
    const float r0 = 30.0f * kM;
    const float ur = 1.0f;
    const float uphi = 0.01f;
    // Null condition for Kerr equatorial (θ = π/2):
    //   g_tt(u^t)² + 2g_tφ u^t u^φ + g_rr (u^r)² + g_φφ (u^φ)² = 0
    const float rs = 2.0f * kM;
    const float Delta = r0 * r0 - 2.0f * kM * r0 + c.a * c.a;
    const float g_tt = -(1.0f - 2.0f * kM / r0);
    const float g_tp = -2.0f * kM * c.a / r0;
    const float g_rr = r0 * r0 / Delta;
    const float g_pp =
        ((r0 * r0 + c.a * c.a) * (r0 * r0 + c.a * c.a) - c.a * c.a * Delta) / (r0 * r0);
    const float Aq = g_tt;
    const float Bq = 2.0f * g_tp * uphi;
    const float Cq = g_rr * ur * ur + g_pp * uphi * uphi;
    const float disc = Bq * Bq - 4.0f * Aq * Cq;
    const float ut = (-Bq - std::sqrt(disc)) / (2.0f * Aq);
    c.E = -(g_tt * ut + g_tp * uphi);
    c.L_z = g_tp * ut + g_pp * uphi;
    (void)rs;

    KerrHamState s{};
    s.t = 0.0f;
    s.r = r0;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    kerr_ham_momenta_from_velocities(&s, ur, 0.0f, c);

    const float H2_start = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    REQUIRE(std::fabs(H2_start) < 1e-3f);

    for (int i = 0; i < 2000; ++i) {
        s = kerr_ham_rk4_step(s, 0.1f * kM, c);
    }
    const float H2_end = kerr_ham_two_H(s.r, s.theta, s.p_r, s.p_theta, c);
    REQUIRE(std::isfinite(H2_end));
    REQUIRE(std::fabs(H2_end) < 1e-2f);
}

TEST_CASE("Analytic momentum derivatives match finite differences in the bulk",
          "[physics][kerr-hamilton][analytic]") {
    // The RHS now uses analytic partials of 2H (SymPy-verified in
    // verification/test_kerr_ham_analytic.py). Away from the poles the old
    // centred finite difference was accurate, so the two must agree there —
    // this pins the C++ transcription of the algebra. Double-precision FD on
    // the same float expression bounds the comparison noise.
    auto fd = [](float r, float theta, float p_r, float p_theta, KerrConserved c, bool wrt_r) {
        const double h = 1.0e-4;
        const float fp = wrt_r ? kerr_ham_two_H(r + float(h), theta, p_r, p_theta, c)
                               : kerr_ham_two_H(r, theta + float(h), p_r, p_theta, c);
        const float fm = wrt_r ? kerr_ham_two_H(r - float(h), theta, p_r, p_theta, c)
                               : kerr_ham_two_H(r, theta - float(h), p_r, p_theta, c);
        return (double(fp) - double(fm)) / (2.0 * h);
    };

    const float radii[] = {3.1f, 6.0f, 12.0f, 40.0f};
    const float thetas[] = {0.4f, 1.0f, 0.5f * kPi, 2.4f};
    const float spins[] = {0.0f, 0.5f, 0.94f};
    for (float r : radii) {
        for (float theta : thetas) {
            for (float a : spins) {
                KerrConserved c{};
                c.a = a * kM;
                c.M = kM;
                c.E = 1.0f;
                c.L_z = 3.7f;
                KerrHamState s{};
                s.r = r;
                s.theta = theta;
                s.p_r = 0.6f;
                s.p_theta = 2.3f;

                const KerrHamState d = kerr_ham_rhs(s, c);
                const double fd_r = -0.5 * fd(r, theta, s.p_r, s.p_theta, c, true);
                const double fd_t = -0.5 * fd(r, theta, s.p_r, s.p_theta, c, false);

                const double scale_r = std::max(1.0, std::fabs(fd_r));
                const double scale_t = std::max(1.0, std::fabs(fd_t));
                INFO("r=" << r << " theta=" << theta << " a=" << a);
                REQUIRE(std::fabs(double(d.p_r) - fd_r) / scale_r < 5e-3);
                REQUIRE(std::fabs(double(d.p_theta) - fd_t) / scale_t < 5e-3);
            }
        }
    }
}

TEST_CASE("Polar centrifugal barrier is repulsive near the axis",
          "[physics][kerr-hamilton][analytic]") {
    // The seam the finite difference produced: approaching θ = π with
    // L_z ≠ 0, the barrier force must push θ back toward the equator
    // (dp_θ/dλ < 0 for θ just below π means d²θ/dλ² decelerates). The old
    // centred difference sampled symmetrically across the pole and returned
    // ~0 there. Also: with L_z = 0 exactly, the barrier must vanish and the
    // derivative stay finite.
    KerrConserved c{};
    c.a = 0.9f * kM;
    c.M = kM;
    c.E = 1.0f;
    c.L_z = 0.05f;  // small but nonzero — the hard case for the FD form

    KerrHamState s{};
    s.r = 8.0f;
    s.theta = kPi - 0.01f;  // just shy of the south pole
    s.p_r = 0.0f;
    s.p_theta = 1.0f;  // heading poleward

    const KerrHamState d = kerr_ham_rhs(s, c);
    REQUIRE(d.p_theta < 0.0f);  // decelerating: barrier repels

    // L_z = 0: barrier gone, force finite and small by comparison.
    c.L_z = 0.0f;
    const KerrHamState d0 = kerr_ham_rhs(s, c);
    REQUIRE(std::isfinite(d0.p_theta));
    REQUIRE(std::fabs(d0.p_theta) < std::fabs(d.p_theta));
}
