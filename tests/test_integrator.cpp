// tests/test_integrator.cpp
//
// Generic RK4 / Euler smoke. The Schwarzschild integrator in
// shared_shader/geodesic_math.h uses the same scheme but specialised on
// State; here we convince ourselves the scheme itself is 4th-order by
// driving it with a problem that has a closed-form solution — the simple
// harmonic oscillator.

#include <catch_amalgamated.hpp>

#include <cmath>

#include "physics/integrator.hpp"

namespace {

// Double-precision so the 4th-order convergence ratio is visible well
// above the rounding floor. The Schwarzschild integrator uses float for
// GPU parity — its convergence is verified by test_phase1_rays.py against
// an independent SciPy reference.
struct Vec2 {
    double x;
    double v;
};

inline Vec2 operator+(Vec2 a, Vec2 b) {
    return {a.x + b.x, a.v + b.v};
}
inline Vec2 operator*(Vec2 a, float k) {
    const double d = double(k);
    return {a.x * d, a.v * d};
}

// y'' + y = 0  =>  (dx/dt, dv/dt) = (v, -x).
Vec2 sho_rhs(const Vec2& s) {
    return {s.v, -s.x};
}

double integrate_sho_rk4(int n_steps, double t_end) {
    Vec2 y{1.0, 0.0};  // x(0)=1, v(0)=0
    const float h = float(t_end / double(n_steps));
    for (int i = 0; i < n_steps; ++i) {
        y = singularity::physics::rk4_step(y, h, sho_rhs);
    }
    return std::fabs(y.x - std::cos(t_end));
}

}  // namespace

TEST_CASE("RK4 shows 4th-order convergence on harmonic oscillator", "[physics][integrator]") {
    // Integrate long enough (~1.6 periods) that accumulated error is well
    // above double-precision noise — otherwise halving h just measures
    // rounding rather than the method's truncation error.
    const double t_end = 10.0;
    const double err_64 = integrate_sho_rk4(64, t_end);
    const double err_128 = integrate_sho_rk4(128, t_end);
    const double err_256 = integrate_sho_rk4(256, t_end);

    REQUIRE(err_256 < 1e-3);

    // Halving h cuts error by ~16x (O(h^4)) in the asymptotic regime.
    // Allow factor-of-7 lower bound to absorb Debug-build optimisation
    // differences (g++ -O0 loses some associativity-based optimisations
    // that reduce float round-off) and constant-factor drift.
    REQUIRE(err_64 / err_128 > 7.0);
    REQUIRE(err_128 / err_256 > 7.0);
}

TEST_CASE("Euler tracks the oscillator coarsely but stays bounded", "[physics][integrator]") {
    // 10 s of SHO at h = 0.01. Forward Euler inflates amplitude but must
    // stay bounded; if this fails, the euler_step template is broken.
    Vec2 y{1.0, 0.0};
    for (int i = 0; i < 1000; ++i) {
        y = singularity::physics::euler_step(y, 0.01f, sho_rhs);
    }
    REQUIRE(std::isfinite(y.x));
    REQUIRE(std::isfinite(y.v));
    REQUIRE(std::fabs(y.x) < 5.0);
    REQUIRE(std::fabs(y.v) < 5.0);
}

namespace {

double integrate_sho_dopri5(int n_steps, double t_end) {
    Vec2 y{1.0, 0.0};
    const float h = float(t_end / double(n_steps));
    for (int i = 0; i < n_steps; ++i) {
        auto result = singularity::physics::dopri5_step(y, h, sho_rhs);
        y = result.y_next;
    }
    return std::fabs(y.x - std::cos(t_end));
}

}  // namespace

TEST_CASE("DOPRI5 shows 5th-order convergence on harmonic oscillator", "[physics][integrator]") {
    // Same SHO integrated for t_end = 10.0 (~1.6 periods). Halving h should
    // cut error by 32x (O(h^5)) in the asymptotic regime. Below h ≈ 0.1 the
    // float32 Butcher-tableau coefficients accumulate ~1e-7 of round-off that
    // dominates the truncation error, so we stay in the regime n ≤ 64 where
    // the 5th-order scaling is cleanly visible. Moving the coefficients to
    // double-precision would push the floor far below 1e-10 but would also
    // diverge from the float-only GPU-port target; this test is the honest
    // check on the integrator as the GPU backends will consume it.
    const double t_end = 10.0;
    const double err_16 = integrate_sho_dopri5(16, t_end);
    const double err_32 = integrate_sho_dopri5(32, t_end);
    const double err_64 = integrate_sho_dopri5(64, t_end);

    REQUIRE(err_64 < 1e-5);
    REQUIRE(err_16 / err_32 > 16.0);
    REQUIRE(err_32 / err_64 > 16.0);
}

TEST_CASE("DOPRI5 error estimate tracks the true error to within a constant",
          "[physics][integrator]") {
    // Driving the SHO for a single large step and comparing the embedded
    // 4th-order error estimate to the 5th-order-vs-closed-form error. They
    // should agree to within a small constant (the error-estimate formula is
    // conservative by ~1.5-3x for smooth problems; a complete mismatch would
    // indicate the e_i coefficients are wrong).
    const float h = 0.3f;
    Vec2 y0{1.0, 0.0};
    auto result = singularity::physics::dopri5_step(y0, h, sho_rhs);

    const double true_err = std::fabs(result.y_next.x - std::cos(double(h)));
    const double est_err = std::fabs(result.error.x);

    // Both should be tiny (DOPRI5 at h = 0.3 on SHO produces err ~ 10⁻⁷) and
    // within 10x of each other; a constant-factor mismatch would suggest the
    // embedded b* weights are subtly wrong.
    REQUIRE(true_err < 1e-5);
    REQUIRE(est_err < 1e-5);
    REQUIRE(est_err / true_err > 0.1);
    REQUIRE(est_err / true_err < 10.0);
}
