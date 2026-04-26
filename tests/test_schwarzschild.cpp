// tests/test_schwarzschild.cpp
//
// Exercises the CPU compile of shared_shader/geodesic_math.h. Because the
// backends include the same header unchanged, passing these tests is a
// necessary condition for every GPU backend to be correct too.
//
// See PHYSICS.md §§3, 5.2.

#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cmath>

#include "physics/integrator.hpp"
#include "physics/schwarzschild.hpp"

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

TEST_CASE("radial free-fall stays radial", "[physics][schwarzschild]") {
    // A massive test particle released at rest at r = 50 M. With u^r =
    // u^θ = u^φ = 0 initially, the only Christoffels that can activate
    // are Γ^r_{tt} (pulls r in) and Γ^t_{tr} (updates u^t as it falls).
    // u^θ and u^φ must stay zero — if they drift, a Christoffel has a bug.
    constexpr float M = 1.0f;
    constexpr float rs = 2.0f * M;

    State s{};
    s.t = 0.0f;
    s.r = 50.0f * M;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    s.ur = 0.0f;
    s.utheta = 0.0f;
    s.uphi = 0.0f;
    // Timelike normalisation g_μν u^μ u^ν = -1 with spatial velocity zero:
    //   -(1 - rs/r)(u^t)^2 = -1  =>  u^t = 1/sqrt(1 - rs/r).
    s.ut = 1.0f / std::sqrt(1.0f - rs / s.r);

    const float r_initial = s.r;
    for (int i = 0; i < 500; ++i) {
        s = rk4_step(s, 0.1f * M, rs);
        REQUIRE(std::fabs(s.utheta) < 1e-5f);
        REQUIRE(std::fabs(s.uphi) < 1e-5f);
        REQUIRE(std::fabs(s.theta - 0.5f * kPi) < 1e-5f);
        REQUIRE(std::fabs(s.phi) < 1e-5f);
    }
    REQUIRE(s.r < r_initial);  // test particle actually fell inward
    REQUIRE(s.r > rs);         // but hasn't crossed the horizon in 500 steps
}

TEST_CASE("circular timelike orbit at r = 6M stays circular", "[physics][schwarzschild]") {
    // Known closed-form circular orbit in Schwarzschild at r = r0, θ = π/2:
    //   u^t = 1 / sqrt(1 - 3M/r0)
    //   u^φ = sqrt(M / r0^3) * u^t       (Carroll §5.5 eq. 5.101)
    // At r = 6M (ISCO), this is the marginally stable orbit. Integrating
    // one period should leave r unchanged to the RK4 tolerance.
    constexpr float M = 1.0f;
    constexpr float rs = 2.0f * M;
    constexpr float r0 = 6.0f * M;

    State s{};
    s.t = 0.0f;
    s.r = r0;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    s.ur = 0.0f;
    s.utheta = 0.0f;
    s.ut = 1.0f / std::sqrt(1.0f - 3.0f * M / r0);
    s.uphi = std::sqrt(M / (r0 * r0 * r0)) * s.ut;

    // One full orbit in affine parameter: 2π / u^φ.
    const float period = 2.0f * kPi / s.uphi;
    const float h = 0.05f * M;
    const int n_steps = int(period / h);

    float r_min = r0, r_max = r0;
    for (int i = 0; i < n_steps; ++i) {
        s = rk4_step(s, h, rs);
        r_min = std::min(r_min, s.r);
        r_max = std::max(r_max, s.r);
        REQUIRE(std::fabs(s.theta - 0.5f * kPi) < 1e-4f);  // stays in plane
    }

    // Fixed-step RK4 at h = 0.05 M gets us ~0.1 %; 0.5 % is the TODO budget.
    const float tol = 0.005f * r0;
    REQUIRE(std::fabs(r_max - r0) < tol);
    REQUIRE(std::fabs(r_min - r0) < tol);
}

TEST_CASE("photon-sphere circular orbit at r = 1.5 r_s closes to 0.5%",
          "[physics][schwarzschild][smoke]") {
    // Null circular orbit at r = 3M with the critical impact parameter
    // b_crit = 3√3 M (PHYSICS.md §5.2). Any Christoffel sign error kicks
    // the photon off this unstable equilibrium fast, so this catches a
    // class of bugs that conserved-quantity checks alone would miss.
    constexpr float M = 1.0f;
    constexpr float rs = 2.0f * M;
    const float r0 = 1.5f * rs;  // 3M

    State s{};
    s.t = 0.0f;
    s.r = r0;
    s.theta = 0.5f * kPi;
    s.phi = 0.0f;
    s.ur = 0.0f;
    s.utheta = 0.0f;
    // For a circular null orbit, the null condition
    // -(1-rs/r)(u^t)^2 + r^2 (u^φ)^2 = 0  (with u^r, u^θ = 0)
    // combined with impact parameter b = L/E = r / sqrt(1 - rs/r) gives
    // u^t/u^φ = b. We normalise by setting u^φ = 1.
    s.uphi = 1.0f;
    s.ut = s.r / std::sqrt(1.0f - rs / s.r);

    // Small step; unstable equilibrium — big h runs away.
    const float h = 0.01f * M;
    const float period = 2.0f * kPi / s.uphi;
    const int n_steps = int(period / h);

    for (int i = 0; i < n_steps; ++i) {
        s = rk4_step(s, h, rs);
    }
    REQUIRE(std::fabs(s.r - r0) < 0.005f * r0);  // 0.5 % closure
}

// Operators the host-side templated integrator in physics/integrator.hpp
// uses on State. Declared at global scope (not in the anonymous namespace)
// so ADL finds them when the template instantiates `y + k1 * h` — ADL looks
// in the namespace of State, which is the global namespace. `inline` keeps
// them non-ODR-violating if any other TU in the tests/ directory ever needs
// the same pair.
inline State operator+(State a, State b) {
    return state_add(a, b);
}
inline State operator*(State a, float k) {
    return state_scale(a, k);
}

TEST_CASE("shared_shader dopri5_step matches host template on Schwarzschild geodesic",
          "[physics][integrator]") {
    // The host-side template dopri5_step in physics/integrator.hpp works for
    // any RHS type. The shared_shader dopri5_step is specialised to State +
    // the Schwarzschild RHS because shader languages (MSL/HLSL/WGSL) can't
    // compile C++ templates. Both implement the same Dormand-Prince 5(4)
    // Butcher tableau; feeding them the same state + RHS + step should
    // produce bit-close results. Disagreement beyond float32 round-off would
    // say one of the two tableaus has a coefficient bug.
    //
    // This is a tableau-correctness test, not a convergence test. Convergence
    // order on the shared_shader float32 integrator is hard to see from the
    // host side because both RK4 and DOPRI5 hit the float32 rounding floor
    // (~1e-6 on state values near 20) before truncation dominates cleanly;
    // the runtime GPU kernels will exercise convergence downstream.
    const float rs = 2.0f;
    const float h = 0.25f;
    const State y0{0.0f, 15.0f, kPi / 2.0f, 0.0f, 1.05f, -0.3f, 0.02f, 0.04f};

    auto rhs = [&](const State& s) { return geodesic_rhs_schwarzschild(s, rs); };
    const auto host = singularity::physics::dopri5_step(y0, h, rhs);
    const State gpu = dopri5_step(y0, h, rs);

    // Agreement within a few ulp (~5e-5 relative for values near 15). The
    // two implementations sum the Butcher-tableau terms in slightly
    // different orders, so float round-off drifts a bit but no coefficient
    // should be systematically wrong.
    const float tol = 5e-5f;
    REQUIRE(std::fabs(host.y_next.t - gpu.t) < tol);
    REQUIRE(std::fabs(host.y_next.r - gpu.r) < tol);
    REQUIRE(std::fabs(host.y_next.theta - gpu.theta) < tol);
    REQUIRE(std::fabs(host.y_next.phi - gpu.phi) < tol);
    REQUIRE(std::fabs(host.y_next.ut - gpu.ut) < tol);
    REQUIRE(std::fabs(host.y_next.ur - gpu.ur) < tol);
    REQUIRE(std::fabs(host.y_next.utheta - gpu.utheta) < tol);
    REQUIRE(std::fabs(host.y_next.uphi - gpu.uphi) < tol);
}
