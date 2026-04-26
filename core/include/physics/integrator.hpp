// core/include/physics/integrator.hpp
//
// Generic host-only integrators used by the unit tests (harmonic-oscillator
// convergence, etc.). The Schwarzschild-specialized rk4_step that GPU
// backends call lives in shared_shader/geodesic_math.h — a template can't
// compile through HLSL/MSL, so the specialised form is kept duplicate-free
// by having both call the same Christoffel code.
//
// See PHYSICS.md §6 for the numerical-integration context.

#ifndef SINGULARITY_PHYSICS_INTEGRATOR_HPP
#define SINGULARITY_PHYSICS_INTEGRATOR_HPP

namespace singularity::physics {

// Classical fourth-order Runge-Kutta. State must support
//   State operator+(State, State)
//   State operator*(State, float)
// RHS is callable as: State rhs(const State&).
template <typename State, typename RHS>
inline State rk4_step(const State& y, float h, RHS rhs) {
    State k1 = rhs(y);
    State k2 = rhs(y + k1 * (0.5f * h));
    State k3 = rhs(y + k2 * (0.5f * h));
    State k4 = rhs(y + k3 * h);
    return y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (h / 6.0f);
}

// Forward Euler. See PHYSICS.md §6.1 — included only to let the toy mode
// illustrate why it's inadequate.
template <typename State, typename RHS>
inline State euler_step(const State& y, float h, RHS rhs) {
    return y + rhs(y) * h;
}

// Dormand-Prince 5(4) single step — 5th-order solution with an embedded
// 4th-order error estimate. PHYSICS.md §6.3; Butcher tableau from Dormand &
// Prince, J. Comp. Appl. Math. 6 (1980) 19-26. Seven RHS evaluations per
// step; the error estimate comes for free from the two different b-vectors
// operating on the same k_i's.
//
// Usage:
//     auto [y_next, err] = dopri5_step(y, h, rhs);
// Where err has the same type as State and stores (y_5th − y_4th). Norm it
// and compare against a tolerance to drive adaptive step-size control; a
// lightweight adaptive driver lives below in dopri5_adaptive_step.
template <typename State, typename RHS>
struct DOPRI5Result {
    State y_next;
    State error;
};

template <typename State, typename RHS>
inline DOPRI5Result<State, RHS> dopri5_step(const State& y, float h, RHS rhs) {
    // Stage coefficients (a_ij) — the Butcher tableau's lower-triangular half.
    constexpr float a21 = 1.0f / 5.0f;

    constexpr float a31 = 3.0f / 40.0f;
    constexpr float a32 = 9.0f / 40.0f;

    constexpr float a41 = 44.0f / 45.0f;
    constexpr float a42 = -56.0f / 15.0f;
    constexpr float a43 = 32.0f / 9.0f;

    constexpr float a51 = 19372.0f / 6561.0f;
    constexpr float a52 = -25360.0f / 2187.0f;
    constexpr float a53 = 64448.0f / 6561.0f;
    constexpr float a54 = -212.0f / 729.0f;

    constexpr float a61 = 9017.0f / 3168.0f;
    constexpr float a62 = -355.0f / 33.0f;
    constexpr float a63 = 46732.0f / 5247.0f;
    constexpr float a64 = 49.0f / 176.0f;
    constexpr float a65 = -5103.0f / 18656.0f;

    constexpr float a71 = 35.0f / 384.0f;
    constexpr float a73 = 500.0f / 1113.0f;
    constexpr float a74 = 125.0f / 192.0f;
    constexpr float a75 = -2187.0f / 6784.0f;
    constexpr float a76 = 11.0f / 84.0f;

    // Weights for the 5th-order solution (b_i) and for the error estimate
    // (e_i = b_i − b_i*, where b_i* is the embedded 4th-order formula).
    // Note the 5th-order uses b_7 = 0 (FSAL property), so y_next is assembled
    // from k1..k6.
    constexpr float e1 = 71.0f / 57600.0f;
    constexpr float e3 = -71.0f / 16695.0f;
    constexpr float e4 = 71.0f / 1920.0f;
    constexpr float e5 = -17253.0f / 339200.0f;
    constexpr float e6 = 22.0f / 525.0f;
    constexpr float e7 = -1.0f / 40.0f;

    State k1 = rhs(y);
    State k2 = rhs(y + k1 * (a21 * h));
    State k3 = rhs(y + (k1 * a31 + k2 * a32) * h);
    State k4 = rhs(y + (k1 * a41 + k2 * a42 + k3 * a43) * h);
    State k5 = rhs(y + (k1 * a51 + k2 * a52 + k3 * a53 + k4 * a54) * h);
    State k6 = rhs(y + (k1 * a61 + k2 * a62 + k3 * a63 + k4 * a64 + k5 * a65) * h);

    State y_next = y + (k1 * a71 + k3 * a73 + k4 * a74 + k5 * a75 + k6 * a76) * h;
    State k7 = rhs(y_next);

    State err = (k1 * e1 + k3 * e3 + k4 * e4 + k5 * e5 + k6 * e6 + k7 * e7) * h;
    return DOPRI5Result<State, RHS>{y_next, err};
}

}  // namespace singularity::physics

#endif  // SINGULARITY_PHYSICS_INTEGRATOR_HPP
