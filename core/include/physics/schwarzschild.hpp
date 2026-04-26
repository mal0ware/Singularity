// core/include/physics/schwarzschild.hpp
//
// Host-side access to the Schwarzschild geodesic machinery. The Christoffel
// symbols, geodesic RHS, and RK4 stepper all live in
// shared_shader/geodesic_math.h — that's the file every GPU backend also
// includes — so the CPU path we unit-test and the GPU path we render are
// literally the same code. See docs/ARCHITECTURE.md §4 for the design.
//
// This header adds only the host-only conveniences (named constants, SI
// conversions) that don't belong in a file GPU compilers also ingest.

#ifndef SINGULARITY_PHYSICS_SCHWARZSCHILD_HPP
#define SINGULARITY_PHYSICS_SCHWARZSCHILD_HPP

#include "geodesic_math.h"

namespace singularity::physics {

// Schwarzschild radius rs = 2M in geometrized units (c = G = 1).
// PHYSICS.md §2.
inline constexpr float schwarzschild_radius(float mass_geom) {
    return 2.0f * mass_geom;
}

// ISCO for a timelike circular orbit in Schwarzschild. PHYSICS.md §5.2 foot.
inline constexpr float isco_timelike(float mass_geom) {
    return 6.0f * mass_geom;
}

// Photon-sphere radius. PHYSICS.md §5.2.
inline constexpr float photon_sphere_radius(float mass_geom) {
    return 3.0f * mass_geom;
}

}  // namespace singularity::physics

#endif  // SINGULARITY_PHYSICS_SCHWARZSCHILD_HPP
