// core/include/physics/vec_types.hpp
//
// Host-side vec3 / vec4 so the shared shader math can be compiled on the CPU
// (unit tests, verification harness). GPU backends get their vector types from
// the platform SDK — see shared_shader/shader_compat.h for the dispatch.
//
// The surface intentionally stays small; extend alongside the geodesic / camera
// math that actually uses it. We don't implement swizzles — none of the shared
// shader code uses them, and keeping the type plain keeps it trivially copyable
// and byte-identical to float[N] (important for device/host parity).

#ifndef SINGULARITY_PHYSICS_VEC_TYPES_HPP
#define SINGULARITY_PHYSICS_VEC_TYPES_HPP

#include <cmath>

namespace singularity {

struct vec3 {
    float x, y, z;
};

struct vec4 {
    float x, y, z, w;
};

inline vec3 operator+(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline vec3 operator-(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline vec3 operator*(vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}
inline vec3 operator*(float s, vec3 a) {
    return a * s;
}

inline float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline float length(vec3 a) {
    return std::sqrt(dot(a, a));
}
inline vec3 normalize(vec3 a) {
    return a * (1.0f / length(a));
}
inline vec3 cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

}  // namespace singularity

using singularity::vec3;
using singularity::vec4;

#endif  // SINGULARITY_PHYSICS_VEC_TYPES_HPP
