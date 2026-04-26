// core/include/physics/disc.hpp
//
// Thin-disc emission physics for the accretion disc around a Schwarzschild
// or Kerr black hole. PHYSICS.md §8.3. This header holds only the radial
// temperature profile and the inner/outer edge computation — the colour
// mapping (blackbody → sRGB via CIE) and the pixel-level combination with
// the §8.1–§8.2 Doppler / redshift machinery live in the disc kernel proper
// once Phase 3 begins to render.
//
// The model is the Novikov-Thorne (1973) simplification: a cold, geometrically
// thin, optically thick disc in the equatorial plane, radiating as a
// position-dependent blackbody. Temperature is a power law in radius with the
// inner-edge correction that drops the flux to zero at the innermost stable
// circular orbit (ISCO). In the simplified form used for rendering —
//
//     T(r) ∝ (r_ISCO / r)^(3/4)   for r ≥ r_ISCO
//
// — the flux falls off as r^(−3) (Stefan-Boltzmann times temperature⁴), which
// gives the classic hot-inner-ring / cool-outer-disc visual signature seen in
// Interstellar and the EHT rendering passes. Inside r_ISCO the model is
// undefined because a stable orbit cannot exist there; the disc's inner edge
// is the ISCO (Schwarzschild: 6M; Kerr: r_ISCO_prograde from kerr.hpp §7.2).

#ifndef SINGULARITY_PHYSICS_DISC_HPP
#define SINGULARITY_PHYSICS_DISC_HPP

#include <cmath>

namespace singularity::physics {

// Radial temperature profile of a thin, optically-thick accretion disc,
// simplified to the dominant ``T ∝ r^(−3/4)`` Novikov-Thorne power law.
// Anchored at a reference point ``(r_ref, T_ref)`` — typically the ISCO and
// the characteristic inner-edge temperature for the scenario (e.g.,
// T_ref ≈ 1e7 K for a 10 M☉ BH accreting at the Eddington limit; ≈1e4 K for
// Sgr A*). The same formula rescales cleanly: in code we track only the
// ratio T/T_ref to stay unit-agnostic.
inline float disc_temperature(float r, float r_ref, float t_ref) {
    return t_ref * std::pow(r_ref / r, 0.75f);
}

// Novikov-Thorne inner-edge suppression factor `f(r, r_ISCO)`, vanishing at
// the ISCO and asymptoting to 1 far away:
//
//     f(r, r_ISCO) = 1 − √(r_ISCO / r)
//
// The true flux profile is T(r)^4 ∝ f(r) · r^(−3), so the *temperature* picks
// up a fourth-root of this factor. For rendering the simpler T ∝ r^(−3/4)
// profile is usually adequate; this factor is exposed for scenes where the
// inner-edge darkening is visually important (e.g., Interstellar-scale
// closeups of the ISCO). PHYSICS.md §8.3 refs; see also Page & Thorne 1974
// §II.B for the derivation.
inline float disc_inner_edge_factor(float r, float r_isco) {
    if (r <= r_isco)
        return 0.0f;
    return 1.0f - std::sqrt(r_isco / r);
}

// Full Novikov-Thorne simplified temperature with inner-edge suppression
// folded in: T(r) = T_ref · (r_ref/r)^(3/4) · f(r, r_ISCO)^(1/4).
// Reduces to ``disc_temperature`` far from the ISCO.
inline float disc_temperature_nt(float r, float r_ref, float t_ref, float r_isco) {
    const float power_law = disc_temperature(r, r_ref, t_ref);
    const float edge = disc_inner_edge_factor(r, r_isco);
    if (edge <= 0.0f)
        return 0.0f;
    return power_law * std::pow(edge, 0.25f);
}

// Stefan-Boltzmann bolometric flux per unit emitting area, proportional to
// T^4. In geometrized units the constant is absorbed into the overall
// brightness — we only ever use *ratios* of flux across the disc — so this
// returns the dimensionless fourth-power that the disc kernel multiplies
// into a scene-wide luminosity scale.
inline float disc_flux_ratio(float temperature, float t_ref) {
    const float ratio = temperature / t_ref;
    const float r2 = ratio * ratio;
    return r2 * r2;
}

// Wien's displacement law — peak wavelength (metres) of the Planck spectrum
// for a body at ``temperature_K``. ``b = 2.898 × 10⁻³ m·K``. Used by the
// disc colour LUT to decide where the spectrum sits in the visible range
// (disc inner edge at ~1e7 K peaks in X-rays; outer at ~1e3 K peaks in
// infrared). PHYSICS.md §8.3.
inline constexpr float kWienConstant_mK = 2.898e-3f;

inline float wien_peak_wavelength_m(float temperature_K) {
    return kWienConstant_mK / temperature_K;
}

// Inverse — temperature at which the blackbody spectrum peaks at a given
// wavelength. Useful when matching "what temperature for this colour?"
inline float wien_temperature_K(float peak_wavelength_m) {
    return kWienConstant_mK / peak_wavelength_m;
}

// Blackbody → sRGB colour approximation (Tanner Helland), valid over the
// 1000 K — 40000 K range that covers everything from the cool outer edge of
// a thin disc (~1e3 K, deep red) to the ionised plasma close to ISCO
// (~1e5 K; the upper formula domain is extrapolated there but the colour is
// saturated blue-white already). Output in linear sRGB (not gamma-encoded),
// in [0, 1], ready to multiply into a flux term.
//
// This is a piecewise logarithmic / power-law fit to the full Planck ×
// CIE 1931 2° observer × sRGB primaries chain. The full integration lives
// in the Phase 3 LUT builder — for rendering we precompute the table and
// this fit is the ground truth the LUT is checked against.
//
// The shape is chosen to match Mitchell Charity's blackbody colour table
// and the corresponding Tanner Helland empirical formula widely used in
// astronomy-visualisation contexts. Values expressed as 0—1 floats rather
// than 0—255 ints so the caller can multiply in their own luminosity scale.
struct sRGBColor {
    float r, g, b;
};

inline float _clamp01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

inline sRGBColor blackbody_srgb_tanner_helland(float temperature_K) {
    // Work in centikelvin per the original formula. The piecewise form hinges
    // on T' = T/100 crossing 66 (roughly 6600 K, near daylight D65).
    const float t = temperature_K * 0.01f;

    float r, g, b;

    if (t < 66.0f) {
        r = 1.0f;
    } else {
        // Warm branch fades from white to blue-white; this power law drops R.
        r = 329.698727446f * std::pow(t - 60.0f, -0.1332047592f);
        r /= 255.0f;
    }

    if (t < 66.0f) {
        g = 99.4708025861f * std::log(t) - 161.1195681661f;
        g /= 255.0f;
    } else {
        g = 288.1221695283f * std::pow(t - 60.0f, -0.0755148492f);
        g /= 255.0f;
    }

    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) {
        b = 0.0f;
    } else {
        b = 138.5177312231f * std::log(t - 10.0f) - 305.0447927307f;
        b /= 255.0f;
    }

    return {_clamp01(r), _clamp01(g), _clamp01(b)};
}

}  // namespace singularity::physics

#endif  // SINGULARITY_PHYSICS_DISC_HPP
