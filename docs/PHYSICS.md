---
layout: default
title: Physics
permalink: /physics/
---

# Physics Reference — Singularity

> The general relativity required to understand and validate every line of code in `singularity/core/`. This document is the single source of truth for the physics. Code comments should reference section numbers here, not duplicate the math.

**Implementation note (cross-cutting).** All physics in this document is implemented in shared C++ headers under `core/include/physics/` and reused without modification across all GPU backends — Metal Shading Language (MSL is C++14-based), HLSL → SPIR-V → Vulkan, CUDA C++, and WebGPU/WASM. The only platform-specific code is the kernel boilerplate that calls these shared functions. This means a physics bug is fixed in one place; it cannot drift between backends. The backend-equivalence test (`verification/test_backend_equivalence.py`) enforces this property at CI time.

**Conventions used throughout:**

- Signature: `(-, +, +, +)` (the "mostly plus" convention used by Carroll and Wald).
- Units: geometrized units `c = G = 1` for derivations; SI units for code that interfaces with user-facing quantities (mass in solar masses, distances in km).
- Indices: Greek letters (μ, ν, ρ, σ) range over 0–3; Latin letters (i, j, k) range over 1–3.
- Einstein summation convention applied throughout.
- Schwarzschild radius written `r_s = 2M` (geometrized) or `2GM/c²` (SI).
- Kerr spin parameter written `a = J/M` (geometrized) with `0 ≤ a ≤ M`.

**Primary references** (cited inline by short tag):
- **MTW** — Misner, Thorne, Wheeler. *Gravitation*. (1973, reissued 2017.)
- **Carroll** — Sean Carroll. *Spacetime and Geometry: An Introduction to General Relativity*. (2003.)
- **Thorne** — Kip Thorne. *The Science of Interstellar*. (2014.)
- **JMO** — James, Tunzelmann, Franklin, Thorne. "Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar." *Class. Quantum Grav.* 32 (2015) 065001. (The actual paper from the Interstellar VFX team.)

---

## 1. Spacetime, geodesics, and what we are actually computing

In Newtonian physics, a black hole pulls light "down" via gravity. This picture is wrong and the simulator must not be built on it. In general relativity, mass-energy curves the geometry of spacetime, and **light always travels in straight lines through that geometry** — the lines just no longer look straight when projected into our Euclidean intuition, in exactly the same way the great-circle route from London to Tokyo looks "curved" on a Mercator map.

The mathematical object describing "the straightest possible line in a curved space" is a **geodesic**. For light specifically, we want **null geodesics** — paths along which the spacetime interval `ds² = 0`, meaning a photon's proper time does not advance. The simulator's job, per pixel, is to integrate the null geodesic equation backwards from the camera until the ray either:

1. crosses the event horizon (color the pixel black),
2. intersects the accretion disc (sample disc temperature, apply Doppler + redshift, color),
3. escapes to spatial infinity (sample the celestial-sphere skybox), or
4. exceeds an integration step budget (color magenta — "we ran out of compute budget here," for debugging).

That's the whole simulator at the conceptual level. Everything below is the math for executing it correctly.

## 2. The Schwarzschild metric

In 1916, Karl Schwarzschild solved the Einstein field equations for the simplest non-trivial case: the spacetime outside a non-rotating, spherically symmetric mass `M`, with everything else in the universe set to zero. The result, in spherical coordinates `(t, r, θ, φ)`:

```
ds² = -(1 - r_s/r) dt² + (1 - r_s/r)⁻¹ dr² + r²(dθ² + sin²θ dφ²)
```

with `r_s = 2M` (geometrized) or `2GM/c²` (SI).

**Why this matters for the code:** the components of the metric tensor `g_μν` appear directly in every Christoffel symbol calculation. They are:

```
g_tt = -(1 - r_s/r)
g_rr =  (1 - r_s/r)⁻¹
g_θθ =  r²
g_φφ =  r² sin²θ
```

All off-diagonal components vanish. The inverse metric `g^μν`:

```
g^tt = -(1 - r_s/r)⁻¹
g^rr =  (1 - r_s/r)
g^θθ =  1/r²
g^φφ =  1/(r² sin²θ)
```

**Singularities of the metric.** There are two:
- `r = 0` is a true curvature singularity (the Kretschmann scalar `R_μνρσ R^μνρσ` diverges). No coordinate change removes it.
- `r = r_s` is a *coordinate* singularity. Schwarzschild coordinates break down here, but spacetime is locally smooth. The simulator handles this by terminating ray integration when `r < r_s + ε` for small `ε` (e.g., `0.01 r_s`).

## 3. Christoffel symbols for Schwarzschild

The Christoffel symbols `Γ^μ_νσ` encode the curvature of spacetime. Defined by:

```
Γ^μ_νσ = ½ g^μρ (∂_ν g_ρσ + ∂_σ g_ρν - ∂_ρ g_νσ)
```

For Schwarzschild, the non-zero components are (let `f = 1 - r_s/r` for compactness):

```
Γ^t_tr = Γ^t_rt =  r_s / (2 r² f)
Γ^r_tt =  r_s f / (2 r²)
Γ^r_rr = -r_s / (2 r² f)
Γ^r_θθ = -r f
Γ^r_φφ = -r f sin²θ
Γ^θ_rθ = Γ^θ_θr = 1/r
Γ^θ_φφ = -sin θ cos θ
Γ^φ_rφ = Γ^φ_φr = 1/r
Γ^φ_θφ = Γ^φ_φθ = cot θ
```

(Reference: Carroll §5.5; MTW Box 25.2.)

**Verification step:** `verification/christoffel_sympy.py` re-derives these symbolically with SymPy and asserts equality with the hand-coded versions in `core/include/physics/schwarzschild.hpp`. If you ever edit the metric, the test catches the algebra error before it reaches any GPU.

## 4. The geodesic equation

A geodesic `x^μ(λ)` parameterized by an affine parameter `λ` satisfies:

```
d²x^μ / dλ² + Γ^μ_νσ (dx^ν/dλ)(dx^σ/dλ) = 0
```

This is one second-order ODE per coordinate — four ODEs total. In code we reduce to first-order by introducing the four-velocity `u^μ = dx^μ/dλ`, giving an eight-dimensional state vector `(x^μ, u^μ)` and eight first-order ODEs:

```
dx^μ / dλ = u^μ
du^μ / dλ = -Γ^μ_νσ u^ν u^σ
```

This is the form integrated by every backend's compute kernel.

**Null condition.** For a photon, the four-velocity must satisfy `g_μν u^μ u^ν = 0`. Initialized at ray generation; checked at each step; if violation grows beyond `10⁻⁴` of `E²`, integrator step size shrinks.

## 5. Conserved quantities and the effective potential

Integrating eight raw ODEs is wasteful when the symmetries of Schwarzschild spacetime hand us conserved quantities for free.

### 5.1 Energy and angular momentum

The Schwarzschild metric is independent of `t` and `φ`, so by Noether-type arguments two quantities are conserved along geodesics:

- **Energy:** `E = -g_tμ u^μ = (1 - r_s/r) dt/dλ`
- **Angular momentum:** `L = g_φμ u^μ = r² sin²θ · dφ/dλ`

For motion in the equatorial plane (`θ = π/2`), the geodesic reduces to a 1D radial motion in an effective potential:

```
(dr/dλ)² = E² - (1 - r_s/r)(L²/r²)
```

This 1D form is what lets us derive the photon sphere analytically.

### 5.2 The photon sphere

Setting `dr/dλ = 0` and `d²r/dλ² = 0` (a circular photon orbit) gives:

```
r_photon = (3/2) r_s = 3M
```

at impact parameter `b_crit = (3√3 / 2) r_s ≈ 2.598 r_s`.

A photon arriving with `b < b_crit` falls into the black hole; one with `b > b_crit` escapes; one with `b = b_crit` orbits forever (unstable equilibrium). This is the **photon ring** visible in Event Horizon Telescope images.

**Test F12 in the verification harness** integrates a circular orbit at `r_photon` with the simulator's RK4 (run on each backend) and asserts the orbit closes within 0.5%.

### 5.3 Carter constant (3D case)

For non-equatorial geodesics in Schwarzschild, conservation of total angular momentum `L²` (not just `L_z`) gives a third constant `Q = L_total² - L_z²`. In Schwarzschild this is derivable from spherical symmetry; in Kerr it becomes the **Carter constant**, an *additional* conserved quantity that makes Kerr geodesics integrable.

Introduced now in Schwarzschild as a test harness so the Kerr port (Phase 6) is mechanical.

## 6. Numerical integration

### 6.1 Why Euler is wrong (and why the first source video gets it slightly wrong)

The first source video starts with Euler's method:

```
x_{n+1} = x_n + h · v_n
v_{n+1} = v_n + h · a_n
```

First-order accurate — error per step `O(h²)`, global error `O(h)`. For a curved geodesic near a black hole, where the "force" changes rapidly with position, Euler steps overshoot in straight lines and the trajectory looks artificially angular. The video acknowledges this and switches to RK4.

### 6.2 RK4

Classical fourth-order Runge-Kutta with fixed step size:

```
k1 = f(y_n)
k2 = f(y_n + h/2 · k1)
k3 = f(y_n + h/2 · k2)
k4 = f(y_n + h   · k3)
y_{n+1} = y_n + h/6 · (k1 + 2 k2 + 2 k3 + k4)
```

Local error `O(h⁵)`, global error `O(h⁴)`. Cost: four geodesic-RHS evaluations per step. Acceptable for Phases 1–6.

### 6.3 Dormand-Prince 5(4) — adaptive

For Phase 7 polish we upgrade to **DOPRI5**, a 5th-order method with embedded 4th-order error estimate. The error estimate lets us shrink the step size near the horizon (where curvature is fierce) and grow it at large `r` (where spacetime is nearly flat). Net result: same accuracy with ~5× fewer steps for typical scenes.

Reference implementation: Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I* §II.5.

### 6.4 Conserved-quantity check

After every N steps (e.g., N=50), the integrator recomputes `E`, `L`, and the null condition `g_μν u^μ u^ν` from the current state and compares to the initial values. If drift exceeds tolerance:

- in the GPU kernel, we tag the pixel for the next frame's diagnostic overlay,
- in the verification harness, the test fails.

This is the single most important safeguard against silent integrator bugs. It does not catch *systematic* errors in the Christoffel symbols (those preserve `E` and `L` while still being wrong), which is why §3 has a separate symbolic-derivation test.

## 7. The Kerr metric

A rotating black hole. In Boyer-Lindquist coordinates `(t, r, θ, φ)`:

```
ds² = -(1 - r_s r / Σ) dt²
     - (2 r_s r a sin²θ / Σ) dt dφ
     + (Σ / Δ) dr²
     + Σ dθ²
     + (r² + a² + r_s r a² sin²θ / Σ) sin²θ dφ²
```

where:

```
Σ = r² + a² cos²θ
Δ = r² - r_s r + a²
```

The off-diagonal `dt dφ` term encodes **frame dragging** — even a stationary observer is forced to rotate with the black hole.

### 7.1 Horizons and the ergosphere

- **Outer event horizon:** `r₊ = M + √(M² - a²)`
- **Inner (Cauchy) horizon:** `r₋ = M - √(M² - a²)`
- **Ergosphere outer boundary:** `r_ergo(θ) = M + √(M² - a² cos²θ)`

Inside the ergosphere, no observer can remain stationary in `(r, θ, φ)`. Visualized as a separate wireframe in Phase 6 scientific overlay.

### 7.2 ISCO

The Innermost Stable Circular Orbit for matter:

```
r_ISCO = M [3 + Z₂ ∓ √((3 - Z₁)(3 + Z₁ + 2 Z₂))]
Z₁ = 1 + (1 - a²/M²)^(1/3) [(1 + a/M)^(1/3) + (1 - a/M)^(1/3)]
Z₂ = √(3 a²/M² + Z₁²)
```

Upper sign: prograde (co-rotating). Lower sign: retrograde.

For `a = 0` (Schwarzschild): `r_ISCO = 6M`.
For `a = M` (extremal Kerr, prograde): `r_ISCO = M`.
For `a = M` (extremal Kerr, retrograde): `r_ISCO = 9M`.

### 7.3 Carter constant and integrable Kerr geodesics

Kerr geodesics admit **four** conserved quantities:

- `E` (energy at infinity)
- `L_z` (axial angular momentum)
- `m²` (rest mass — zero for photons)
- `Q` (Carter constant — discovered 1968)

With four conserved quantities and four ODEs, Kerr geodesics are **completely integrable** — we can write `dr/dλ`, `dθ/dλ`, `dφ/dλ`, `dt/dλ` as algebraic expressions in `(r, θ, E, L_z, Q)` rather than integrating Christoffel symbols. Dramatically faster on the GPU and avoids the numerical instability of explicit Kerr Christoffels (23 nonzero components vs Schwarzschild's 9).

Used throughout Phase 6. Reference: Carroll §6.7; JMO §3.

## 8. Null geodesics, redshift, and what the camera receives

### 8.1 Gravitational redshift

A photon emitted at `r_emit` with frequency `ν_emit` is detected at `r_obs` with frequency:

```
ν_obs / ν_emit = √(g_tt(r_emit) / g_tt(r_obs))
              = √((1 - r_s/r_emit) / (1 - r_s/r_obs))   [Schwarzschild]
```

(Derivation: a photon with conserved energy-at-infinity `E` is measured at
frequency `ω = E / √(1 - r_s/r)` by a static observer at radius `r`; taking
the ratio of the emitter's and observer's local clocks gives the factor
above. With `r_emit < r_obs` the emitter's tick runs slower, so the photon
arrives with lower frequency — the redshift.)

For `r_obs → ∞`:

```
ν_∞ / ν_emit = √(1 - r_s/r_emit)
```

Disc material at `r = 6M` is observed redshifted by `√(1 - r_s/r) = √(1 - 2/6) = √(2/3) ≈ 0.816` (using the `r_s = 2M` convention from §2).

### 8.2 Doppler beaming

Disc material moves relativistically. For an observer with four-velocity `u_obs`:

```
ν_obs = -p_μ u_obs^μ
```

For the disc: `u_obs^μ` is the four-velocity of a circular geodesic at the emission point (Schwarzschild: prograde Keplerian; Kerr: prograde Boyer-Lindquist circular). Combining with §8.1 gives total shift `g = ν_obs / ν_emit`.

Observed specific intensity transforms as:

```
I_obs / I_emit = g⁴
```

(Rybicki & Lightman, *Radiative Processes in Astrophysics*, §4.9.)

This is why one side of *Interstellar's* Gargantua disc *should* be substantially brighter than the other — and why Nolan's team chose to suppress the asymmetry. Singularity restores it (with a toggle).

### 8.3 Color from temperature

Disc material at radius `r` has effective temperature (simplified Novikov-Thorne):

```
T(r) ∝ (M_dot · M / r³)^(1/4) · f(r, a)
```

where `f(r, a)` is a relativistic correction vanishing at ISCO. We map `T(r)` to a perceptual blackbody color (Planck spectrum integrated against CIE color-matching functions, then mapped to sRGB) via a precomputed 1D LUT. Hot inner edge → blue-white. Cool outer edge → red-orange.

Redshift factor `g` from §8.2 then applied as a color shift on top.

## 9. Tidal forces (context, not rendered)

```
F_tidal ≈ (2 G M / r³) · dr · m
```

Useful as docs-site context (re: *Interstellar's* Miller's planet). Doesn't render.

## 10. Wormhole metric (Phase 9 stretch)

The Morris-Thorne wormhole used in *Interstellar*:

```
ds² = -dt² + dl² + (b² + l²) (dθ² + sin²θ dφ²)
```

`l` is a proper-distance coordinate ranging over `(-∞, +∞)`, with `l = 0` at the throat of radius `b`. Two universes connect at the throat. Geodesic integration is conceptually identical to Schwarzschild — different metric, same kernel structure. Renderer needs a second skybox for the "other side" universe.

Reference: Morris & Thorne, *Am. J. Phys.* 56, 395 (1988).

## 11. Validation strategy

Every physics module ships with a verification test. The matrix:

| What | Analytical result | Test code | Tolerance |
|---|---|---|---|
| Schwarzschild Christoffels | SymPy symbolic derivation | `verification/christoffel_sympy.py` | exact (rational arithmetic) |
| Photon sphere | `r = 1.5 r_s` | `verification/test_photon_sphere.py` | 0.5% |
| Weak-field deflection | `Δφ = 4GM/(bc²)` for `b ≫ r_s` | `verification/test_deflection.py` | 1% |
| Energy conservation | `E` constant along geodesic | `verification/test_conserved.py` | 10⁻⁶ over 10K steps |
| Kerr ISCO (a/M = 0, 0.5, 0.94, 0.998) | §7.2 closed form | `verification/test_isco.py` | 0.1% |
| Kerr horizon radii | `M ± √(M² - a²)` | `verification/test_kerr_horizons.py` | 0.01% |
| Gravitational redshift | §8.1 | `verification/test_redshift.py` | 0.5% |
| Carter constant conservation (Kerr) | constant | `verification/test_carter.py` | 10⁻⁵ over 5K steps |
| Golden image renders, per backend | Stored PNG, 256² each | `verification/test_golden_images.py` | perceptual-hash distance < 4 |
| **Backend equivalence** | Metal output ≈ Vulkan output for fixed scene | `verification/test_backend_equivalence.py` | perceptual-hash distance < 4 |

The CI matrix runs all of these on macOS and Windows runners. **No PR merges with a failing physics test.** Once a physics bug is in the codebase, the visual output may still look "plausible" and the bug becomes invisible — so the test gate is rigid by design.

## 12. Glossary

- **Affine parameter (λ):** parameter along a geodesic for which the geodesic equation has no first-derivative term. For null geodesics, proper time isn't defined, so λ takes its place.
- **Black hole shadow:** the region of the observer's sky from which no light escapes to infinity.
- **Christoffel symbol:** `Γ^μ_νσ`, encodes how basis vectors rotate under parallel transport. Not a tensor.
- **Event horizon:** surface from which no future-directed timelike or null curve escapes to infinity.
- **Frame dragging:** rotating spacetime forces inertial frames to co-rotate with the source mass.
- **Geodesic:** curve along which the tangent vector is parallel-transported along itself; curved-space generalization of "straight line."
- **ISCO:** Innermost Stable Circular Orbit. Inner edge of an accretion disc.
- **Null geodesic:** geodesic for which `g_μν u^μ u^ν = 0`. Path of a massless particle.
- **Photon sphere:** `r = 1.5 r_s` for Schwarzschild. Locus of unstable circular photon orbits.
- **Schwarzschild radius:** `r_s = 2GM/c² = 2M` (geometrized). Radius of the event horizon for a non-rotating BH.
- **Tetrad / orthonormal frame:** four orthonormal basis vectors at a spacetime point, used to convert between coordinate-basis and locally-Minkowskian quantities. Essential for computing what an observer "sees."
