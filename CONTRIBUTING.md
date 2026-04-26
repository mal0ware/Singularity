# Contributing to Singularity

Thanks for your interest. This project is primarily a solo portfolio effort, but focused PRs are welcome — especially for Phase 9 stretch items (Linux packaging, additional spacetimes, educational overlays).

## Before you start

1. Read [`docs/PRD.md`](docs/PRD.md) so you understand what's in and out of scope.
2. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) so you know where things live.
3. If you're touching physics, read [`docs/PHYSICS.md`](docs/PHYSICS.md) — every physics change must be accompanied by a verification test at the tolerance stated in §11.
4. Check [`docs/TODO.md`](docs/TODO.md). The phase discipline matters: we don't accept, e.g., Kerr work before Schwarzschild ships (Phase 3), or abstraction refactors before Phase 4.

## Dev loop

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build

# C++ unit tests
ctest --test-dir build --output-on-failure

# Python physics verification
pip install -r verification/requirements.txt
pytest verification/
```

See [`BUILDING.md`](BUILDING.md) for platform-specific setup.

## Code style

- **C++20.** Enforced by compiler flags in [`CMakeLists.txt`](CMakeLists.txt).
- **`clang-format`** using the rules in [`.clang-format`](.clang-format). Run `clang-format -i` on any file you change. CI rejects unformatted code.
- **`clang-tidy`** using the rules in [`.clang-tidy`](.clang-tidy). New warnings should be fixed, not suppressed.
- **Python** (for `verification/` only): `ruff` + `black`, both enforced in CI.

## Commit messages

Short subject line, imperative mood, prefix with a tag where it helps reviewers:

- `[physics]` — changes to anything under `core/include/physics/` or `shared_shader/geodesic_math.h`
- `[metal]`, `[vulkan]`, `[cuda]` — backend-specific
- `[refactor]` — mechanical restructuring with no behavior change
- `[golden]` — deliberately regenerates reference images (always review the PNG diff by eye before committing these)
- `[ci]`, `[build]`, `[docs]` — self-explanatory

Body paragraph should explain *why* if it's non-obvious. Tag completed TODO items with the commit SHA as described at the top of `docs/TODO.md`.

## Test requirements for PRs

| Kind of change | Required tests |
|---|---|
| New physics (metric, integrator, conserved quantity) | SymPy symbolic derivation + analytical tolerance test under `verification/` |
| Backend implementation change | Pass existing golden-image tests for that backend, plus the backend-equivalence test if a second backend exists |
| Visual change (tone map, UI, overlays) | Regenerate affected golden images deliberately, commit with `[golden]` prefix |
| Pure refactor | All existing tests pass unchanged |

## Don't

- Introduce a dependency not listed in [`docs/PRD.md`](docs/PRD.md) §7.1 without discussing first.
- Add a database, a Redis, a Stripe integration, or any SaaS infrastructure. See `docs/PRD.md` §2.3 and §7.2.
- Skip the verification tests "because the visual output looks fine." A wrong Christoffel sign can look plausible and still be wrong.
- `git push --force` to `main`.
- Commit secrets, signing keys, or `.p12` certificates. The `.gitignore` tries to catch common cases; double-check before pushing.

## Questions

Open an issue rather than emailing. Issues are searchable; email isn't.
