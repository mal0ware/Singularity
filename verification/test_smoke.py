"""Liveness checks. Run with ``pytest -k smoke`` — no physics, just a
sanity check that the build pipeline produced a usable CLI.

Deliberately tiny: these tests should complete in under a second on CI and
give us an unambiguous red light if the binary is missing or crashes at
startup, before any expensive physics tests bother to run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.smoke
def test_cli_help_exits_zero(singularity_cli: Path) -> None:
    """``singularity_cli --help`` should exit 0 and print something."""
    result = subprocess.run(
        [str(singularity_cli), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"singularity_cli --help exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout or result.stderr, "expected --help to produce output"


@pytest.mark.smoke
def test_python_environment() -> None:
    """Import the scientific stack we rely on. If any of these fail, the
    venv was not provisioned from verification/pyproject.toml."""
    import imagehash  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import scipy  # noqa: F401
    import sympy  # noqa: F401


@pytest.mark.smoke
@pytest.mark.parametrize(
    "metric,extra_args",
    [
        ("schw", []),
        ("kerr", ["--spin", "0.9"]),
    ],
)
def test_benchmark_emits_deterministic_json(
    singularity_cli: Path, metric: str, extra_args: list[str]
) -> None:
    """``--mode benchmark`` prints one JSON line; total_steps is deterministic
    across runs at the same workload, guarding against accidental changes to
    the integrator step budget or the ray-fan initial conditions. Exercises
    both the Schwarzschild RK4 and Kerr Hamiltonian integrator paths."""
    args = [
        str(singularity_cli),
        "--mode",
        "benchmark",
        "--metric",
        metric,
        "--n-rays",
        "8",
        "--max-steps",
        "500",
        *extra_args,
    ]
    first = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout.strip())
    assert payload["mode"] == "benchmark"
    assert payload["metric"] == metric
    assert payload["n_rays"] == 8
    assert payload["max_steps"] == 500
    assert payload["total_steps"] > 0
    assert payload["total_ms"] >= 0.0

    second = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert second.returncode == 0, second.stderr
    payload2 = json.loads(second.stdout.strip())
    assert payload2["total_steps"] == payload["total_steps"], (
        f"{metric} integrator step count drifted between runs — "
        "ray-fan init or integrator step is nondeterministic"
    )
