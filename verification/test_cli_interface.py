"""Smoke tests for the CLI's user-facing surface — argument parsing, mode
dispatch, error reporting. These are lighter than the physics-heavy tests
elsewhere in this directory but catch regressions in the interface that
would otherwise only surface when a user tries the docs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_EXPECTED_MODES = [
    "2d-toy",
    "kerr-2d-toy",
    "kerr-geometry",
    "photon-orbit",
    "disc-preview",
    "cpu-render",
    "kerr-cpu-render",
]

_EXPECTED_FLAGS = [
    "--mode",
    "--output",
    "--spin",
    "--mass",
    "--b-range",
    "--n-rays",
    "--r-init",
    "--orbits",
    "--h-step",
    "--r-inner",
    "--r-outer",
    "--scene",
    "--resolution",
    "--camera-distance",
    "--camera-elevation",
    "--camera-fov",
]


@pytest.mark.smoke
def test_help_lists_every_mode(singularity_cli: Path) -> None:
    """--help output documents every implemented mode. A mode added without
    updating the help string would slip past this check."""
    result = subprocess.run(
        [str(singularity_cli), "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    for mode in _EXPECTED_MODES:
        assert mode in combined, f"--help missing mode '{mode}'"


@pytest.mark.smoke
def test_help_lists_every_flag(singularity_cli: Path) -> None:
    """--help output mentions every top-level flag. Keeps the CLI surface
    and its documentation honest with each other."""
    result = subprocess.run(
        [str(singularity_cli), "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    for flag in _EXPECTED_FLAGS:
        assert flag in combined, f"--help missing flag '{flag}'"


@pytest.mark.smoke
def test_unknown_mode_returns_error(singularity_cli: Path) -> None:
    """Unknown modes should exit non-zero with a clear message. A silent
    failure here would cause downstream scripts to produce empty outputs
    without flagging the misconfiguration."""
    result = subprocess.run(
        [str(singularity_cli), "--mode", "not-a-real-mode"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "not-a-real-mode" in (result.stderr + result.stdout)


@pytest.mark.smoke
def test_unknown_flag_returns_error(singularity_cli: Path) -> None:
    result = subprocess.run(
        [str(singularity_cli), "--not-a-flag", "42"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


@pytest.mark.smoke
def test_scene_file_not_found_fails_gracefully(singularity_cli: Path) -> None:
    """A missing scene file should exit with a helpful error, not crash."""
    result = subprocess.run(
        [
            str(singularity_cli),
            "--scene",
            "/does/not/exist/scene.conf",
            "--mode",
            "kerr-geometry",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "scene" in (result.stderr + result.stdout).lower()


@pytest.mark.smoke
def test_scene_override_applies(singularity_cli: Path, tmp_path: Path) -> None:
    """Scene values load through --scene and influence mode behaviour. We
    use M = 1 here so ``spin`` in the config equals a/M unambiguously."""
    scene = tmp_path / "test.conf"
    scene.write_text("mass = 1.0\n" "spin = 0.7\n" "h_step = 0.02\n")
    out = tmp_path / "geom.json"
    result = subprocess.run(
        [
            str(singularity_cli),
            "--scene",
            str(scene),
            "--mode",
            "kerr-geometry",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    import json

    data = json.loads(out.read_text())
    assert abs(data["mass"] - 1.0) < 1e-4
    assert abs(data["spin"] - 0.7) < 1e-4
    assert abs(data["spin_over_mass"] - 0.7) < 1e-4


@pytest.mark.smoke
@pytest.mark.parametrize(
    "mode,extra",
    [
        ("cpu-render", []),
        ("cpu-render", ["--supersample", "2"]),
        ("kerr-cpu-render", ["--spin", "0.9"]),
        ("kerr-cpu-render", ["--spin", "0.5", "--supersample", "2"]),
    ],
)
def test_cpu_renderers_produce_valid_png(
    singularity_cli: Path, tmp_path: Path, mode: str, extra: list[str]
) -> None:
    """End-to-end smoke: both CPU ray tracers produce valid PNG at small
    resolutions. Catches regressions in pixel-loop plumbing, threading,
    supersampling, and the PNG writer for both Schwarzschild and Kerr paths."""
    out = tmp_path / f"render_{mode}.png"
    result = subprocess.run(
        [
            str(singularity_cli),
            "--mode",
            mode,
            "--resolution",
            "32x32",
            "--output",
            str(out),
        ]
        + extra,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    with out.open("rb") as fh:
        header = fh.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"
    # Size is non-trivial — at least a few hundred bytes.
    assert out.stat().st_size > 200


@pytest.mark.smoke
def test_resolution_flag_requires_WxH(singularity_cli: Path) -> None:
    """A --resolution without an x separator should fail with a clear error."""
    result = subprocess.run(
        [str(singularity_cli), "--mode", "cpu-render", "--resolution", "256"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0


@pytest.mark.smoke
def test_command_line_overrides_scene_file(singularity_cli: Path, tmp_path: Path) -> None:
    """A flag passed after --scene wins against the scene's value — the
    standard config-then-overrides ergonomic we document in the CLI help."""
    scene = tmp_path / "test.conf"
    scene.write_text("mass = 1.0\nspin = 0.5\n")
    out = tmp_path / "geom.json"
    result = subprocess.run(
        [
            str(singularity_cli),
            "--scene",
            str(scene),
            "--spin",
            "0.2",  # override scene
            "--mode",
            "kerr-geometry",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    import json

    data = json.loads(out.read_text())
    assert abs(data["mass"] - 1.0) < 1e-4
    assert abs(data["spin"] - 0.2) < 1e-4
    assert abs(data["spin_over_mass"] - 0.2) < 1e-4
