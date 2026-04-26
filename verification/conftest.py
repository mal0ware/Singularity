"""Shared pytest fixtures for the Singularity verification harness.

The central fixture is :func:`singularity_cli`, which returns the path to the
built ``singularity_cli`` binary. Tests use it via subprocess rather than
importing Python bindings — the CLI is the same artifact the CI pipeline ships,
so testing through it exercises the real code paths end to end.

Resolution order for the binary:

1. ``SINGULARITY_CLI`` environment variable, if set.
2. Common CMake build directories under the repo root
   (``build/``, ``build-release/``, ``out/build/<preset>/``).
3. ``shutil.which("singularity_cli")`` as a last resort.

Tests that require the binary should declare the fixture; if it is missing
the test is skipped with a clear message rather than failing. That way the
pure-Python symbolic tests (SymPy Christoffel re-derivation, etc.) still run
on a clean checkout before any C++ code has been built.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_BINARY_NAME = "singularity_cli.exe" if os.name == "nt" else "singularity_cli"


def _candidate_paths() -> list[Path]:
    """Plausible locations for the built CLI, in priority order."""
    env = os.environ.get("SINGULARITY_CLI")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))

    for build_dir in (
        "build",
        "build-release",
        "build-debug",
        "cmake-build-debug",
        "cmake-build-release",
    ):
        root = REPO_ROOT / build_dir
        candidates.append(root / "cli" / _BINARY_NAME)
        candidates.append(root / "cli" / "Release" / _BINARY_NAME)
        candidates.append(root / "cli" / "Debug" / _BINARY_NAME)
        candidates.append(root / _BINARY_NAME)

    vs_out = REPO_ROOT / "out" / "build"
    if vs_out.is_dir():
        for preset in vs_out.iterdir():
            candidates.append(preset / "cli" / _BINARY_NAME)

    on_path = shutil.which("singularity_cli")
    if on_path:
        candidates.append(Path(on_path))

    return candidates


_APP_BUNDLE_NAME = "singularity.app"
_APP_EXECUTABLE_NAME = "singularity.exe" if os.name == "nt" else "singularity"

_CUDA_CLI_NAME = "singularity_cuda_cli.exe" if os.name == "nt" else "singularity_cuda_cli"


def _cuda_cli_candidates() -> list[Path]:
    """Plausible locations of the offline CUDA renderer."""
    env = os.environ.get("SINGULARITY_CUDA_CLI")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    for build_dir in ("build-cuda", "build", "build-release"):
        root = REPO_ROOT / build_dir
        candidates.append(root / "cuda_cli" / _CUDA_CLI_NAME)
        candidates.append(root / "cuda_cli" / "Release" / _CUDA_CLI_NAME)
        candidates.append(root / "cuda_cli" / "Debug" / _CUDA_CLI_NAME)
    return candidates


def _app_binary_candidates() -> list[Path]:
    """Plausible locations of the Metal/Vulkan app binary (not the CLI)."""
    env = os.environ.get("SINGULARITY_APP")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    for build_dir in ("build", "build-release", "build-debug"):
        root = REPO_ROOT / build_dir
        # macOS: built as Foo.app/Contents/MacOS/Foo
        candidates.append(root / "app" / _APP_BUNDLE_NAME / "Contents" / "MacOS" / "singularity")
        # Windows / Linux: plain executable
        candidates.append(root / "app" / _APP_EXECUTABLE_NAME)
        candidates.append(root / "app" / "Release" / _APP_EXECUTABLE_NAME)
        candidates.append(root / "app" / "Debug" / _APP_EXECUTABLE_NAME)
    return candidates


@pytest.fixture(scope="session")
def singularity_app() -> Path:
    """Absolute path to the real-time ``singularity`` app binary.

    Skips the calling test if no build artifact can be located — required
    by the backend-equivalence tests that need a GPU renderer.
    """
    for path in _app_binary_candidates():
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    pytest.skip(
        "singularity app binary not found. Build with "
        "cmake -B build && cmake --build build --target singularity "
        "or set SINGULARITY_APP."
    )


@pytest.fixture(scope="session")
def singularity_cli() -> Path:
    """Absolute path to the built ``singularity_cli`` binary.

    Skips the calling test if no build artifact can be located — useful on a
    fresh checkout where only the symbolic / pure-Python tests should run.
    """
    for path in _candidate_paths():
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()

    pytest.skip(
        "singularity_cli binary not found. Build the project "
        "(cmake -B build && cmake --build build --target singularity_cli) "
        "or set the SINGULARITY_CLI environment variable."
    )


@pytest.fixture(scope="session")
def singularity_cuda_cli() -> Path:
    """Absolute path to the built ``singularity_cuda_cli`` binary.

    Skips the calling test when no CUDA build artifact is found — the CUDA
    backend is opt-in (``SINGULARITY_BACKEND_CUDA=ON``) and the rest of the
    test suite must remain runnable on hosts without CUDA Toolkit.
    """
    for path in _cuda_cli_candidates():
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    pytest.skip(
        "singularity_cuda_cli binary not found. Build with "
        "cmake -B build-cuda -DSINGULARITY_BACKEND_CUDA=ON "
        "&& cmake --build build-cuda --target singularity_cuda_cli "
        "or set SINGULARITY_CUDA_CLI."
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def golden_dir(repo_root: Path) -> Path:
    """Directory containing per-backend golden reference images."""
    return repo_root / "verification" / "golden"
