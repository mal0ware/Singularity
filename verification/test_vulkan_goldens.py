"""Vulkan-backend regression test against the committed goldens.

Renders the three scenes in ``verification/golden/vulkan/`` with the
live app and pHash-compares each against the baseline PNG. Tight
tolerance (distance ≤ 4) — same backend on the same hardware should
hash-match within pHash's 64-bit resolution.

**Opt-in**: set ``SINGULARITY_VULKAN_GOLDEN_CHECK=1`` before pytest.
Skipped otherwise because the macOS CI leg uses Metal (perceptually
different output → would false-positive) and the Linux/Windows
build-test legs run headless without a GPU driver. Local Windows +
NVIDIA developer workflow is the intended venue.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

if not os.environ.get("SINGULARITY_VULKAN_GOLDEN_CHECK"):
    pytest.skip(
        "set SINGULARITY_VULKAN_GOLDEN_CHECK=1 to run Vulkan golden diffs",
        allow_module_level=True,
    )

imagehash = pytest.importorskip("imagehash")
PIL = pytest.importorskip("PIL.Image")

GOLDEN_DIR = Path(__file__).parent / "golden" / "vulkan"
PHASH_TOLERANCE = 4  # same backend on same hardware should match tightly.

SCENES = [
    ("schw.png", []),
    ("kerr_a05.png", ["--kerr", "--spin=0.5"]),
    ("kerr_a094.png", ["--kerr", "--spin=0.94"]),
]


@pytest.mark.slow
@pytest.mark.parametrize("golden_name,extra_args", SCENES)
def test_vulkan_matches_golden(
    tmp_path: Path,
    singularity_app: Path,
    golden_name: str,
    extra_args: list[str],
) -> None:
    golden = GOLDEN_DIR / golden_name
    assert golden.is_file(), f"missing golden: {golden}"

    rendered = tmp_path / f"rendered_{golden_name}"
    cmd = [
        str(singularity_app),
        f"--capture={rendered}",
        "--capture-ss=1",
        "--res=256x256",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert (
        result.returncode == 0
    ), f"render failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert rendered.is_file(), "expected capture PNG to exist"

    hash_golden = imagehash.phash(PIL.open(golden).convert("RGB"), hash_size=8)
    hash_rendered = imagehash.phash(PIL.open(rendered).convert("RGB"), hash_size=8)
    distance = hash_golden - hash_rendered
    assert distance <= PHASH_TOLERANCE, (
        f"{golden_name}: pHash distance {distance} > tolerance {PHASH_TOLERANCE}\n"
        f"  golden   = {hash_golden}\n"
        f"  rendered = {hash_rendered}"
    )
