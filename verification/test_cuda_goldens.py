"""CUDA-backend regression test against the committed CUDA goldens.

Renders the three scenes in ``verification/golden/cuda/`` with
``singularity_cuda_cli`` and pHash-compares each against the baseline
PNG. Tight tolerance (distance ≤ 4) — same backend on the same hardware
should hash-match within pHash's 64-bit resolution.

This is the CUDA twin of ``test_vulkan_goldens.py``: a pure regression
gate that catches kernel / tone-map / camera-default changes that
weren't matched by a refresh of the goldens. The cross-backend
"CUDA structurally agrees with Vulkan" check lives in
``test_backend_equivalence.py``.

Always-on: unlike the Vulkan golden test (which gates on
``SINGULARITY_VULKAN_GOLDEN_CHECK`` because most CI legs lack a GPU
driver), the CUDA cli runs headless against the device — so the test
just skips cleanly when no CUDA cli build is present, and the
``singularity_cuda_cli`` fixture handles that.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

imagehash = pytest.importorskip("imagehash")
PIL = pytest.importorskip("PIL.Image")

GOLDEN_DIR = Path(__file__).parent / "golden" / "cuda"
PHASH_TOLERANCE = 4  # same backend, same hardware → tight match expected.

# (golden filename, scene file relative to repo root, extra cli args).
# scene=None means render the default Schwarzschild scene with no --scene.
SCENES = [
    ("schw.png", None),
    ("kerr_a05.png", "scenes/kerr_a05.scene"),
    ("kerr_a094.png", "scenes/kerr_a094.scene"),
]


@pytest.mark.slow
@pytest.mark.parametrize("golden_name,scene_rel", SCENES)
def test_cuda_matches_golden(
    tmp_path: Path,
    repo_root: Path,
    singularity_cuda_cli: Path,
    golden_name: str,
    scene_rel: str | None,
) -> None:
    golden = GOLDEN_DIR / golden_name
    assert golden.is_file(), f"missing golden: {golden}"

    rendered = tmp_path / f"rendered_{golden_name}"
    cmd = [
        str(singularity_cuda_cli),
        "--output",
        str(rendered),
        "--res",
        "256x256",
    ]
    if scene_rel is not None:
        scene_path = repo_root / scene_rel
        assert scene_path.is_file(), f"missing scene file: {scene_path}"
        cmd += ["--scene", str(scene_path)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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


@pytest.mark.smoke
def test_cuda_multi_frame_orbit(
    tmp_path: Path,
    repo_root: Path,
    singularity_cuda_cli: Path,
) -> None:
    """`--frames 4 --output-pattern ...` writes 4 PNGs with the camera
    advancing 90° in azimuth between frames. Smoke-level guard against
    regressions in the multi-frame video path; verifies (a) all four
    files land, (b) camera actually moved (frame_00 and frame_02 must
    differ — they're 180° apart, so the Doppler-bright side flips).
    """
    # Schwarzschild is rotationally symmetric, so 0° and 180° camera
    # azimuth produce hash-identical frames. Use Kerr so the
    # Doppler-bright wing breaks the symmetry and lets us prove the
    # orbit actually advanced between frames.
    pattern = str(tmp_path / "frame_%02d.png")
    cmd = [
        str(singularity_cuda_cli),
        "--frames",
        "4",
        "--output-pattern",
        pattern,
        "--res",
        "64x64",
        "--samples-per-pixel",
        "1",
        "--scene",
        str(repo_root / "scenes" / "kerr_a094.scene"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert (
        result.returncode == 0
    ), f"render failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"

    paths = sorted(tmp_path.glob("frame_*.png"))
    assert len(paths) == 4, f"expected 4 frames, got {len(paths)}: {paths}"

    # 0° vs 180°: same image up to flip would still produce non-zero pHash
    # distance under the hash's left-right asymmetry to the disc gradient.
    h0 = imagehash.phash(PIL.open(paths[0]).convert("RGB"), hash_size=8)
    h2 = imagehash.phash(PIL.open(paths[2]).convert("RGB"), hash_size=8)
    assert h0 - h2 > 0, (
        "frame_00 and frame_02 (180° apart in azimuth) hash-matched — "
        "camera orbit isn't actually moving"
    )
