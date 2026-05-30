"""Tests for core.forge.multiply — TDD, written before implementation."""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

# Skip entire module if ffmpeg is not available
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg / ffprobe not found", allow_module_level=True)


def _make_image_master(tmp_path: Path) -> Path:
    out = tmp_path / "master.png"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=540x960:duration=1",
            "-frames:v", "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _make_video_master(tmp_path: Path) -> Path:
    out = tmp_path / "master.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=540x960:rate=24:duration=2",
            "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _get_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of a media file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Image tests
# ---------------------------------------------------------------------------

class TestImageMultiply:
    def test_returns_correct_count(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 6, out_dir)

        assert len(results) == 6

    def test_all_files_exist(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 6, out_dir)

        for p in results:
            assert Path(p).exists(), f"{p} does not exist"

    def test_all_extensions_match_master(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 6, out_dir)

        for p in results:
            assert Path(p).suffix.lower() == ".png"

    def test_dimensions_preserved(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 6, out_dir)

        for p in results:
            w, h = _get_dimensions(Path(p))
            assert (w, h) == (540, 960), f"{p}: expected 540x960, got {w}x{h}"

    def test_variants_are_perceptually_distinct(self, tmp_path):
        from core.forge.multiply import multiply, _dhash

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 6, out_dir)
        hashes = [_dhash(Path(p)) for p in results]

        # Every pair must differ by >= 5 bits (slightly looser than internal threshold of 6)
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                dist = _hamming(hashes[i], hashes[j])
                assert dist >= 5, (
                    f"Variants {i} and {j} are too similar: Hamming distance = {dist}"
                )

    def test_base_name_applied(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "variants"

        results = multiply(master, 3, out_dir, base_name="clip")

        for p in results:
            assert Path(p).name.startswith("clip_")

    def test_outdir_created(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_image_master(tmp_path)
        out_dir = tmp_path / "deep" / "nested" / "variants"

        multiply(master, 2, out_dir)

        assert out_dir.exists()


# ---------------------------------------------------------------------------
# Video tests
# ---------------------------------------------------------------------------

class TestVideoMultiply:
    @pytest.mark.timeout(120)
    def test_returns_correct_count_and_files_exist(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_video_master(tmp_path)
        out_dir = tmp_path / "video_variants"

        results = multiply(master, 4, out_dir)

        assert len(results) == 4
        for p in results:
            assert Path(p).exists(), f"{p} does not exist"
            assert Path(p).suffix.lower() == ".mp4"
            assert Path(p).stat().st_size > 0, f"{p} has zero size"

    @pytest.mark.timeout(120)
    def test_video_dimensions_preserved(self, tmp_path):
        from core.forge.multiply import multiply

        master = _make_video_master(tmp_path)
        out_dir = tmp_path / "video_variants"

        results = multiply(master, 2, out_dir)

        for p in results:
            w, h = _get_dimensions(Path(p))
            assert (w, h) == (540, 960), f"{p}: expected 540x960, got {w}x{h}"
