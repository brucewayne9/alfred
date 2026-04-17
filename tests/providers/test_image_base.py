import pytest
from pathlib import Path
from scripts.providers.image.base import ImageProvider, ImageRequest, ImageResult
from scripts.providers.image.higgsfield import HiggsfieldImage


class FakeImage(ImageProvider):
    name = "fake"

    def gen(self, req: ImageRequest) -> ImageResult:
        return ImageResult(image_path=Path("/tmp/fake.png"), width=req.width, height=req.height)


def test_image_provider_contract():
    p = FakeImage()
    res = p.gen(ImageRequest(prompt="a test", width=1024, height=1024))
    assert res.image_path == Path("/tmp/fake.png")
    assert res.width == 1024


def test_higgsfield_stub_raises():
    p = HiggsfieldImage()
    with pytest.raises(NotImplementedError):
        p.gen(ImageRequest(prompt="x"))
