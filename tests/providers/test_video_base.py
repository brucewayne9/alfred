import pytest
from pathlib import Path
from scripts.providers.video.base import VideoProvider, VideoRequest, VideoResult
from scripts.providers.video.higgsfield import HiggsfieldVideo

class FakeVideo(VideoProvider):
    name = "fake"
    def gen(self, req: VideoRequest) -> VideoResult:
        return VideoResult(video_path=Path("/tmp/fake.mp4"), duration_s=req.duration_s)

def test_video_provider_contract():
    p = FakeVideo()
    res = p.gen(VideoRequest(prompt="test", duration_s=5.0))
    assert res.video_path == Path("/tmp/fake.mp4")
    assert res.duration_s == 5.0

def test_higgsfield_video_stub_raises():
    with pytest.raises(NotImplementedError):
        HiggsfieldVideo().gen(VideoRequest(prompt="x", duration_s=3))
