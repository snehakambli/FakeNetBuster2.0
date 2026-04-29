"""Video preview route — transcodes uploaded video to browser-safe H.264 MP4."""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import imageio_ffmpeg

router = APIRouter(prefix="/preview", tags=["preview"])
logger = logging.getLogger(__name__)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Resolve the uploads directory to an absolute path for path traversal protection
_UPLOADS_DIR = Path("uploads").resolve()


def _cleanup(path: str):
    """Delete temp file after response is sent."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


@router.get("/video")
async def preview_video(path: str):
    """
    Transcode a video file to H.264/AAC MP4 that any browser can play.
    ?path=backend/uploads/abc.mp4
    """
    # Security: resolve and verify the path is inside the uploads directory
    try:
        # Handle both absolute and relative paths
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(os.getcwd()) / candidate
        resolved = candidate.resolve()
        resolved.relative_to(_UPLOADS_DIR)  # raises ValueError if outside
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    out_path = tmp.name

    cmd = [
        FFMPEG, "-y",
        "-i", str(resolved),
        "-vcodec", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-acodec", "aac",
        "-movflags", "+faststart",
        "-preset", "ultrafast",
        "-crf", "28",
        out_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            _cleanup(out_path)
            logger.error("ffmpeg failed: %s", result.stderr.decode()[-300:])
            raise HTTPException(status_code=500, detail="Video transcoding failed")
    except subprocess.TimeoutExpired:
        _cleanup(out_path)
        raise HTTPException(status_code=504, detail="Transcoding timed out")
    except HTTPException:
        raise
    except Exception as e:
        _cleanup(out_path)
        logger.error("Transcoding error: %s", e)
        raise HTTPException(status_code=500, detail="Transcoding error")

    return FileResponse(
        out_path,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(_cleanup, out_path),
    )
