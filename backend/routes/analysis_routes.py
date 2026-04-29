"""Analysis API routes."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from backend.services.analysis_engine import analyze_file, analyze_text

router = APIRouter(prefix="/analyze", tags=["analysis"])


class AnalyzeFileRequest(BaseModel):
    file_path: str
    content_type_hint: Optional[str] = None


class AnalyzeTextRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None


@router.post("/file")
async def analyze_uploaded_file(request: AnalyzeFileRequest):
    """
    Analyze an uploaded file.
    Provide the file_path returned from /upload.
    """
    import os
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        report = await analyze_file(request.file_path, request.content_type_hint)
        return report
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)  # shows in uvicorn console
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/news")
async def analyze_news(request: AnalyzeTextRequest):
    """
    Analyze news text or URL for fake news detection.
    """
    input_data = request.url or request.text
    if not input_data:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'url'")

    try:
        report = await analyze_text(input_data)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full")
async def analyze_full(file_path: Optional[str] = None,
                       text: Optional[str] = None,
                       url: Optional[str] = None):
    """
    Unified analysis endpoint. Accepts file path, text, or URL.
    """
    if not any([file_path, text, url]):
        raise HTTPException(status_code=400, detail="Provide file_path, text, or url")

    try:
        if file_path:
            import os
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")
            return await analyze_file(file_path)
        else:
            return await analyze_text(url or text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
