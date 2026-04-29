"""Upload API routes."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.file_handler import save_upload_file
from inference.content_router import detect_content_type

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for analysis.
    Returns file metadata and detected content type.
    """
    try:
        file_meta = await save_upload_file(file)
        content_type = detect_content_type(file_meta["file_path"])
        return {
            "file_id": file_meta["file_id"],
            "filename": file_meta["filename"],
            "file_path": file_meta["file_path"],
            "content_type_detected": content_type,
            "file_size_mb": file_meta["file_size_mb"],
            "message": "File uploaded successfully. Use /analyze to process."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
