"""Report API routes."""

from fastapi import APIRouter, HTTPException
from backend.services.report_service import fetch_report, fetch_history

router = APIRouter(prefix="/report", tags=["reports"])


@router.get("/history")
async def get_history(limit: int = 50):
    """Get list of all analysis reports."""
    reports = await fetch_history(limit=limit)
    return {"reports": reports, "total": len(reports)}


@router.get("/{report_id}")
async def get_report(report_id: str):
    """Get a specific analysis report by ID."""
    report = await fetch_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return report


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """Delete a report by ID."""
    import os
    report_path = f"reports/generated/{report_id}.json"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    os.remove(report_path)
    return {"message": f"Report {report_id} deleted"}
