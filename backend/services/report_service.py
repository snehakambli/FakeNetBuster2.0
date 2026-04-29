"""
Report service - handles report retrieval and storage.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from reports.report_generator import get_report, list_reports


async def fetch_report(report_id: str) -> dict:
    report = get_report(report_id)
    if not report:
        return None
    return report


async def fetch_history(limit: int = 50) -> list:
    return list_reports(limit=limit)
