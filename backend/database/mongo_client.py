"""
MongoDB client for FakeNetBuster.
Stores analysis reports and upload metadata.
"""

import os
import yaml
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING


_client = None
_db = None


def get_config():
    try:
        with open("configs/system_configs.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg["database"]
    except Exception:
        return {
            "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            "name": "fakenetbuster",
            "collections": {"reports": "analysis_reports", "uploads": "uploaded_files"}
        }


async def get_database():
    global _client, _db
    if _db is None:
        cfg = get_config()
        _client = AsyncIOMotorClient(cfg["uri"])
        _db = _client[cfg["name"]]
        # Create indexes
        await _db[cfg["collections"]["reports"]].create_index(
            [("timestamp", DESCENDING)]
        )
        await _db[cfg["collections"]["reports"]].create_index("report_id", unique=True)
    return _db


async def close_database():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
