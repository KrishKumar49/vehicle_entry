import os
import sys
import collections
import collections.abc

# 🛠️ Fix Python 3.12 compatibility bug for FastReID
# This maps the expected classes from collections.abc directly back onto collections
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

# Forces Python to discover the cloned fast-reid folder layout automatically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTREID_PATH = os.path.join(CURRENT_DIR, "fast-reid")
if FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)

# Forces Python to discover the cloned fast-reid folder layout automatically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTREID_PATH = os.path.join(CURRENT_DIR, "fast-reid")
if FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)
    
    
from fastapi import FastAPI, Response, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, Dict
import uvicorn

# Module processing engine dependencies
from vehicle_entry import start_live_vehicle_entry
from database import get_active_visit_id, create_active_visit_session

app = FastAPI(title="Dedicated Vehicle Embedding Microservice", version="1.0.0")

# Prevents multiple execution pipelines from cross-contaminating the same camera source feed
ACTIVE_STREAMS: Dict[str, str] = {}

class VehicleEntryRequest(BaseModel):
    source: Union[int, str]        # Camera index (0) or live RTSP streaming network connection URL
    employeeId: str                
    cameraId: Optional[str] = "GATE_ENTRY_MAIN"


@app.get("/streams/status")
def active_streams_manifest():
    return {"running_streams": ACTIVE_STREAMS}


@app.post("/vehicle-entry")
def process_vehicle_entry(data: VehicleEntryRequest, background_tasks: BackgroundTasks):
    # Lock down the stream if it's already active
    if data.cameraId in ACTIVE_STREAMS:
        return {
            "status": "active_processing",
            "message": f"Continuous stream channel '{data.cameraId}' is already active and writing to Neon."
        }

    try:
        # Resolve or automatically provision an active session context
        visit_id = get_active_visit_id(data.employeeId)
        if not visit_id:
            visit_id = create_active_visit_session(data.employeeId)

        # Flag stream as active in memory
        ACTIVE_STREAMS[data.cameraId] = "RUNNING"

        # Offload computer vision processing loop to background execution pools 24/7
        background_tasks.add_task(
            start_live_vehicle_entry,
            source=data.source,
            # frame_skip=5,
            max_frames=None,  # None removes limits so the engine streams endlessly
            show_window=False,
            persist=True,
            employee_id=data.employeeId,
            visit_id=visit_id,
            camera_id=data.cameraId
        )

        return {
            "status": "pipeline_initiated",
            "resolved_visit_id": visit_id,
            "message": f"Endless vision loop spawned for connection source endpoint: {data.cameraId}"
        }

    except Exception as e:
        if data.cameraId in ACTIVE_STREAMS:
            del ACTIVE_STREAMS[data.cameraId]
        raise HTTPException(status_code=500, detail=f"Failed to bind operational framework layers: {str(e)}")



if __name__ == "__main__":
    # This allows Render to tell your app which port to use
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)