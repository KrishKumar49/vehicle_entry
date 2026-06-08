import os
import sys
import logging
import importlib
import time

import cv2
import torch
import numpy as np

# Forces Python to discover the cloned fast-reid folder layout automatically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTREID_PATH = os.path.join(CURRENT_DIR, "fast-reid")
if FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)

import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

BASE_DIR = os.path.dirname(__file__)
from download_models import ensure_vehicle_model_assets

ensure_vehicle_model_assets()

from ultralytics import YOLO
import torchvision.transforms as T

from database import get_active_visit_id, save_vehicle_entry_record

try:
    fastreid_config = importlib.import_module("fastreid.config")
    fastreid_engine = importlib.import_module("fastreid.engine")
    get_cfg = fastreid_config.get_cfg
    DefaultPredictor = fastreid_engine.DefaultPredictor
except Exception as exc:
    get_cfg = None
    DefaultPredictor = None
    print(f"Warning: FastReID is unavailable ({exc}); vehicle embeddings will be skipped")

logger = logging.getLogger(__name__)

DETECTION_MODEL_PATH = os.getenv("VEHICLE_DETECTION_MODEL", os.path.join(BASE_DIR, "yolov8n.pt"))
REID_WEIGHTS_PATH = os.getenv("VEHICLE_REID_WEIGHTS", os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth"))
REID_CONFIG_PATH = os.getenv(
    "VEHICLE_REID_CONFIG",
    os.path.join(FASTREID_PATH, "configs", "VehicleID", "bagtricks_R50-ibn.yml"),
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

if not os.path.isfile(DETECTION_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle detection model not found: {DETECTION_MODEL_PATH}")

detector = YOLO(DETECTION_MODEL_PATH)

cfg = None
predictor = None

if get_cfg is not None and DefaultPredictor is not None:
    cfg = get_cfg()
    if os.path.isfile(REID_CONFIG_PATH):
        cfg.merge_from_file(REID_CONFIG_PATH)
        cfg.MODEL.WEIGHTS = REID_WEIGHTS_PATH
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.isfile(cfg.MODEL.WEIGHTS):
            predictor = DefaultPredictor(cfg)
        else:
            logger.warning("FastReID checkpoint not found at %s; vehicle embeddings will be skipped", cfg.MODEL.WEIGHTS)
    else:
        logger.warning("FastReID config not found at %s; vehicle embeddings will be skipped", REID_CONFIG_PATH)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def get_vehicle_embedding(crop):
    if predictor is None or crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb)
    tensor = tensor.unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        embedding = predictor.model({"images": tensor})

    embedding = embedding.cpu().numpy().flatten()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None

    return embedding / norm


def _resolve_visit_id(employee_id, visit_id):
    if visit_id:
        return visit_id
    if employee_id:
        return get_active_visit_id(employee_id)
    return None


def start_live_vehicle_entry(
    source=0,
    # frame_skip=5,
    max_frames=None,
    show_window=False,
    persist=False,
    employee_id=None,
    visit_id=None,
    camera_id=None,
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open video source %s", source)
        return

    logger.info("Processing source with ByteTrack: %s", source)
    resolved_visit_id = _resolve_visit_id(employee_id, visit_id)
    if persist and not resolved_visit_id:
        logger.warning("Persistence is enabled but no visit_id could be resolved.")

    frame_count = 0
    processed = 0
    saved_detections = 0
    
    tracked_vehicles = {}
    VALID_CLASSES = {3}
    MIN_EMBEDDINGS_TO_SAVE = 15
    MAX_EMBEDDINGS = 30
    STALE_TRACK_FRAMES = 90

    def save_track_if_ready(track_id):
        nonlocal saved_detections

        track_data = tracked_vehicles.get(track_id)
        if not track_data or track_data["saved"]:
            return

        if len(track_data["embeddings"]) < MIN_EMBEDDINGS_TO_SAVE:
            return

        mean_embedding = np.mean(track_data["embeddings"], axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm == 0:
            logger.warning("Skipping track %s because mean embedding norm is zero", track_id)
            return

        mean_embedding = mean_embedding / norm

        try:
            entry_id = save_vehicle_entry_record(
                employee_id=employee_id,
                visit_id=resolved_visit_id,
                vehicle_embedding=mean_embedding,
                vehicle_class=track_data["vehicle_class"],
                plate_number=None,
                camera_id=camera_id,
            )
            track_data["saved"] = True
            saved_detections += 1
            logger.info("Saved track %s as entry record id=%s", track_id, entry_id)
        except Exception as exc:
            logger.exception("Failed to save vehicle entry record for track %s: %s", track_id, exc)

    while True:
        ret, frame = cap.read()
        
        # --- ADD THIS: Reconnection Logic ---
        if not ret:
            logger.warning("Stream disconnected! Attempting to reconnect to %s...", source)
            time.sleep(5)  # Wait 5 seconds before retrying
            cap.release()
            cap = cv2.VideoCapture(source)
            continue
        # ------------------------------------

        frame_count += 1
        # if frame_skip and (frame_count % frame_skip) != 0:
        #     continue

        processed += 1

        try:
            # 🚀 Native Ultralytics ByteTrack invocation
            results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            
            if results.boxes is not None and results.boxes.id is not None:
                box_ids = results.boxes.id.int().cpu().tolist()
                xyxy_coords = results.boxes.xyxy.int().cpu().tolist()
                cls_indices = results.boxes.cls.int().cpu().tolist()

                for idx, track_id in enumerate(box_ids):
                    cls_id = cls_indices[idx]
                    if cls_id not in VALID_CLASSES:
                        continue

                    x1, y1, x2, y2 = xyxy_coords[idx]
                    crop = frame[max(0, y1):y2, max(0, x1):x2]

                    if crop.size == 0:
                        continue

                    if track_id not in tracked_vehicles:
                        tracked_vehicles[track_id] = {
                            "embeddings": [],
                            "last_seen": frame_count,
                            "saved": False,
                            "vehicle_class": results.names[cls_id],
                        }

                    tracked_vehicles[track_id]["last_seen"] = frame_count
                    tracked_vehicles[track_id]["vehicle_class"] = results.names[cls_id]

                    if tracked_vehicles[track_id]["saved"]:
                        continue

                    if frame_count % 3 == 0:
                        embedding = get_vehicle_embedding(crop)
                    else:
                        embedding = None
                        
                    if (
                        embedding is not None
                        and len(tracked_vehicles[track_id]["embeddings"]) < MAX_EMBEDDINGS
                    ):
                        tracked_vehicles[track_id]["embeddings"].append(embedding)

                    if (
                        not tracked_vehicles[track_id]["saved"]
                        and len(tracked_vehicles[track_id]["embeddings"]) >= MIN_EMBEDDINGS_TO_SAVE
                        and persist
                        and resolved_visit_id
                        and employee_id
                    ):
                        save_track_if_ready(track_id)
        except Exception as e:
            logger.exception("Tracking evaluation step error: %s", e)

        if frame_count % 30 == 0:
            tracked_vehicles = {
                tid: data
                for tid, data in tracked_vehicles.items()
                if frame_count - data["last_seen"] <= STALE_TRACK_FRAMES
            }

        if max_frames and processed >= max_frames:
            logger.info("Reached max processed frames: %s", max_frames)
            break

    cap.release()

    logger.info("Vehicle entry run complete: processed=%s saved=%s", processed, saved_detections)