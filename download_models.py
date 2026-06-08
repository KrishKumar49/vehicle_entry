# import os
# import gdown

# BASE_DIR = os.path.dirname(__file__)

# # Replace these strings with the actual File IDs you extracted from Google Drive
# YOLO_DRIVE_ID = os.getenv("VEHICLE_DETECTION_MODEL_URL")
# REID_DRIVE_ID = os.getenv("VEHICLE_REID_FILE_ID")

# def ensure_vehicle_model_assets():
#     yolo_path = os.path.join(BASE_DIR, "yolov8n.pt")
#     reid_path = os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth")

#     # If YOLO weights don't exist locally, download them automatically
#     if not os.path.exists(yolo_path):
#         print("Downloading YOLOv8 weights from Google Drive...")
#         url = YOLO_DRIVE_ID if YOLO_DRIVE_ID else f"https://drive.google.com/uc?id={YOLO_DRIVE_ID}"
#         gdown.download(url, yolo_path, quiet=False)

#     # If FastReID weights don't exist locally, download them automatically
#     if not os.path.exists(reid_path):
#         print("Downloading FastReID weights from Google Drive...")
#         url = f"https://drive.google.com/uc?id={REID_DRIVE_ID}"
#         gdown.download(url, reid_path, quiet=False)



import os
import re
import gdown
from dotenv import load_dotenv

# Ensure environment variables are loaded from your .env file
load_dotenv()

BASE_DIR = os.path.dirname(__file__)

# Pull the full URLs from your .env file
YOLO_URL = os.getenv("VEHICLE_DETECTION_MODEL_URL")
REID_URL = os.getenv("VEHICLE_REID_MODEL_URL")

def extract_gdrive_id(url: str) -> str:
    """Helper function to cleanly slice out the File ID from a full Google Drive sharing URL."""
    if not url:
        return None
    # Regular expression searches for the string of characters between /d/ and /view
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else None

def ensure_vehicle_model_assets():
    yolo_path = os.path.join(BASE_DIR, "yolov8n.pt")
    reid_path = os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth")

    # Extract clean IDs from full URLs
    yolo_id = extract_gdrive_id(YOLO_URL)
    reid_id = extract_gdrive_id(REID_URL)

    # Verification fallback step
    if not yolo_id or not reid_id:
        print("⚠️ Warning: Could not parse Google Drive File IDs from your .env URLs.")
        print(f"Parsed YOLO ID: {yolo_id} | Parsed REID ID: {reid_id}")
        return

    # 1. Download YOLO weights if missing locally
    if not os.path.exists(yolo_path):
        print("Downloading YOLOv8 weights from Google Drive...")
        download_url = f"https://drive.google.com/uc?id={yolo_id}"
        gdown.download(download_url, yolo_path, quiet=False)

    # 2. Download FastReID weights if missing locally
    if not os.path.exists(reid_path):
        print("Downloading FastReID weights from Google Drive...")
        download_url = f"https://drive.google.com/uc?id={reid_id}"
        gdown.download(download_url, reid_path, quiet=False)