# Live face identification fro a webcam.
# Run: python identify_camera.py Press q to quit

from pathlib import Path

import cv2
import face_recognition
from ultralytics import YOLO

#settings
FACE_MODEL = "yolovn11n-face.pt"
KNOWN_DIR = Path("known_faces")
TOLERANCE = 0.5
CAMERA = 0       #0= first webcam. Try 1 if it opens the wrong one
EVERY_N_FRAMES = 5     # identify every 5th frame to keep the video smooth

#learn the known people (same as identify.py)
known_names, known_encodings = [],[]
for path in sorted(KNOWN_DIR.iterdir()):
    if path.suffix.lower() in {".jpg",".jpeg",".png"}:
        found = face_recognition.face_encodings()

