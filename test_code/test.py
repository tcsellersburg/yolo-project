
import cv2
import face_recognition
import os
import numpy as np

# ==============================
# SETTINGS
# ==============================

KNOWN_FACES_DIR = "known_faces"

# Lower = stricter recognition
FACE_TOLERANCE = 0.48

# Webcam
CAMERA_INDEX = 0

# ==============================
# LOAD KNOWN FACES
# ==============================

known_encodings = []
known_names = []

print("[INFO] Loading known faces...")

for filename in os.listdir(KNOWN_FACES_DIR):

    filepath = os.path.join(KNOWN_FACES_DIR, filename)

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    image = face_recognition.load_image_file(filepath)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"[WARNING] No face found in {filename}")
        continue

    name = os.path.splitext(filename)[0]

    known_encodings.append(encodings[0])
    known_names.append(name)

    print(f"[LOADED] {name}")

print(f"[INFO] Loaded {len(known_names)} people.")

# ==============================
# START WEBCAM
# ==============================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("[ERROR] Could not access webcam.")
    exit()

print("[INFO] Face recognition started.")
print("[INFO] Press Q to quit.")

# ==============================
# MAIN LOOP
# ==============================

while True:

    success, frame = camera.read()

    if not success:
        print("[ERROR] Could not read webcam.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(
        frame, (0, 0),
        fx=0.25, fy=0.25
    )

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        small_frame, cv2.COLOR_BGR2RGB
    )

    # Detect faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations
    )

    # ==============================
    # RECOGNIZE EACH FACE
    # ==============================

    for face_encoding, face_location in zip(
        face_encodings, face_locations
    ):

        name = "UNKNOWN"
        color = (0, 0, 255)  # Red

        # Compare against known faces
        if known_encodings:

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match = np.argmin(distances)

            if distances[best_match] < FACE_TOLERANCE:

                name = known_names[best_match]
                color = (0, 255, 0)  # Green

        # Face coordinates
        top, right, bottom, left = face_location

        # Scale coordinates back to original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            3
        )

        # Draw name label
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # ==============================
    # DISPLAY STATUS
    # ==============================

    cv2.putText(
        frame,
        "FACE RECOGNITION ACTIVE",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Recognition System", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

camera.release()
cv2.destroyAllWindows()

print("[INFO] Application closed.")
