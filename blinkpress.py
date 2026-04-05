import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import pyautogui
import keyboard
import time
from collections import deque

# --- Mode toggles: set these to True to activate ---
slam_press  = True   # Press 'p' to trigger spacebar
blink_press = True   # Blink triggers a click at where mouse was 0.5s ago
mouth_press = True   # Mouth open triggers a click at where mouse was 0.5s ago

MODEL_PATH = "face_landmarker.task"  # must be in same folder as this script

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

blink_counter = 0

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESH = 0.21
MAR_THRESH   = 20   # pixels
DELAY        = 0.5  # seconds into the past to click

# --- Mouse position buffer ---
# Each entry is (timestamp, x, y)
mouse_buffer = deque()
BUFFER_DURATION = 2.0  # keep up to 2 seconds of history (more than enough)

# --- Cooldown to prevent rapid repeat triggers ---
last_trigger_time = 0.0
COOLDOWN = 0.4  # seconds between allowed triggers


def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    points = np.array([
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in eye_points
    ])
    vertical1  = np.linalg.norm(points[1] - points[5])
    vertical2  = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_open_distance(landmarks, top_idx, bot_idx, img_w, img_h):
    top = landmarks[top_idx]
    bot = landmarks[bot_idx]
    return abs(int(top.y * img_h) - int(bot.y * img_h))


def get_delayed_mouse_pos():
    """Return the mouse position from DELAY seconds ago, or None if not enough history."""
    target_time = time.time() - DELAY
    # Find the entry closest to target_time
    best = None
    for entry in mouse_buffer:
        if entry[0] <= target_time:
            best = entry
        else:
            break
    return (best[1], best[2]) if best else None


def trigger_click(reason):
    """Click at the delayed mouse position if available."""
    global last_trigger_time
    now = time.time()
    if now - last_trigger_time < COOLDOWN:
        return  # still in cooldown
    pos = get_delayed_mouse_pos()
    if pos:
        x, y = pos
        pyautogui.click(x, y)
        print(f"{reason} → clicked at ({x}, {y}) [position from {DELAY}s ago]")
        last_trigger_time = now
    else:
        print(f"{reason} → not enough mouse history yet, skipping click")


cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

print("Running — press ESC to quit.")

try:
    while True:
        now = time.time()

        # --- Record current mouse position into buffer ---
        mx, my = pyautogui.position()
        mouse_buffer.append((now, mx, my))

        # --- Trim old entries beyond BUFFER_DURATION ---
        while mouse_buffer and mouse_buffer[0][0] < now - BUFFER_DURATION:
            mouse_buffer.popleft()

        success, frame = cap.read()
        if not success:
            print("WARNING: Failed to read frame, retrying...")
            continue

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = face_landmarker.detect(mp_image)

        # ── slam mode: 'p' key → spacebar ────────────────────────────────────
        if slam_press:
            if keyboard.is_pressed('p'):
                pyautogui.press('space')
                print("KeySpace Pressed")

        # ── face-based modes (need a detected face) ───────────────────────────
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            if blink_press:
                left_EAR  = eye_aspect_ratio(landmarks, LEFT_EYE,  img_w, img_h)
                right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE, img_w, img_h)
                avg_EAR   = (left_EAR + right_EAR) / 2.0
                if avg_EAR < BLINK_THRESH:
                    blink_counter += 1
                    print(f"Blink detected! Counter: {blink_counter}")
                    trigger_click("Blink")

            if mouth_press:
                mar = mouth_open_distance(landmarks, 13, 14, img_w, img_h)
                if mar > MAR_THRESH:
                    trigger_click("Mouth open")

        else:
            if blink_press or mouth_press:
                print("No face detected")  # remove if too noisy

        cv2.imshow("Face Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()