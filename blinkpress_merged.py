import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import pyautogui
import keyboard
import time
from collections import deque

# ---------------------------------------------------------------------------
# Mode toggles
# ---------------------------------------------------------------------------
gaze_move   = True   # Move mouse with eye gaze (iris landmarks)
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

# Also keep the legacy face_mesh for iris landmarks (landmarks 474-478)
# FaceLandmarker doesn't expose iris landmarks directly, so we use both.
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

blink_counter = 0

LEFT_EYE     = [33, 160, 158, 133, 153, 144]
RIGHT_EYE    = [362, 385, 387, 263, 373, 380]
BLINK_THRESH = 0.21
MAR_THRESH   = 20       # pixels
DELAY        = 0.5      # seconds into the past to click

# Gaze smoothing — increase for smoother but laggier movement
SMOOTHING = 7           # number of frames to average

screen_w, screen_h = pyautogui.size()

# ---------------------------------------------------------------------------
# Mouse position buffer (for delayed click)
# ---------------------------------------------------------------------------
mouse_buffer    = deque()
BUFFER_DURATION = 2.0   # keep up to 2 seconds of history

# Gaze position history for smoothing
gaze_history = deque(maxlen=SMOOTHING)

# Cooldown to prevent rapid repeat triggers
last_trigger_time = 0.0
COOLDOWN = 0.4          # seconds between allowed triggers


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
    """Return the mouse position from DELAY seconds ago, or None."""
    target_time = time.time() - DELAY
    best = None
    for entry in mouse_buffer:
        if entry[0] <= target_time:
            best = entry
        else:
            break
    return (best[1], best[2]) if best else None


def trigger_click(reason):
    """Click at the delayed mouse position if cooldown has passed."""
    global last_trigger_time
    now = time.time()
    if now - last_trigger_time < COOLDOWN:
        return
    pos = get_delayed_mouse_pos()
    if pos:
        x, y = pos
        pyautogui.click(x, y)
        print(f"{reason} → clicked at ({x}, {y}) [position from {DELAY}s ago]")
        last_trigger_time = now
    else:
        print(f"{reason} → not enough mouse history yet, skipping click")


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Try camera indices 0, 1, 2 automatically
cap = None
for cam_index in [0, 1, 2, 3]:
    test = cv2.VideoCapture(cam_index)
    if test.isOpened():
        ret, _ = test.read()
        if ret:
            print(f"Using camera index {cam_index}")
            cap = test
            break
        test.release()
    else:
        test.release()

if cap is None:
    print("ERROR: Could not open any camera (tried indices 0-3).")
    input("Press Enter to exit...")
    exit(1)

print("Running — press ESC to quit.")
print(f"  gaze_move   = {gaze_move}")
print(f"  blink_press = {blink_press}")
print(f"  mouth_press = {mouth_press}")
print(f"  slam_press  = {slam_press}")

try:
    while True:
        now = time.time()

        # Record current mouse position into buffer
        mx, my = pyautogui.position()
        mouse_buffer.append((now, mx, my))

        # Trim entries older than BUFFER_DURATION
        while mouse_buffer and mouse_buffer[0][0] < now - BUFFER_DURATION:
            mouse_buffer.popleft()

        success, frame = cap.read()
        if not success:
            print("WARNING: Failed to read frame, retrying...")
            continue

        frame    = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Gaze mouse movement (face_mesh iris landmarks) ──────────────────
        if gaze_move:
            mesh_result = face_mesh.process(rgb_frame)
            if mesh_result.multi_face_landmarks:
                mesh_landmarks = mesh_result.multi_face_landmarks[0].landmark

                # Iris landmarks 474-477; landmark 475 is the iris centre
                iris = mesh_landmarks[474:478]
                iris_center = iris[1]   # index 1 within the slice = landmark 475

                # Draw iris tracking dots
                for i, lm in enumerate(iris):
                    cx = int(lm.x * img_w)
                    cy = int(lm.y * img_h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                # Map iris position to screen
                raw_x = screen_w * iris_center.x
                raw_y = screen_h * iris_center.y

                gaze_history.append((raw_x, raw_y))
                if len(gaze_history) == SMOOTHING:
                    smooth_x = sum(p[0] for p in gaze_history) / SMOOTHING
                    smooth_y = sum(p[1] for p in gaze_history) / SMOOTHING
                    pyautogui.moveTo(smooth_x, smooth_y)

        # ── FaceLandmarker (blink / mouth / slam) ───────────────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result   = face_landmarker.detect(mp_image)

        # slam mode: 'p' key → spacebar
        if slam_press and keyboard.is_pressed('p'):
            pyautogui.press('space')
            print("KeySpace Pressed")

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
                pass  # uncomment below to debug
                # print("No face detected")

        # ── Display ─────────────────────────────────────────────────────────
        # Overlay active modes in corner
        modes = []
        if gaze_move:   modes.append("GAZE")
        if blink_press: modes.append("BLINK")
        if mouth_press: modes.append("MOUTH")
        if slam_press:  modes.append("SLAM[p]")
        cv2.putText(frame, " | ".join(modes), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

        cv2.imshow("Face Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    face_mesh.close()
    print("Exited cleanly.")
    input("Press Enter to close...")