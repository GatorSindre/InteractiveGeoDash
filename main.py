import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import time
import threading
import tkinter as tk
from mediapipe.python.solutions.face_mesh import FaceMesh  # type: ignore

# --- Shared mode toggles (thread-safe via GIL for simple bool reads/writes) ---
modes = {
    "slam_press":  True,
    "blink_press": True,
    "mouth_press": True, 
}

# --- Thresholds ---
BLINK_THRESH  = 0.18
BLINK_HOLD_SEC = 0.25   # how long to hold space after a blink
MAR_THRESH    = 5       # pixels
COOLDOWN_SEC  = 0.1     # seconds between triggers for mouth/blink

# --- Landmark index groups ---
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
 
# Signal to stop the CV thread cleanly
stop_event = threading.Event()

def on_space():
    if modes["slam_press"]:
        pyautogui.press('w')
        print("KeySpace Pressed (from keyboard event)")
    
# ── Tkinter U I ────────────────────────────────────────────────────────────────
   
class ControlPanel(tk.Tk):
    LABELS = {
        "slam_press":  "Slam Mode",
        "blink_press": "Blink Mode",
        "mouth_press": "Mouth Mode",
    }
    ON_COLOR  = "#4CAF50"   # green
    OFF_COLOR = "#9E9E9E"   # grey

    def __init__(self):
        super().__init__()
        self.title("Face Control")
        self.resizable(False, False)
        self.configure(bg="#1E1E2E", padx=20, pady=20)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(
            self,
            text="Input Mode",
            bg="#1E1E2E", fg="#CDD6F4",
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(0, 14))

        self._buttons: dict[str, tk.Button] = {}
        for key, label in self.LABELS.items():
            btn = tk.Button(
                self,
                text=label,
                width=26,
                font=("Segoe UI", 11),
                fg="white",
                activeforeground="white",
                relief="flat",
                bd=0,
                padx=10,
                pady=10,
                cursor="hand2",
                command=lambda k=key: self._toggle(k),
            )
            btn.pack(pady=5, fill="x")
            self._buttons[key] = btn
            self._refresh(key)   # set initial colour

    def _toggle(self, key: str):
        modes[key] = not modes[key]
        self._refresh(key)

    def _refresh(self, key: str):
        active = modes[key]
        btn = self._buttons[key]
        color = self.ON_COLOR if active else self.OFF_COLOR
        btn.configure(
            bg=color,
            activebackground=color,
        )

    def _on_close(self):
        stop_event.set()
        self.destroy()


# ── Helper functions ──────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_points, img_w, img_h):
    points = np.array([
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in eye_points
    ])
    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h  = np.linalg.norm(points[0] - points[3])
    return (v1 + v2) / (2.0 * h) if h else 0.0


def mouth_open_distance(landmarks, top_idx, bot_idx, img_w, img_h):
    top = landmarks[top_idx]
    bot = landmarks[bot_idx]
    return abs(int(top.y * img_h) - int(bot.y * img_h))


# ── CV / detection loop (runs in background thread) ──────────────────────────

def cv_loop():
    face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=True)

    # Open camera
    cap = None
    for idx in range(3):
        candidate = cv2.VideoCapture(idx)
        if candidate.isOpened():
            cap = candidate
            print(f"Opened camera index {idx}")
            break
        candidate.release()

    if cap is None:
        print("ERROR: Could not open any camera.")
        stop_event.set()
        return

    # State
    mouth_was_open    = False
    eyes_were_closed  = False   # tracks previous frame eye state for blink detection
    space_was_pressed = False   # tracks previous frame space state for slam detection
    last_blink_time   = 0.0
    last_mouth_time   = 0.0
    last_slam_time    = 0.0

    print("CV loop running — press ESC in the camera window to quit.")

    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            continue

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        now = time.time()

        # ── Slam mode ────────────────────────────────────────────────────────
        if modes["slam_press"]:
            space_pressed = keyboard.is_pressed('space')
            if space_pressed:
                print(f"Slam! Holding w for {BLINK_HOLD_SEC} s")
  
                def _hold_w():
                    pyautogui.keyDown('w')
                    time.sleep(0.15)
                    pyautogui.keyUp('w')

                threading.Thread(target=_hold_w, daemon=True).start()

        # ── Face-based modes ─────────────────────────────────────────────────
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # Blink — fires on open → closed transition, holds space for BLINK_HOLD_SEC
            # Each blink spawns its own independent thread; cooldown = BLINK_HOLD_SEC - 0.2
            # so rapid blinking works without blocking detection
            if modes["blink_press"]:
                left_EAR  = eye_aspect_ratio(lm, LEFT_EYE,  img_w, img_h)
                right_EAR = eye_aspect_ratio(lm, RIGHT_EYE, img_w, img_h)
                eyes_closed = (left_EAR + right_EAR) / 2.0 < BLINK_THRESH

                if not eyes_were_closed and eyes_closed and (now - last_blink_time) > (BLINK_HOLD_SEC - 0.2):
                    last_blink_time = now
                    print(f"Blink detected! Holding space for {BLINK_HOLD_SEC} s")

                    def _hold_space():
                        pyautogui.keyDown('w')
                        time.sleep(BLINK_HOLD_SEC)
                        pyautogui.keyUp('w')

                    threading.Thread(target=_hold_space, daemon=True).start()

                eyes_were_closed = eyes_closed

            # Mouth open — fires on closed → open transition, holds space for BLINK_HOLD_SEC
            # Each trigger spawns its own independent thread; cooldown = BLINK_HOLD_SEC - 0.2
            if modes["mouth_press"]:
                mar = mouth_open_distance(lm, 13, 14, img_w, img_h)
                mouth_open = mar > MAR_THRESH

                if not mouth_was_open and mouth_open and (now - last_mouth_time) > (BLINK_HOLD_SEC - 0.2):
                    last_mouth_time = now
                    print(f"Mouth open → holding space for {BLINK_HOLD_SEC} s (MAR={mar}px)")

                    def _hold_space_mouth():
                        pyautogui.keyDown('w')
                        time.sleep(BLINK_HOLD_SEC)
                        pyautogui.keyUp('w')

                    threading.Thread(target=_hold_space_mouth, daemon=True).start()

                mouth_was_open = mouth_open
        else:
            eyes_were_closed = False
            mouth_was_open   = False

        # ── Display ──────────────────────────────────────────────────────────
        cv2.imshow("Face Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("CV loop stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start CV loop in a daemon thread so it dies if the UI is closed
    cv_thread = threading.Thread(target=cv_loop, daemon=True)
    cv_thread.start()

    keyboard.on_press_key("space", lambda e: on_space(), suppress=True)
    # tkinter must run on the main thread
    app = ControlPanel()
    app.mainloop()

    # Ensure the CV thread stops after the window closes
    stop_event.set()
    cv_thread.join(timeout=3)
    print("Exited cleanly.")