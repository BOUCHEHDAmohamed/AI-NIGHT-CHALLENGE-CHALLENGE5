"""
YOLOv8 Geofencing Person Detector
==================================
On first launch, draw a polygon zone by clicking on the frame.
Press ENTER when done. Alert (beep + red flash) triggers the moment
a person's bounding box touches or overlaps the zone.

Requirements:
    pip install ultralytics opencv-python numpy

    Audio (choose one based on your OS):
      Linux  : sudo apt-get install sox        (uses 'play' command)
      macOS  : built-in via 'afplay'
      Windows: built-in via winsound module

Usage:
    python geofence_detector.py                      # webcam 0
    python geofence_detector.py --source video.mp4   # video file
    python geofence_detector.py --source 1           # webcam 1
"""

import argparse
import sys
import os
import threading
import time
import platform
import cv2
import numpy as np
from ultralytics import YOLO

# ── Global state for mouse drawing ──────────────────────────────────────────
zone_points: list[tuple[int, int]] = []
drawing_done = False

# ── Beep state: prevent overlapping beeps ───────────────────────────────────
_beep_lock = threading.Lock()
_last_beep_time = 0.0
BEEP_COOLDOWN = 0.8  # seconds between beeps


# ────────────────────────────────────────────────────────────────────────────
#  CROSS-PLATFORM BEEP
# ────────────────────────────────────────────────────────────────────────────
def _beep_thread():
    global _last_beep_time
    with _beep_lock:
        now = time.time()
        if now - _last_beep_time < BEEP_COOLDOWN:
            return
        _last_beep_time = now

    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.Beep(1000, 300)  # 1000 Hz, 300 ms
        elif system == "Darwin":  # macOS
            os.system("afplay /System/Library/Sounds/Funk.aiff 2>/dev/null")
        else:  # Linux
            if os.system("which play >/dev/null 2>&1") == 0:
                os.system("play -nq -t alsa synth 0.3 sine 1000 2>/dev/null")
            elif os.system("which paplay >/dev/null 2>&1") == 0:
                os.system("paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null")
            elif os.system("which aplay >/dev/null 2>&1") == 0:
                import struct, math, subprocess
                rate, freq, dur = 44100, 1000, 0.3
                samples = int(rate * dur)
                data = b"".join(
                    struct.pack("<h", int(32767 * math.sin(2 * math.pi * freq * i / rate)))
                    for i in range(samples)
                )
                p = subprocess.Popen(
                    ["aplay", "-r", str(rate), "-f", "S16_LE", "-c", "1", "-t", "raw"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                p.communicate(data)
            else:
                print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def beep():
    """Non-blocking beep."""
    threading.Thread(target=_beep_thread, daemon=True).start()


# ────────────────────────────────────────────────────────────────────────────
#  ZONE DRAWING UI
# ────────────────────────────────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global zone_points, drawing_done
    if drawing_done:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))


def draw_zone_preview(frame: np.ndarray, points: list) -> np.ndarray:
    overlay = frame.copy()
    if len(points) > 1:
        cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, (0, 200, 255), 2)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, (0, 100, 255), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def define_zone(cap: cv2.VideoCapture) -> np.ndarray:
    global zone_points, drawing_done

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from video source.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    win = "Define Zone  |  Click = add point  |  ENTER = confirm  |  R = reset"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        display = draw_zone_preview(frame.copy(), zone_points)
        for i, line in enumerate([
            "Click to add polygon vertices",
            f"Points added: {len(zone_points)}",
            "ENTER to confirm (min 3)  |  R to reset",
        ]):
            cv2.putText(display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(zone_points) >= 3:   
            drawing_done = True
            break
        elif key == ord('r'):
            zone_points = []

    cv2.destroyWindow(win)
    return np.array(zone_points, dtype=np.int32)


# ────────────────────────────────────────────────────────────────────────────
#  COLLISION: bounding box touches/overlaps polygon
# ────────────────────────────────────────────────────────────────────────────
def segments_intersect(p1, p2, p3, p4) -> bool:
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def bbox_touches_zone(x1: int, y1: int, x2: int, y2: int,
                      polygon: np.ndarray) -> bool:
    """True if bbox rectangle touches or overlaps the zone polygon."""
    corners = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

    # 1. Any corner of box inside polygon
    for c in corners:
        if cv2.pointPolygonTest(polygon, (float(c[0]), float(c[1])), False) >= 0:
            return True

    # 2. Any polygon vertex inside box
    for pt in polygon:
        px, py = int(pt[0]), int(pt[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    # 3. Any edge of polygon intersects any edge of box
    box_edges = [(corners[i], corners[(i+1)%4]) for i in range(4)]
    n = len(polygon)
    poly_edges = [(tuple(polygon[i]), tuple(polygon[(i+1)%n])) for i in range(n)]
    for be in box_edges:
        for pe in poly_edges:
            if segments_intersect(be[0], be[1], pe[0], pe[1]):
                return True

    return False


# ────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────
def run(source):
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    print("[INFO] First frame loaded. Please draw your geofence zone.")
    zone = define_zone(cap)
    print(f"[INFO] Zone defined with {len(zone)} vertices.")

    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    win = "Geofence Detector  |  Q = quit"
    cv2.namedWindow(win)

    ALERT_COLOR = (0, 0, 255)
    SAFE_COLOR  = (0, 255, 0)
    FLASH_DURATION = 8   # frames the red flash lasts

    flash_frames_left = 0
    prev_intrusion = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            # ── YOLOv8 inference ─────────────────────────────────────────
            results = model(frame, classes=[0], verbose=False)[0]
            intrusion_detected = False

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                touching = bbox_touches_zone(x1, y1, x2, y2, zone)

                box_color = ALERT_COLOR if touching else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"Person {conf:.2f}{'  !! ALERT' if touching else ''}"
                cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                if touching:
                    intrusion_detected = True

            # ── Beep on rising edge (new intrusion event) ─────────────────
            if intrusion_detected and not prev_intrusion:
                beep()
                flash_frames_left = FLASH_DURATION
            if intrusion_detected:
                flash_frames_left = FLASH_DURATION   # keep flashing
            prev_intrusion = intrusion_detected

            # ── Draw zone polygon ─────────────────────────────────────────
            zone_color = ALERT_COLOR if intrusion_detected else SAFE_COLOR
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], zone_color)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
            cv2.polylines(frame, [zone], True, zone_color, 2)

            # ── Full-screen red flash ─────────────────────────────────────
            if flash_frames_left > 0:
                flash = np.zeros_like(frame)
                flash[:] = (0, 0, 220)
                alpha = 0.35 * (flash_frames_left / FLASH_DURATION)
                frame = cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0)
                flash_frames_left -= 1

            # ── Status banner ─────────────────────────────────────────────
            h, w = frame.shape[:2]
            status = "!! INTRUSION DETECTED !!" if intrusion_detected else "Zone Clear"
            banner_color = ALERT_COLOR if intrusion_detected else SAFE_COLOR
            cv2.rectangle(frame, (0, 0), (w, 42), (0, 0, 0), -1)
            tw = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
            cv2.putText(frame, status, ((w - tw) // 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, banner_color, 2)

            cv2.imshow(win, frame)

            # Quit on Q or window close
            key = cv2.waitKey(1) & 0xFF
            window_open = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
            if key == ord('q') or not window_open:
                print("[INFO] Quit requested.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user (Ctrl+C).")

    finally:
        print("[INFO] Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("[INFO] Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Geofencing Person Detector")
    parser.add_argument("--source", default="0",
                        help="Video source: webcam index (0,1,...) or path to video file")
    args = parser.parse_args()
    run(args.source)