"""
Model Diagnostics – run this FIRST to see exactly what your models detect.
It will print class names, run one inference frame from webcam, and show
all detections with their class IDs, names, and confidence scores.

Usage:
    python diagnose_models.py
"""

import cv2
import numpy as np
from ultralytics import YOLO

MODELS = {
    "FIRE":  r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\FIRE\best.pt",
    "FAINT": r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\FAINT\best.pt",
    "PPE":   r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\PPE\best.pt",
}

WEBCAM_SOURCE = 0   # change if needed

print("\n" + "="*65)
print("  AI SECURITY DASHBOARD  –  MODEL DIAGNOSTICS")
print("="*65)

# Grab a single frame from webcam for test inference
print(f"\n[INFO] Opening webcam {WEBCAM_SOURCE} for test frame...")
cap = cv2.VideoCapture(WEBCAM_SOURCE)
test_frame = None
if cap.isOpened():
    ret, test_frame = cap.read()
    cap.release()
    if ret:
        print(f"[INFO] Got test frame: {test_frame.shape}")
    else:
        print("[WARN] Could not read frame – will run inference on blank image")
else:
    print("[WARN] Could not open webcam – will run inference on blank image")

if test_frame is None:
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

for zone_name, model_path in MODELS.items():
    print(f"\n{'─'*65}")
    print(f"  ZONE: {zone_name}")
    print(f"  Path: {model_path}")
    print(f"{'─'*65}")
    try:
        model = YOLO(model_path)

        print(f"\n  model.names (class ID → label):")
        for cid, cname in model.names.items():
            print(f"    [{cid}] '{cname}'")

        print(f"\n  Running inference on test frame (conf=0.01 to catch everything)...")
        results = model(test_frame, conf=0.01, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            print("  No detections at conf=0.01 – model may need a better test image,")
            print("  but the class names above are still valid.")
        else:
            print(f"  Detections ({len(results.boxes)} found):")
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = model.names.get(cls_id, f"ID:{cls_id}")
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                print(f"    cls_id={cls_id}  name='{name}'  conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")

        # Show annotated frame in a window
        annotated = results.plot()
        annotated = cv2.resize(annotated, (800, 500))
        win_title = f"Diagnostics: {zone_name} (any key to continue)"
        cv2.imshow(win_title, annotated)
        print(f"\n  >> Preview window open. Press any key to continue to next model.")
        cv2.waitKey(0)
        cv2.destroyWindow(win_title)

    except Exception as e:
        print(f"  [ERROR] Failed to load/run model: {e}")

cv2.destroyAllWindows()

print("\n" + "="*65)
print("  DIAGNOSTICS COMPLETE")
print("  Copy the class names above into ai_security_dashboard.py:")
print()
print("  FIRE_ALERT_CLASSES  = { exact names that mean fire/gas }")
print("  FAINT_ALERT_CLASSES = { exact name for fall/faint }")
print("  PPE_WORKER_NAMES    = { exact name for worker }")
print("  PPE_SAFETY_NAMES    = { exact names for vest, helmet }")
print("="*65 + "\n")