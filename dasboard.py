"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AI SECURITY DASHBOARD  –  4-Zone Threat Detection                ║
║  Zone 1 │ Fire & Smoke   (YOLOv8,  classes: fire / smoke)                  ║
║  Zone 2 │ Fall / Faint   (YOLOv8,  class:  Fall Detected)                  ║
║  Zone 3 │ PPE Compliance (YOLOv5,  torch.hub)                              ║
║  Zone 4 │ Geofence       (YOLOv8n, COCO person)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY FIX: One shared CameraReader thread owns the VideoCapture.
         All zone workers pull frames from it — no camera conflicts.

Requirements:
    pip install ultralytics opencv-python numpy torch torchvision pandas seaborn tqdm
"""

import sys, os, threading, time, math
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch not available – Zone 3 PPE disabled.")
    print("       Fix: pip install torch torchvision pandas seaborn tqdm")

# ═══════════════════════════════════════════════════════════════════
#  ZONE TYPES
# ═══════════════════════════════════════════════════════════════════
ZONE_TYPE_STANDARD = "standard"
ZONE_TYPE_PPE      = "ppe"
ZONE_TYPE_GEO      = "geo"

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
FIRE_MODEL_PATH  = r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\FIRE\best.pt"
FAINT_MODEL_PATH = r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\FAINT\best.pt"
PPE_MODEL_PATH   = r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\PPE\best.pt"

WEBCAM_INDEX = 0   # single webcam shared by ALL zones

CONF_FIRE  = 0.20
CONF_FAINT = 0.30
CONF_PPE   = 0.40
CONF_GEO   = 0.50

# ═══════════════════════════════════════════════════════════════════
#  ALERT CLASSES
# ═══════════════════════════════════════════════════════════════════
FIRE_ALERT_CLASSES  = {"fire", "smoke"}
FAINT_ALERT_CLASSES = {"fall detected"}

PPE_WORKER_NAMES = {"worker", "person"}
PPE_SAFETY_NAMES = {"vest", "helmet", "hard-hat", "safety vest",
                    "safety-vest", "hardhat", "protective helmet"}

# ═══════════════════════════════════════════════════════════════════
#  ACTION MESSAGES
# ═══════════════════════════════════════════════════════════════════
ACTION_MAP = {
    "fire":          ("FIRE DETECTED",      "Activate sprinklers / Evacuate / Call 18"),
    "smoke":         ("SMOKE DETECTED",     "Check fire source / Ventilate / Call 18"),
    "fall detected": ("PERSON FALLEN",      "Dispatch first aid / Call 15 (SAMU)"),
    "ppe_missing":   ("PPE NON-COMPLIANCE", "Stop work / Equip worker / Log incident"),
    "geofence":      ("ZONE INTRUSION",     "Dispatch security / Review access / Alert guard"),
}

# ═══════════════════════════════════════════════════════════════════
#  COLOURS (BGR)
# ═══════════════════════════════════════════════════════════════════
C_RED    = (0,   30, 220)
C_ORANGE = (0,  140, 255)
C_GREEN  = (50, 210,  80)
C_CYAN   = (220, 210,  0)
C_WHITE  = (255, 255, 255)
C_DARK   = (18,   18,  28)

# ═══════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════
DASH_W, DASH_H = 1600, 900
ZONE_W, ZONE_H = 760, 400
TOP_BAR_H      = 48
ACTION_PANEL_W = 340
ACTION_PANEL_H = 400
MARGIN         = 8

ZONE_POS = [
    (MARGIN,          TOP_BAR_H + MARGIN),
    (MARGIN*2+ZONE_W, TOP_BAR_H + MARGIN),
    (MARGIN,          TOP_BAR_H + MARGIN*2 + ZONE_H),
    (MARGIN*2+ZONE_W, TOP_BAR_H + MARGIN*2 + ZONE_H),
]
ACTION_PANEL_X = DASH_W - ACTION_PANEL_W - MARGIN
ACTION_PANEL_Y = TOP_BAR_H + ZONE_H*2 - ACTION_PANEL_H + MARGIN

# ═══════════════════════════════════════════════════════════════════
#  BEEP
# ═══════════════════════════════════════════════════════════════════
_beep_lock      = threading.Lock()
_last_beep_time = 0.0
BEEP_COOLDOWN   = 1.5

def _beep_thread():
    global _last_beep_time
    with _beep_lock:
        now = time.time()
        if now - _last_beep_time < BEEP_COOLDOWN:
            return
        _last_beep_time = now
    try:
        import winsound
        winsound.Beep(1100, 350)
    except Exception:
        print("\a", end="", flush=True)

def beep():
    threading.Thread(target=_beep_thread, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════
#  SHARED CAMERA READER
#  One thread owns the VideoCapture. All zones call get_frame().
# ═══════════════════════════════════════════════════════════════════
class CameraReader(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.source   = source
        self._lock    = threading.Lock()
        self._frame   = None
        self._running = True
        self.ready    = False   # True once first frame received

    def get_frame(self):
        """Returns a copy of the latest frame, or None if not ready."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self): self._running = False

    def run(self):
        try:
            src = int(self.source)
        except (ValueError, TypeError):
            src = self.source

        print(f"[CAM] Opening camera {src} ...")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[CAM] ERROR: Cannot open camera {src}")
            return

        # Warm up
        for _ in range(5):
            cap.read()

        print("[CAM] Camera ready — streaming frames to all zones.")
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self.ready  = True
            else:
                time.sleep(0.01)

        cap.release()
        print("[CAM] Camera released.")

# ═══════════════════════════════════════════════════════════════════
#  GEOFENCE DRAWING
# ═══════════════════════════════════════════════════════════════════
_geo_pts  = []
_geo_done = False

def _geo_mouse_cb(event, x, y, flags, param):
    global _geo_pts, _geo_done
    if not _geo_done and event == cv2.EVENT_LBUTTONDOWN:
        _geo_pts.append((x, y))

def define_geofence(still_frame):
    global _geo_pts, _geo_done
    _geo_pts  = []
    _geo_done = False

    win = "Draw Geofence  |  Click=add point  |  ENTER=confirm (min 3)  |  R=reset"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 560)
    cv2.setMouseCallback(win, _geo_mouse_cb)

    while True:
        disp = still_frame.copy()
        if len(_geo_pts) > 1:
            cv2.polylines(disp,[np.array(_geo_pts,np.int32)],False,(0,200,255),2)
        if len(_geo_pts) >= 3:
            cv2.polylines(disp,[np.array(_geo_pts,np.int32)],True,(0,200,255),1)
        for i, pt in enumerate(_geo_pts):
            cv2.circle(disp,pt,6,(0,80,255),-1)
            cv2.putText(disp,str(i+1),(pt[0]+7,pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,C_WHITE,1)
        for i,(txt,col) in enumerate([
            ("Models loading in background  \u2713", C_GREEN),
            (f"Geofence points: {len(_geo_pts)}",   C_CYAN),
            ("ENTER to confirm (>= 3)  |  R to reset", (0,230,255)),
        ]):
            cv2.putText(disp,txt,(10,28+i*26),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)
        cv2.imshow(win,disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(_geo_pts) >= 3:
            _geo_done = True
            break
        elif key == ord('r'):
            _geo_pts = []

    cv2.destroyWindow(win)
    return np.array(_geo_pts, dtype=np.int32)

# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════
def _segs_intersect(p1,p2,p3,p4):
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    d1=cross(p3,p4,p1); d2=cross(p3,p4,p2)
    d3=cross(p1,p2,p3); d4=cross(p1,p2,p4)
    return ((d1>0 and d2<0)or(d1<0 and d2>0)) and \
           ((d3>0 and d4<0)or(d3<0 and d4>0))

def bbox_in_zone(x1,y1,x2,y2,poly):
    corners=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    for c in corners:
        if cv2.pointPolygonTest(poly,(float(c[0]),float(c[1])),False)>=0: return True
    for pt in poly:
        px,py=int(pt[0]),int(pt[1])
        if x1<=px<=x2 and y1<=py<=y2: return True
    n=len(poly)
    be=[(corners[i],corners[(i+1)%4]) for i in range(4)]
    pe=[(tuple(poly[i]),tuple(poly[(i+1)%n])) for i in range(n)]
    return any(_segs_intersect(b[0],b[1],p[0],p[1]) for b in be for p in pe)

# ═══════════════════════════════════════════════════════════════════
#  MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════
class ModelWrapper:
    def __init__(self, model_path, zone_type, conf):
        self.conf      = conf
        self.zone_type = zone_type
        self.names     = {}
        self._v5       = False
        self._model    = None
        if zone_type == ZONE_TYPE_PPE:
            self._load_v5(model_path)
        else:
            self._load_v8(model_path)

    def _load_v8(self, path):
        self._model = YOLO(path)
        self.names  = {k: v.lower().strip() for k,v in self._model.names.items()}
        print(f"    [YOLOv8] names = {self.names}")

    def _load_v5(self, path):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "torch not installed.\n"
                "Fix: pip install torch torchvision pandas seaborn tqdm")
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            self._model = torch.hub.load(
                'ultralytics/yolov5','custom',
                path=path,force_reload=False,
                verbose=False,trust_repo=True)
        self._model.conf = self.conf
        self._model.iou  = 0.45
        self._v5 = True
        raw = self._model.names
        if isinstance(raw, dict):
            self.names = {k: str(v).lower().strip() for k,v in raw.items()}
        elif isinstance(raw, list):
            self.names = {i: str(n).lower().strip() for i,n in enumerate(raw)}
        print(f"    [YOLOv5] names = {self.names}")

    def infer(self, frame):
        if self._v5:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._model(rgb)
            out = []
            for *xyxy,c,cls_id in results.xyxy[0].tolist():
                if c < self.conf: continue
                cls_id=int(cls_id)
                name=self.names.get(cls_id,str(cls_id))
                x1,y1,x2,y2=map(int,xyxy)
                out.append((cls_id,float(c),name,x1,y1,x2,y2))
            return out
        else:
            results=self._model(frame,conf=self.conf,verbose=False)[0]
            out=[]
            for box in results.boxes:
                cls_id=int(box.cls[0]); c=float(box.conf[0])
                name=self.names.get(cls_id,str(cls_id))
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                out.append((cls_id,c,name,x1,y1,x2,y2))
            return out

# ═══════════════════════════════════════════════════════════════════
#  ZONE WORKER  –  receives frames from shared CameraReader
# ═══════════════════════════════════════════════════════════════════
class ZoneWorker(threading.Thread):
    def __init__(self, zone_id, camera, model_path, conf,
                 alert_classes, label,
                 zone_type=ZONE_TYPE_STANDARD, geo_zone=None):
        super().__init__(daemon=True)
        self.zone_id       = zone_id
        self.camera        = camera          # CameraReader instance
        self.model_path    = model_path
        self.conf          = conf
        self.alert_classes = alert_classes
        self.label         = label
        self.zone_type     = zone_type
        self._geo_zone     = geo_zone
        self._geo_lock     = threading.Lock()

        self.frame_lock    = threading.Lock()
        self.latest_frame  = None
        self.alert         = False
        self.active_events = []
        self.fps           = 0.0
        self._flash        = 0
        self._running      = True

    def set_geo_zone(self, zone):
        with self._geo_lock:
            self._geo_zone = zone

    def get_geo_zone(self):
        with self._geo_lock:
            return self._geo_zone

    def stop(self): self._running = False

    def _status_frame(self, msg, color=C_ORANGE, hint=""):
        ph = np.zeros((ZONE_H,ZONE_W,3),dtype=np.uint8)
        cv2.rectangle(ph,(0,0),(ZONE_W,26),(30,30,50),-1)
        cv2.putText(ph,f"  ZONE {self.zone_id}: {self.label}",
                    (4,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,C_WHITE,1)
        cv2.putText(ph,msg,(14,ZONE_H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
        if hint:
            for i,line in enumerate(hint.split("\n")):
                cv2.putText(ph,line,(14,ZONE_H//2+22+i*22),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,C_ORANGE,1)
        return ph

    def run(self):
        # Show status immediately
        with self.frame_lock:
            self.latest_frame = self._status_frame("Loading model...")

        # Load model (no camera needed)
        print(f"[INFO] Zone {self.zone_id}: loading model ...")
        try:
            model = ModelWrapper(self.model_path, self.zone_type, self.conf)
        except Exception as e:
            err  = str(e)
            hint = ""
            if any(x in err.lower() for x in ["pandas","tqdm","seaborn"]):
                hint = "Fix: pip install pandas seaborn tqdm"
            elif "torch" in err.lower():
                hint = "Fix: pip install torch torchvision"
            print(f"[ERROR] Zone {self.zone_id}: {err}")
            with self.frame_lock:
                self.latest_frame = self._status_frame(
                    "Model error",C_RED,err[:65]+("\n"+hint if hint else ""))
            return

        # Geo zone: wait for polygon injection
        if self.zone_type == ZONE_TYPE_GEO:
            with self.frame_lock:
                self.latest_frame = self._status_frame(
                    "Waiting for geofence...",C_CYAN,
                    "Draw polygon in popup window")
            while self._running and self.get_geo_zone() is None:
                time.sleep(0.05)
            if not self._running:
                return

        # Wait for camera to be ready
        with self.frame_lock:
            self.latest_frame = self._status_frame("Waiting for camera...",C_CYAN)
        while self._running and not self.camera.ready:
            time.sleep(0.05)

        print(f"[INFO] Zone {self.zone_id}: running. conf={self.conf}")
        t0 = time.time(); fc = 0

        while self._running:
            # Get latest frame from shared camera
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame,(ZONE_W,ZONE_H))
            alert       = False
            active_evts = []

            try:
                detections = model.infer(frame)
            except Exception as e:
                print(f"[WARN] Zone {self.zone_id} infer: {e}")
                detections = []

            # ── PPE logic ────────────────────────────────────────
            if self.zone_type == ZONE_TYPE_PPE:
                workers=[(x1,y1,x2,y2)
                         for (_,_,n,x1,y1,x2,y2) in detections
                         if n in PPE_WORKER_NAMES]
                safety =[(n,x1,y1,x2,y2)
                         for (_,_,n,x1,y1,x2,y2) in detections
                         if n in PPE_SAFETY_NAMES]

                if fc==0 and detections:
                    print(f"[DEBUG Z3] {[(n,round(c,2)) for _,c,n,*_ in detections]}")

                def overlaps(ax1,ay1,ax2,ay2,bx1,by1,bx2,by2):
                    ix1=max(ax1,bx1); iy1=max(ay1,by1)
                    ix2=min(ax2,bx2); iy2=min(ay2,by2)
                    return ix2>ix1 and iy2>iy1 and (ix2-ix1)*(iy2-iy1)>200

                for (wx1,wy1,wx2,wy2) in workers:
                    has_vest  =any(n in {"vest","safety vest","safety-vest"}
                                   and overlaps(wx1,wy1,wx2,wy2,bx1,by1,bx2,by2)
                                   for (n,bx1,by1,bx2,by2) in safety)
                    has_helmet=any(n in {"helmet","hard-hat","hardhat","protective helmet"}
                                   and overlaps(wx1,wy1,wx2,wy2,bx1,by1,bx2,by2)
                                   for (n,bx1,by1,bx2,by2) in safety)
                    missing=(["VEST"] if not has_vest else [])+(["HELMET"] if not has_helmet else [])
                    w_alert=bool(missing)
                    col=C_RED if w_alert else C_GREEN
                    cv2.rectangle(frame,(wx1,wy1),(wx2,wy2),col,2)
                    txt=f"MISSING:{','.join(missing)}" if missing else "PPE OK"
                    (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
                    cv2.rectangle(frame,(wx1,max(wy1-th-6,0)),(wx1+tw+4,wy1),col,-1)
                    cv2.putText(frame,txt,(wx1+2,max(wy1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.48,C_WHITE,1)
                    if w_alert:
                        alert=True
                        if "ppe_missing" not in active_evts:
                            active_evts.append("ppe_missing")

                for (n,bx1,by1,bx2,by2) in safety:
                    cv2.rectangle(frame,(bx1,by1),(bx2,by2),(200,160,0),1)
                    cv2.putText(frame,n,(bx1+2,max(by1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,0),1)

            # ── Standard / Geofence ──────────────────────────────
            else:
                geo = self.get_geo_zone()
                for (cls_id,conf,name,x1,y1,x2,y2) in detections:
                    name_lc=name.lower().strip()
                    if self.zone_type==ZONE_TYPE_GEO and geo is not None:
                        is_alert=bbox_in_zone(x1,y1,x2,y2,geo)
                        if is_alert and "geofence" not in active_evts:
                            active_evts.append("geofence")
                    else:
                        is_alert=name_lc in self.alert_classes
                        if is_alert and name_lc not in active_evts:
                            active_evts.append(name_lc)
                    col=C_RED if is_alert else C_ORANGE
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                    lbl=f"{name} {conf:.2f}"+(" !" if is_alert else "")
                    (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                    cv2.rectangle(frame,(x1,max(y1-th-6,0)),(x1+tw+4,y1),col,-1)
                    cv2.putText(frame,lbl,(x1+2,max(y1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,C_WHITE,1)
                    if is_alert: alert=True

                if self.zone_type==ZONE_TYPE_GEO and geo is not None:
                    zc=C_RED if alert else C_GREEN
                    ov=frame.copy()
                    cv2.fillPoly(ov,[geo],zc)
                    frame=cv2.addWeighted(ov,0.22,frame,0.78,0)
                    cv2.polylines(frame,[geo],True,zc,2)

            # ── Flash ────────────────────────────────────────────
            if alert: self._flash=6
            if self._flash>0:
                fl=np.zeros_like(frame); fl[:]=(0,0,200)
                a=0.28*(self._flash/6)
                frame=cv2.addWeighted(fl,a,frame,1-a,0)
                self._flash-=1

            # ── Title bar ────────────────────────────────────────
            cv2.rectangle(frame,(0,0),(ZONE_W,26),
                          C_RED if alert else (30,30,50),-1)
            cv2.putText(frame,f"  ZONE {self.zone_id}: {self.label}",
                        (4,18),cv2.FONT_HERSHEY_SIMPLEX,0.6,C_WHITE,1)
            st =("!! ALERT" if alert else "CLEAR")
            tw2=cv2.getTextSize(st,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)[0][0]
            cv2.putText(frame,st,(ZONE_W-tw2-10,18),cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,C_RED if alert else C_GREEN,2)
            cv2.putText(frame,f"conf>{self.conf}  {self.fps:.1f}fps",
                        (ZONE_W-155,ZONE_H-8),cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,(110,110,110),1)

            fc+=1
            el=time.time()-t0
            if el>=1.0:
                self.fps=fc/el; fc=0; t0=time.time()

            with self.frame_lock:
                self.latest_frame  = frame.copy()
                self.alert         = alert
                self.active_events = active_evts

# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD UI
# ═══════════════════════════════════════════════════════════════════
def draw_top_bar(canvas, any_alert, t):
    cv2.rectangle(canvas,(0,0),(DASH_W,TOP_BAR_H),(14,14,24),-1)
    pulse=(0,60+int(40*abs(math.sin(t*2))),180) if not any_alert \
          else (0,0,150+int(80*abs(math.sin(t*4))))
    cv2.line(canvas,(0,TOP_BAR_H-1),(DASH_W,TOP_BAR_H-1),pulse,2)
    cv2.putText(canvas,"AI SECURITY DASHBOARD",(20,33),
                cv2.FONT_HERSHEY_DUPLEX,0.9,C_WHITE,1)
    cv2.putText(canvas,time.strftime("%H:%M:%S"),(DASH_W-200,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,C_CYAN,2)
    cv2.putText(canvas,time.strftime("%a %d %b %Y"),(DASH_W-200,46),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(160,160,180),1)
    if any_alert:
        if int(t*4)%2==0:
            cv2.putText(canvas,"  !! THREAT DETECTED !!  ",
                        (DASH_W//2-160,32),cv2.FONT_HERSHEY_SIMPLEX,0.75,C_RED,2)
    else:
        cv2.putText(canvas,"  ALL SYSTEMS NOMINAL",
                    (DASH_W//2-120,32),cv2.FONT_HERSHEY_SIMPLEX,0.65,C_GREEN,1)


def draw_action_panel(canvas, workers, t):
    px,py=ACTION_PANEL_X,ACTION_PANEL_Y
    pw,ph=ACTION_PANEL_W,ACTION_PANEL_H
    cv2.rectangle(canvas,(px,py),(px+pw,py+ph),(18,18,35),-1)
    cv2.rectangle(canvas,(px,py),(px+pw,py+ph),(60,60,100),1)
    cv2.rectangle(canvas,(px,py),(px+pw,py+30),(35,35,65),-1)
    cv2.putText(canvas,"  ACTION REQUIRED",(px+6,py+21),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,C_CYAN,1)

    events=[]
    for w in workers:
        with w.frame_lock:
            evts=list(w.active_events)
        for ev in evts:
            if ev in ACTION_MAP:
                events.append((w.zone_id,ev,ACTION_MAP[ev]))

    y=py+52
    if not events:
        cv2.putText(canvas,"No active alerts",(px+10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,140),1)
        return

    for (zone_id,_,(title,action)) in events[:5]:
        bc=C_RED if int(t*3)%2==0 else (80,0,0)
        cv2.circle(canvas,(px+14,y-4),6,bc,-1)
        cv2.rectangle(canvas,(px+26,y-14),(px+80,y+2),(50,20,20),-1)
        cv2.putText(canvas,f"ZONE {zone_id}",(px+28,y-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,150,150),1)
        cv2.putText(canvas,title,(px+86,y-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,C_RED,1)
        ay=y+14
        for part in action.split(" / "):
            cv2.putText(canvas,f"  > {part}",(px+14,ay),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,(200,200,220),1)
            ay+=14
        cv2.line(canvas,(px+8,ay+4),(px+pw-8,ay+4),(50,50,80),1)
        y=ay+18
        if y>py+ph-10: break


def draw_ticker(canvas, workers, t):
    bar_y=DASH_H-24
    cv2.rectangle(canvas,(0,bar_y),(DASH_W,DASH_H),(12,12,22),-1)
    msgs=[]
    for w in workers:
        with w.frame_lock:
            for ev in w.active_events:
                if ev in ACTION_MAP:
                    msgs.append(f"  [ZONE {w.zone_id}] {ACTION_MAP[ev][0]}"
                                f"  \u25cf  {ACTION_MAP[ev][1]}     ")
    if not msgs:
        txt="  ALL ZONES CLEAR  \u25cf  AI SECURITY DASHBOARD ACTIVE  \u25cf  MONITORING..."
        c=C_GREEN
    else:
        txt="  !!!  "+"".join(msgs*3); c=C_RED
    total =cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)[0][0]
    offset=int(t*120)%(total+DASH_W)
    cv2.putText(canvas,txt,(DASH_W-offset,bar_y+16),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1)

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("="*66)
    print("  AI SECURITY DASHBOARD  –  starting up")
    print("="*66)

    # ── STEP 1: Start shared camera immediately ──────────────────
    camera = CameraReader(WEBCAM_INDEX)
    camera.start()

    # Wait briefly for first frame (for geofence still)
    print("[CAM] Waiting for first frame...")
    deadline = time.time() + 8
    while not camera.ready and time.time() < deadline:
        time.sleep(0.05)

    still = camera.get_frame()
    if still is None:
        print("[WARN] No camera frame – using blank canvas for geofence.")
        still = np.zeros((480,640,3),dtype=np.uint8)
        cv2.putText(still,"No camera feed",(180,240),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,C_ORANGE,2)

    # ── STEP 2: Start ALL model workers in parallel ───────────────
    # They all share the same CameraReader — no camera conflicts.
    zones = [
        ZoneWorker(1, camera, FIRE_MODEL_PATH,  CONF_FIRE,
                   FIRE_ALERT_CLASSES,  "FIRE & SMOKE DETECTION",
                   zone_type=ZONE_TYPE_STANDARD),
        ZoneWorker(2, camera, FAINT_MODEL_PATH, CONF_FAINT,
                   FAINT_ALERT_CLASSES, "FALL / FAINT DETECTION",
                   zone_type=ZONE_TYPE_STANDARD),
        ZoneWorker(3, camera, PPE_MODEL_PATH,   CONF_PPE,
                   set(),               "PPE COMPLIANCE",
                   zone_type=ZONE_TYPE_PPE),
        ZoneWorker(4, camera, "yolov8n.pt",     CONF_GEO,
                   set(),               "GEOFENCE INTRUSION",
                   zone_type=ZONE_TYPE_GEO, geo_zone=None),
    ]

    print("[INFO] Starting all model loaders in parallel...")
    for z in zones:
        z.start()

    # ── STEP 3: Draw geofence while models load ───────────────────
    print("[INFO] Draw your geofence in the popup. Models loading in background.")
    geo_zone = define_geofence(still)
    print(f"[INFO] Geofence confirmed: {len(geo_zone)} vertices.")

    # ── STEP 4: Inject geo_zone → Zone 4 starts inference ────────
    zones[3].set_geo_zone(geo_zone)
    print("[INFO] All zones now active!")

    # ── STEP 5: Dashboard loop ────────────────────────────────────
    WIN = "AI Security Dashboard  |  Q = quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DASH_W, DASH_H)

    t_start    = time.time()
    prev_alert = False
    print("[INFO] Dashboard live. Press Q to quit.\n")

    while True:
        t      = time.time()-t_start
        canvas = np.full((DASH_H,DASH_W,3),C_DARK,dtype=np.uint8)
        any_alert = False

        for zone,(zx,zy) in zip(zones,ZONE_POS):
            with zone.frame_lock:
                fr    = zone.latest_frame
                alert = zone.alert
            tile=cv2.resize(
                fr if fr is not None else np.zeros((ZONE_H,ZONE_W,3),dtype=np.uint8),
                (ZONE_W,ZONE_H))
            if alert:
                bc=C_RED if int(t*4)%2==0 else (100,0,0)
                cv2.rectangle(tile,(0,0),(ZONE_W-1,ZONE_H-1),bc,5)
                any_alert=True
            canvas[zy:zy+ZONE_H, zx:zx+ZONE_W]=tile

        if any_alert and not prev_alert: beep()
        prev_alert=any_alert

        if any_alert and int(t*4)%2==0:
            fl=canvas[:TOP_BAR_H].copy(); fl[:]=(0,0,60)
            canvas[:TOP_BAR_H]=cv2.addWeighted(fl,0.6,canvas[:TOP_BAR_H],0.4,0)

        draw_top_bar(canvas,any_alert,t)
        draw_action_panel(canvas,zones,t)
        draw_ticker(canvas,zones,t)

        for zx,zy in ZONE_POS:
            cv2.line(canvas,(zx,zy+ZONE_H),(zx+ZONE_W,zy+ZONE_H),(40,40,60),1)
            cv2.line(canvas,(zx+ZONE_W,zy),(zx+ZONE_W,zy+ZONE_H),(40,40,60),1)

        cv2.imshow(WIN,canvas)
        key=cv2.waitKey(1)&0xFF
        try:    vis=cv2.getWindowProperty(WIN,cv2.WND_PROP_VISIBLE)
        except: vis=0
        if key==ord('q') or vis<1: break

    # ── Cleanup ───────────────────────────────────────────────────
    print("[INFO] Shutting down...")
    for z in zones: z.stop()
    camera.stop()
    for z in zones: z.join(timeout=3)
    camera.join(timeout=3)
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)
    print("[INFO] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()