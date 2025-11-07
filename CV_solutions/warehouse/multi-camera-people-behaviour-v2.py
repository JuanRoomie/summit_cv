#!/usr/bin/env python3
# multi-camera-people-behaviour.py
# YOLO (Ultralytics) -> (pose/bbox) -> overlay zones -> RTSP (MediaMTX) + (opcional) eventos al backend

import threading
import subprocess, shlex
import time
import numpy as np
import cv2
import json
from typing import Dict, List, Tuple
from ultralytics import YOLO
from metadata_writer import MetadataWriter

# <<< NUEVO: importar la función simple de alertas >>>
from zone_alerts import check_and_post_zone_alerts

"""
Ejemplo de zones_config.json (claveado por nombre de stream, p.ej. "people-beh"):
{
  "people-beh": {
    "zones": [
      {"id":"restricted_dock","name":"Restricted Dock","prohibited":true,"color":[255,0,0],
       "points":[[0.12,0.18],[0.32,0.20],[0.35,0.45],[0.15,0.47]]}
    ]
  }
}
"""

# ===== Writer global + lock =====
from metadata_writer import MetadataWriter
metadata_writer = MetadataWriter(
    base_filename="metadata_personas",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración =====
MODEL_PATH = "yolo11s-pose.engine"
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.20:554"
]
STREAM_NAMES = ["people-beh"]
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.60

DRAW_MODE = "kpts"  # 'bbox' | 'kpts' | 'both'

# Backend para alertas
API_BASE = "http://192.168.1.72:8000"

# ===== ZONAS =====
ZONES_CONFIG_PATH = "zones_config.json"
try:
    with open(ZONES_CONFIG_PATH, "r") as f:
        ZONES_CONFIG = json.load(f)
    print(f"[ZONES] Cargado {ZONES_CONFIG_PATH}")
except Exception as e:
    print(f"[ZONES] No se pudo cargar {ZONES_CONFIG_PATH}: {e}")
    ZONES_CONFIG = {}

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(cmd)

def point_in_poly(pt: Tuple[int, int], poly_pts: np.ndarray) -> bool:
    res = cv2.pointPolygonTest(poly_pts, (float(pt[0]), float(pt[1])), measureDist=False)
    return res >= 0.0

def draw_zones(frame: np.ndarray, zones: List[dict], debug: bool = False):
    h, w = frame.shape[:2]
    drawn = 0
    overlay = frame.copy()
    for z in zones:
        pts_norm = z.get("points", [])
        if not pts_norm:
            continue
        is_normalized = all(
            isinstance(p, (list, tuple)) and len(p) == 2 and 0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0
            for p in pts_norm
        )
        if is_normalized:
            pts = np.array([(int(float(px) * w), int(float(py) * h)) for (px, py) in pts_norm], dtype=np.int32)
        else:
            pts = np.array([(int(float(px)), int(float(py))) for (px, py) in pts_norm], dtype=np.int32)
        rgb = z.get("color", [0, 255, 0])
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0])) if len(rgb) == 3 else (0, 255, 0)
        alpha = 0.20 if z.get("prohibited") else 0.12
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
        thickness = 3 if z.get("prohibited") else 2
        cv2.polylines(frame, [pts], isClosed=True, color=bgr, thickness=thickness)
        label = z.get("name") or z.get("id", "zone")
        if z.get("prohibited"):
            label += " (PROHIBITED)"
        p0 = pts[0].tolist()
        cv2.rectangle(frame, (p0[0], max(0, p0[1] - 22)), (p0[0] + 8 * len(label), p0[1] - 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (p0[0] + 4, p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        drawn += 1
    if debug:
        print(f"[ZONES] Drawn {drawn}/{len(zones)} on frame {w}x{h}")

def run_tracker_and_publish(src, publish_name, source_index):
    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    # Zonas para este stream
    zones_for_cam = ZONES_CONFIG.get(publish_name, {}).get("zones", [])
    print(f"[ZONES] Stream '{publish_name}': {len(zones_for_cam)} zone(s) loaded")

    try:
        for r in model.track(source=src, stream=True, conf=CONF, save=False, show=False, verbose=False):
            boxes = getattr(r, "boxes", None)
            class_name_for_idx = r.names if hasattr(r, "names") else {}

            detections = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls  = boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.id is not None:
                    obj_ids = boxes.id.detach().cpu().numpy().astype(int)
                else:
                    obj_ids = np.full((len(boxes),), -1, dtype=int)
                ts = time.time()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    leftx  = float(x1); topy = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])
                    class_name = str(class_name_for_idx.get(class_id, "unknown")).replace(" ", "_")
                    detections.append({
                        "obj_id": object_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox": (leftx, topy, width, height),
                        "xyxy": (x1, y1, x2, y2)
                    })
                    with write_lock:
                        metadata_writer.write(
                            time=f"{ts}", source=f"{source_index}", frame=frame_idx,
                            object_id=object_id, class_id=class_id, product_id=class_name,
                            leftx=leftx, topy=topy, width=width, height=height, conf=1.0
                        )

            # Dibujo de resultados
            if DRAW_MODE == "bbox":
                frame = r.plot(boxes=True,  kpt_line=False, kpt_radius=0)
            elif DRAW_MODE == "kpts":
                frame = r.plot(boxes=False, kpt_line=True)
            else:
                frame = r.plot(boxes=True,  kpt_line=True)
            if frame is None:
                continue

            # Dibuja zonas
            draw_zones(frame, zones_for_cam, debug=False)

            # <<< NUEVO: verificar centro del bbox en zonas prohibidas y postear alerta >>>
            check_and_post_zone_alerts(
                zones_for_cam=zones_for_cam,
                detections=detections,
                frame_shape=frame.shape[:2],   # (h, w)
                publish_name=publish_name,
                api_base=API_BASE
            )

            # Publicación RTSP
            h, w = frame.shape[:2]
            if ffmpeg_proc is None:
                cmd = ffmpeg_writer_cmd(w, h, FPS, publish_url)
                ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                print(f"[WARN] FFmpeg cerró stdin para {publish_name}")
                break

            frame_idx += 1

    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                ffmpeg_proc.wait(timeout=3)
            except Exception:
                pass

def main():
    threads = []
    for idx, (src, name) in enumerate(zip(SOURCES, STREAM_NAMES)):
        t = threading.Thread(
            target=run_tracker_and_publish, args=(src, name, idx), daemon=True
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    with write_lock:
        metadata_writer.close()

if __name__ == "__main__":
    main()
