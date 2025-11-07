# multi_camera_antitheft.py
# Pipeline base: YOLO -> tracking -> regla "zona obligatoria" + "línea de salida" -> evento sospechoso
# Publicación por RTSP (MediaMTX) vía FFmpeg
# Requisitos: pip install ultralytics opencv-python requests
#
# Características:
# - BBox de cada track con color aleatorio (estable por ID)
# - Si hay sospecha, el bbox se pinta ROJO y muestra "Comportamiento sospechoso"
# - Duración de la alerta configurable (ALERT_DURATION_SEC)
# - Muestra etiqueta del producto (clase + confianza)
# - Envía incidentes al backend: producto, hora (ISO-UTC), estación/cámara

import threading
import subprocess, shlex
import time
from datetime import datetime, timezone
from typing import List, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from antitheft_core import ZoneManager, SuspiciousDetector, draw_track_id

# ---------------- Config ----------------
MODEL_PATH = "bimbo.engine"           # o .pt; mismo para todas las fuentes
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.50:554",
]
STREAM_NAMES = ["robo"]           # debe corresponder al JSON (zones_config.json -> streams.products)
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS_OUT = 10
CONF = 0.0

ZONES_JSON = "zones_config.json"

# Activar envío al backend y endpoint para incidentes
POST_EVENTS = True
EVENTS_ENDPOINT = "http://192.168.1.72:8000/lp/analytics/incidents"

TRACKER = "bytetrack.yaml"

# >>> Duración visible de la alerta (bbox rojo + texto), en segundos
ALERT_DURATION_SEC = 40

# ---------------- Helpers ----------------

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """
    return shlex.split(cmd.strip())

def post_incidents(incidents: List[dict]):
    if not POST_EVENTS or not incidents:
        return
    try:
        # Backend acepta lista (batch)
        requests.post(EVENTS_ENDPOINT, json=incidents, timeout=2.0)
    except Exception as e:
        print("[WARN] incident POST failed:", e)

def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Color BGR determinista por track_id, brillante (evita tonos oscuros)."""
    rng = np.random.default_rng(seed=int(track_id) * 9973 + 12345)
    b, g, r = (int(60 + rng.integers(0, 196)),
               int(60 + rng.integers(0, 196)),
               int(60 + rng.integers(0, 196)))
    return (b, g, r)

def iso_utc_from_epoch(sec: float) -> str:
    """Convierte epoch segundos a ISO-8601 UTC con sufijo Z."""
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat().replace("+00:00", "Z")

# ---------------- Worker por stream ----------------

def run_stream(src: str, stream_name: str, zones: ZoneManager):
    print(f"[INFO] Starting stream '{stream_name}' from {src}")
    model = YOLO(MODEL_PATH)

    detector = SuspiciousDetector(stream_name=stream_name, zones=zones, decay_seconds=12.0)
    publish_url = f"{PUBLISH_BASE}/{stream_name}"
    ffmpeg_proc = None

    # Mapa: track_id -> timestamp hasta el que debe mostrarse alerta (rojo + texto)
    suspicious_until: dict[int, float] = {}

    try:
        # Ultralytics tracking en streaming
        for r in model.track(source=src, stream=True, conf=CONF, show=False, verbose=False, tracker=TRACKER):
            frame = r.orig_img  # frame original (BGR)
            if frame is None:
                continue

            now = time.time()

            # Limpieza de expirados
            expired = [tid for tid, ts in suspicious_until.items() if now >= ts]
            for tid in expired:
                suspicious_until.pop(tid, None)

            # --- Recolectar detecciones ---
            tracks_xyxy: List[Tuple[int, List[float]]] = []  # para el detector
            tracks_draw = []  # (tid, box, cls, conf) para el overlay
            labels_by_tid = {}  # tid -> (class_name, conf)

            if r.boxes is not None and r.boxes.id is not None:
                ids  = r.boxes.id.int().tolist()
                xyxy = r.boxes.xyxy.tolist()
                cls  = r.boxes.cls.tolist() if r.boxes.cls is not None else [None]*len(ids)
                conf = r.boxes.conf.tolist() if r.boxes.conf is not None else [None]*len(ids)

                for tid, box, c, cf in zip(ids, xyxy, cls, conf):
                    tid = int(tid)
                    tracks_xyxy.append((tid, box))
                    tracks_draw.append((tid, box, c, cf))
                    # nombre de clase
                    class_name = None
                    try:
                        if c is not None and hasattr(r, "names"):
                            class_name = r.names[int(c)]
                    except Exception:
                        class_name = None
                    labels_by_tid[tid] = (class_name, cf)

            # Detector de regla
            events = detector.update(frame, tracks_xyxy)

            # Renovar duración de alerta y preparar payload para backend
            if events:
                print(f"[EVENT] {len(events)} sospechosos en {stream_name}")
                incident_payload = []
                for e in events:
                    tid = int(e["track_id"])
                    suspicious_until[tid] = now + ALERT_DURATION_SEC
                    name, cf = labels_by_tid.get(tid, (None, None))
                    when_iso = iso_utc_from_epoch(e.get("when", now))
                    incident_payload.append({
                        "ts": when_iso,          # ISO UTC con Z
                        "type": "THEFT",
                        "subject": "product",
                        "label": name,
                        "sku": None,
                        "station": stream_name,
                        "class_id": None,
                        "source_id": None,
                        # >>> agrega la regla (usa la del evento, con fallback):
                        "rule": e.get("reason", "line_cross_without_zone"),
                    })
                post_incidents(incident_payload)

            # Dibujo de tracks (bbox con color; rojo si en ventana de alerta + texto)
            for tid, box, c, cf in tracks_draw:
                is_suspicious_active = tid in suspicious_until and now < suspicious_until[tid]
                color = (0, 0, 255) if is_suspicious_active else color_for_id(tid)

                # Bbox + ID
                draw_track_id(frame, box, tid, color=color)

                # Etiqueta (clase + conf)
                x1, y1, x2, y2 = [int(v) for v in box]
                class_name = None
                try:
                    if c is not None and hasattr(r, "names"):
                        class_name = r.names[int(c)]
                except Exception:
                    class_name = None

                label_txt = None
                if class_name is not None and cf is not None:
                    label_txt = f"{class_name} {cf:.2f}"
                elif class_name is not None:
                    label_txt = f"{class_name}"
                elif cf is not None:
                    label_txt = f"{cf:.2f}"

                if label_txt:
                    (tw, th), bl = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label_txt, (x1 + 3, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)

                # Texto de alerta si corresponde
                if is_suspicious_active:
                    cv2.putText(
                        frame,
                        "Comportamiento sospechoso",
                        (x1, max(0, y1 - 12 - 18)),  # un poco arriba de la etiqueta
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        lineType=cv2.LINE_AA
                    )

            # (Opcional) Overlay minimalista con nombre del stream
            cv2.putText(frame, stream_name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 240, 80), 2)

            # Publicar por RTSP con FFmpeg
            h, w = frame.shape[:2]
            if ffmpeg_proc is None:
                ffmpeg_proc = subprocess.Popen(ffmpeg_writer_cmd(w, h, FPS_OUT, publish_url), stdin=subprocess.PIPE)
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                print(f"[WARN] FFmpeg closed for {stream_name}")
                break

    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait(timeout=3)
        print(f"[INFO] Stream '{stream_name}' finished")

# ---------------- Main ----------------

def main():
    zones = ZoneManager.from_json(ZONES_JSON)

    # Si tienes más fuentes que nombres, duplica el último nombre; o asértalo aquí
    if len(STREAM_NAMES) < len(SOURCES):
        last = STREAM_NAMES[-1]
        STREAM_NAMES.extend([f"{last}_{i}" for i in range(len(STREAM_NAMES), len(SOURCES))])

    threads = []
    for src, name in zip(SOURCES, STREAM_NAMES):
        t = threading.Thread(target=run_stream, args=(src, name, zones), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
