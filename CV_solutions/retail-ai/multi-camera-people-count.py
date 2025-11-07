#!/usr/bin/env python3
# multi-camera-people-count.py
# Conteo en vivo por cámara + total, con reset inmediato a 0 cuando no hay detecciones.

import os, time, threading, subprocess, shlex
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ============ CONFIG ============

MODEL_PATH = os.getenv("PEOPLE_MODEL_PATH", "yolo11s.engine")

# Cámaras (RTSP o archivo)
SOURCES: List[str] = [
    # "rtsp://usuario:pass@192.168.1.20:554/stream1",
    "rtsp://admin:R00m13b0t@192.168.1.20:554",
]

# Nombre de publicación (si publicas video anotado por RTSP/MediaMTX)
STREAM_NAMES: List[str] = [
    "people",
]

# Publicación de video anotado (opcional)
PUBLISH_BASE = os.getenv("PUBLISH_BASE", "rtsp://127.0.0.1:8554")
FPS = int(os.getenv("PUB_FPS", "10"))

# Detección/tracking
CONF = float(os.getenv("CONF", "0.35"))
ONLY_PERSONS = True
PERSON_CLASS_IDS = [0]  # COCO: 0=person

# Emisión de métricas en vivo
# SIN prefijos:         http://192.168.1.72:8000/retail/live/update_count
# Con root_path /wsl:   http://192.168.1.72:8000/wsl/retail/live/update_count
METRICS_POST_URL = os.getenv(
    "METRICS_POST_URL",
    "http://192.168.1.72:8000/retail/live/update_count"
)
API_TOKEN = os.getenv("API_TOKEN", "changeme")
TTL_SEC = float(os.getenv("TTL_SEC", "2.0"))          # Un ID sigue presente hasta 2 s sin verlo
POST_INTERVAL = float(os.getenv("POST_INTERVAL", "0.5"))

# ============ ESTADO GLOBAL ============

# last_seen_by_cam[idx][track_id] = last_timestamp_seen
last_seen_by_cam: Dict[int, Dict[int, float]] = defaultdict(dict)
agg_lock = threading.Lock()

emit_queue: "Queue[dict]" = Queue()


# ============ HELPERS ============

def ffmpeg_writer_cmd(w: int, h: int, fps: int, publish_url: str) -> List[str]:
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(cmd)


def update_presence_and_count(source_index: int, obj_ids: List[int], now_ts: float) -> int:
    """Actualiza presencia de IDs para una cámara y devuelve cuántos están activos."""
    with agg_lock:
        cam_dict = last_seen_by_cam[source_index]
        # refresca last_seen para los IDs vistos ahora
        for oid in obj_ids:
            cam_dict[int(oid)] = now_ts

        # limpia IDs expirados por TTL
        to_del = [tid for tid, ts in cam_dict.items() if (now_ts - ts) > TTL_SEC]
        for tid in to_del:
            del cam_dict[tid]

        return len(cam_dict)


def snapshot_counts() -> Tuple[Dict[int, int], int]:
    """Devuelve (por_camara, total)."""
    with agg_lock:
        per_cam = {int(k): len(v) for k, v in last_seen_by_cam.items()}
    total = sum(per_cam.values()) if per_cam else 0
    return per_cam, total


# ============ EMISOR NO BLOQUEANTE ============

def emitter_thread():
    last_sent = 0.0
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

    while True:
        now = time.time()
        if now - last_sent < POST_INTERVAL:
            time.sleep(0.05)
            continue

        per_cam, total = snapshot_counts()

        # 🔥 Si todas las cámaras reportan 0, fuerza total=0 (cero inmediato)
        if not per_cam or all(v == 0 for v in per_cam.values()):
            total = 0

        payload = {
            "metric": "customers_in_store",
            "total": int(total),
            "per_camera": {int(k): int(v) for k, v in per_cam.items()},
            "timestamp": now,
        }

        try:
            requests.post(METRICS_POST_URL, json=payload, headers=headers, timeout=0.5)
        except Exception as e:
            print(f"[EMIT WARN] {e}")

        last_sent = now


# ============ LOOP DE CADA CÁMARA ============

def run_tracker_and_publish(src: str, publish_name: str, source_index: int):
    """
    - Infiere + trackea con YOLO.
    - Actualiza el conteo de IDs por cámara.
    - Publica video anotado por RTSP (opcional).
    """
    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    try:
        stream = model.track(
            source=src,
            stream=True,
            conf=CONF,
            save=False,
            show=False,
            verbose=False,
            classes=PERSON_CLASS_IDS if ONLY_PERSONS else None
        )

        for r in stream:
            frame = r.plot()
            if frame is None:
                continue

            boxes = getattr(r, "boxes", None)
            now_ts = time.time()

            if boxes is not None and len(boxes) > 0:
                # === Extrae IDs del tracker ===
                if boxes.id is not None:
                    obj_ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
                else:
                    # ✅ NO metas [-1]; si no hay IDs, no contamos nada
                    obj_ids = []

                # Actualiza presencia (y limpia por TTL)
                _active = update_presence_and_count(source_index, obj_ids, now_ts)

            else:
                # ✅ SIN DETECCIONES: resetea la cámara para caer a 0 inmediato
                with agg_lock:
                    last_seen_by_cam[source_index].clear()

            # ---- Publicación por FFmpeg (opcional) ----
            if PUBLISH_BASE:
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


# ============ MAIN ============

def main():
    # Lanza emisor en segundo plano
    t_emit = threading.Thread(target=emitter_thread, daemon=True)
    t_emit.start()

    threads = []
    for idx, (src, name) in enumerate(zip(SOURCES, STREAM_NAMES)):
        t = threading.Thread(target=run_tracker_and_publish, args=(src, name, idx), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    print("[INFO] METRICS_POST_URL =", METRICS_POST_URL)
    print("[INFO] API_TOKEN set:", bool(API_TOKEN))
    main()
