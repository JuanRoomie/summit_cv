import threading
import subprocess, shlex
import cv2
from ultralytics import YOLO
import os
import time

# ==== NUEVO: import del módulo de persistencia ====
from damage_persistence import DamagePersistenceEmitter, parse_ultralytics_result

# ===== Configuración =====
MODEL_PATH = "boxes.engine"
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.50:554"
]
STREAM_NAMES = ["damaged"]
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.50

# ==== NUEVO: Backend y parámetros de persistencia (editable por env) ====
API_BASE = os.getenv("API_BASE", "http://192.168.1.72:8000")    # tu FastAPI
API_TOKEN = os.getenv("API_TOKEN", "")                       # opcional
MIN_DURATION_SEC = float(os.getenv("DAMAGE_MIN_DURATION_SEC", "3.0"))
DEBOUNCE_SEC = float(os.getenv("DAMAGE_DEBOUNCE_SEC", "5.0"))

# Clases del modelo (exacto como salen de YOLO)
DAMAGE_CLASSES = {"torn", "wrinkle"}
IGNORE_CLASS = "no defect"
PRODUCT_NAME = os.getenv("DAMAGE_PRODUCT_NAME", "Caja de empaquetado")

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    ffmpeg_cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(ffmpeg_cmd)

def run_tracker_and_publish(src, publish_name):
    model = YOLO(MODEL_PATH)

    # ==== NUEVO: emisor por hilo/cámara ====
    emitter = DamagePersistenceEmitter(
        api_base=API_BASE,
        api_token=API_TOKEN or None,
        product_name=PRODUCT_NAME,
        min_duration_sec=MIN_DURATION_SEC,
        conf_threshold=CONF,
        damage_classes=DAMAGE_CLASSES,
        ignore_class=IGNORE_CLASS,
        debounce_sec=DEBOUNCE_SEC,
    )

    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"

    try:
        for r in model.track(source=src, stream=True, conf=CONF, save=False, show=False, verbose=False):
            # ==== NUEVO: persistencia de daño ====
            dets = parse_ultralytics_result(r, names=getattr(r, "names", None))
            emitter.process(detections=dets, station=publish_name)

            # ===== Tu flujo de video =====
            frame = r.plot()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if ffmpeg_proc is None:
                cmd = ffmpeg_writer_cmd(w, h, FPS, publish_url)
                ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                print(f"[WARN] FFmpeg cerró stdin para {publish_name}")
                break

    finally:
        if ffmpeg_proc is not None:
            try: ffmpeg_proc.stdin.close()
            except Exception: pass
            try: ffmpeg_proc.wait(timeout=3)
            except Exception: pass
        try: emitter.stop()
        except Exception: pass

def main():
    threads = []
    for src, name in zip(SOURCES, STREAM_NAMES):
        t = threading.Thread(target=run_tracker_and_publish, args=(src, name), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
