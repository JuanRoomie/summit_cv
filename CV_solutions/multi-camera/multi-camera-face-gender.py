#!/usr/bin/env python3
import threading
import subprocess, shlex
import time
import numpy as np
import cv2
from ultralytics import YOLO
from metadata_writer import MetadataWriter

# ---- Recomendado: evita race conditions internas de OpenCV
cv2.setNumThreads(1)

# ===== Writer global + lock =====
metadata_writer = MetadataWriter(
    base_filename="metadata_faces",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración =====
MODEL_PATH = "yolov11n-face.engine"
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.20:554"
    "rtsp://admin:R00m13b0t@192.168.1.20:554"
]
STREAM_NAMES = ["gender"]   # nombre para la publicación
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.35

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(cmd)

def run_tracker_and_publish(src, publish_name, source_index):
    """
    - track() en src con YOLO face (.engine)
    - publica video anotado por FFmpeg
    - escribe metadatos en CSV
    """
    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    # Fallback de nombres por si el .engine no trae labels
    fallback_names = {0: "face"}

    try:
        for r in model.track(
            source=src,
            stream=True,
            conf=CONF,
            save=False,
            show=False,
            verbose=False,
            persist=True,     # IDs consistentes si el tracker lo soporta
            classes=None      # Face-only model: no filtrar
        ):
            frame = r.plot()
            if frame is None:
                continue

            # ===== METADATA por detección =====
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls  = boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.id is not None:
                    obj_ids = boxes.id.detach().cpu().numpy().astype(int)
                else:
                    obj_ids = np.full((len(boxes),), -1, dtype=int)

                ts = time.time()
                names = r.names if getattr(r, "names", None) else fallback_names

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    leftx  = float(x1)
                    topy   = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])

                    class_name = str(names.get(class_id, "face")).replace(" ", "_")

                    # ===== Escritura de metadatos =====
                    with write_lock:
                        metadata_writer.write(
                            time=f"{ts}",
                            source=f"{source_index}",
                            frame=frame_idx,
                            object_id=object_id,
                            class_id=class_id,
                            product_id=f"{class_name}",
                            leftx=leftx,
                            topy=topy,
                            width=width,
                            height=height,
                            conf=1.0  # la firma lo acepta; tu CSV no lo usa
                        )

            # ===== PUBLICAR POR FFMPEG =====
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
            ffmpeg_proc.wait(timeout=3)

def main():
    threads = []
    for idx, (src, name) in enumerate(zip(SOURCES, STREAM_NAMES)):
        t = threading.Thread(
            target=run_tracker_and_publish,
            args=(src, name, idx),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    with write_lock:
        metadata_writer.close()

if __name__ == "__main__":
    main()
