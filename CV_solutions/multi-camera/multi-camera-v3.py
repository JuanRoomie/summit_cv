import threading
import subprocess, shlex
import time
import numpy as np
import cv2
from ultralytics import YOLO
from metadata_writer import MetadataWriter

# ===== Writer global + lock =====
metadata_writer = MetadataWriter(
    base_filename="metadata_productos",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración =====
MODEL_PATH = "yolo11s.engine"
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.20:554"
]
STREAM_NAMES = ["people"]   # solo para la publicación
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.35

ONLY_PERSONS = True           # <— bandera para detectar solo personas (COCO id=0)
PERSON_CLASS_IDS = [0]        # COCO: 0 = person

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    import shlex
    return shlex.split(cmd)

def run_tracker_and_publish(src, publish_name, source_index):
    """
    - track() en src
    - publica video anotado por FFmpeg
    - escribe metadatos en un archivo único, con 'Source' = source_index
    """
    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    try:
        for r in model.track(source=src, stream=True, conf=CONF, save=False, show=False, verbose=False, classes=PERSON_CLASS_IDS if ONLY_PERSONS else None):
            frame = r.plot()
            if frame is None:
                continue

            # ===== METADATA por detección =====
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                # a numpy, independiente si está en GPU
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls  = boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.id is not None:
                    obj_ids = boxes.id.detach().cpu().numpy().astype(int)
                else:
                    obj_ids = np.full((len(boxes),), -1, dtype=int)

                ts = time.time()  # epoch con decimales
                #product_id = None  # placeholder; integras tu lógica cuando la tengas

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    leftx  = float(x1)
                    topy   = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])

                    class_name = str(r.names.get(class_id, "unknown")).replace(" ", "_")

                    # Escritura segura entre hilos
                    with write_lock:
                        metadata_writer.write(
                            time=f"{ts}",                 # ej: 1740879278.26535
                            source=f"{source_index}",     # ej: 0, 1, ...
                            frame=frame_idx,
                            object_id=object_id,
                            class_id=class_id,
                            product_id=class_name,
                            leftx=leftx,
                            topy=topy,
                            width=width,
                            height=height,
                            conf=1.0  # este arg existe en la firma; no se usa en el archivo
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
            args=(src, name, idx),  # <— pasamos el índice de la fuente
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
