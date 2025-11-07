import threading
import subprocess, shlex
import time
import numpy as np
import cv2
from ultralytics import YOLO
from metadata_writer import MetadataWriter
from analytics_engine import AnalyticsEngine

# Crea el motor
analytics = AnalyticsEngine()

# Define ROIs normalizados para el stream "retail"
analytics.set_rois("retail", [
    {"name":"zona_entrada", "type":"rect", "xywh":[0.05, 0.05, 0.30, 0.40]},      # x,y,w,h normalizados
    {"name":"cajas",        "type":"poly", "points":[(0.65,0.60),(0.95,0.60),(0.95,0.95),(0.70,0.95)]}
])

# ===== Writer global + lock =====
metadata_writer = MetadataWriter(
    base_filename="metadata_personas",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración =====
MODEL_PATH = "yolo11s-pose.engine"     # <- corregido
SOURCES = [
    #"cremramirez_rostros.mp4"
    "rtsp://admin:R00m13b0t@192.168.1.20:554"
]
STREAM_NAMES = ["retail"]
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.60

# <<< Bandera de dibujo: 'bbox' | 'kpts' | 'both' >>>
DRAW_MODE = "bbox"

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(cmd)

def run_tracker_and_publish(src, publish_name, source_index):
    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    try:
        for r in model.track(source=src, stream=True, conf=CONF, save=False, show=False, verbose=False):
            # ====== METADATA (siempre BBX) ======
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls  = boxes.cls.detach().cpu().numpy().astype(int)
                if boxes.id is not None:
                    obj_ids = boxes.id.detach().cpu().numpy().astype(int)
                else:
                    obj_ids = np.full((len(boxes),), -1, dtype=int)

                dets = []
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    dets.append({
                        "id": int(obj_ids[i]),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "class_id": int(cls[i]),
                        "class_name": str(r.names.get(int(cls[i]), "unknown"))
                    })

                # Actualiza analíticas
                result = analytics.update(
                    stream_name=publish_name,       # en tu código es "retail"
                    frame_idx=frame_idx,
                    frame_shape=frame.shape,
                    detections=dets
                )

                # (opcional) dibujar ROIs encima:
                frame = analytics.draw_overlays(publish_name, frame)

                # (opcional) imprimir métricas por ROI cada N frames
                if frame_idx % 30 == 0:
                    print(f"[{publish_name}] frame {frame_idx} ->", result["per_roi"])

                ts = time.time()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    leftx  = float(x1)
                    topy   = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])
                    class_name = str(r.names.get(class_id, "unknown")).replace(" ", "_")

                    with write_lock:
                        metadata_writer.write(
                            time=f"{ts}",
                            source=f"{source_index}",
                            frame=frame_idx,
                            object_id=object_id,
                            class_id=class_id,
                            product_id=class_name,  # ClassName
                            leftx=leftx,
                            topy=topy,
                            width=width,
                            height=height,
                            conf=1.0  # no se usa en el archivo
                        )

            # ====== DIBUJO SEGÚN BANDERA ======
            if DRAW_MODE == "bbox":
                frame = r.plot(boxes=True,  kpt_line=False, kpt_radius=0)
            elif DRAW_MODE == "kpts":
                frame = r.plot(boxes=False, kpt_line=True)
            else:  # "both"
                frame = r.plot(boxes=True,  kpt_line=True)

            if frame is None:
                continue

            # ====== PUBLICAR POR FFMPEG ======
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
