import threading
import subprocess, shlex
import cv2
from ultralytics import YOLO
from metadata_writer import MetadataWriter
import time

metadata_writer = MetadataWriter(
    base_filename="metadata_productos",
    header="Time,Source,Frame,ObjectID,ClassID,ProductID,Leftx,Topy,Width,Height,Conf\n",
    output_dir="metadata"
)

# ===== Configuración =====
MODEL_PATH = "bimbo.engine"          # mismo modelo para todas las fuentes
SOURCES = [
    "rtsp://admin:R00m13b0t@192.168.1.50:554"
]
STREAM_NAMES = ["products"]        # nombre único para publicar cada salida
PUBLISH_BASE = "rtsp://127.0.0.1:8554" # tu servidor RTSP (MediaMTX)
FPS = 10                               # fps de salida (ajústalo a tu caso)
CONF = 0.1                          # umbral de confianza

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    """
    Construye el comando FFmpeg que recibe rawvideo (BGR24) por stdin y publica RTSP.
    """
    ffmpeg_cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(ffmpeg_cmd)

def run_tracker_and_publish(src, publish_name):
    """
    1) Corre YOLO.track(stream=True) sobre 'src'
    2) Pinta resultados r.plot() -> BGR
    3) Publica los frames por FFmpeg a rtsp://.../<publish_name>
    """
    model = YOLO(MODEL_PATH)

    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"

    try:
        # stream=True entrega un generador de resultados (uno por frame procesado)
        for r in model.track(source=src, stream=True, conf=CONF, save=False, show=False, verbose=False):
            # Frame con anotaciones (BGR)
            frame = r.plot()  # numpy array (h, w, 3) en BGR
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Lanzamos FFmpeg hasta tener el primer frame (para conocer WxH)
            if ffmpeg_proc is None:
                cmd = ffmpeg_writer_cmd(w, h, FPS, publish_url)
                ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            # Escribimos el frame en bruto (BGR24) por stdin hacia FFmpeg
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                # Si FFmpeg se cayó por alguna razón, salimos del bucle
                print(f"[WARN] FFmpeg cerró stdin para {publish_name}")
                break

    finally:
        # Cierre ordenado del proceso FFmpeg
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait(timeout=3)

def main():
    threads = []
    for src, name in zip(SOURCES, STREAM_NAMES):
        t = threading.Thread(target=run_tracker_and_publish, args=(src, name), daemon=True)
        t.start()
        threads.append(t)

    # Espera a que terminen (si los streams finalizan)
    for t in threads:
        t.join()

    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
