#!/usr/bin/env python3
import os, sys, cv2, time, subprocess, shlex, argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--in_rtsp", required=True, help="RTSP de la cámara")
ap.add_argument("--model", required=True, help=".pt o .engine")
ap.add_argument("--name", default="stream", help="nombre del stream en MediaMTX")
ap.add_argument("--imgw", type=int, default=640, help="ancho de inferencia")
ap.add_argument("--imgh", type=int, default=640, help="alto de inferencia")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--w", type=int, default=0, help="force width (0=auto)")
ap.add_argument("--h", type=int, default=0, help="force height (0=auto)")
args = ap.parse_args()

# 1) Modelo
model_path = args.model
model = YOLO(model_path, task="detect")

if model_path.endswith(".engine"):
    print("[INFO] Cargando modelo TensoRT:", model_path)
else:
    print("[INFO] Cargando modelo PyTorch:", model_path)

# 2) Cámara
cap = cv2.VideoCapture(args.in_rtsp)
if not cap.isOpened():
    print("No se pudo abrir la cámara RTSP", file=sys.stderr); sys.exit(1)

# Detecta dimensiones si no se forzaron
W = args.w or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
H = args.h or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
FPS = args.fps

print(f"[INFO] Input {W}x{H}@{FPS} -> publish rtsp://127.0.0.1:8554/{args.name}")

# 3) Proceso FFmpeg: lee raw BGR por stdin y publica por RTSP
# CPU-friendly y baja latencia (ajustable):
ffmpeg_cmd = f"""
ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {FPS} -i - -an \
-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {FPS} -b:v 4M \
-f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/{args.name}
""".strip()

# --- Alternativa con encoder HW (si tu build de ffmpeg lo soporta):
# -c:v h264_v4l2m2m -b:v 4M -maxrate 4M -bufsize 4M -g {FPS} -bf 0 -pix_fmt yuv420p

proc = subprocess.Popen(
    shlex.split(ffmpeg_cmd),
    stdin=subprocess.PIPE
)

last = time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        # Inferencia YOLO
        res = model.predict(frame, imgsz=(args.imgh,args.imgw), conf=args.conf, device=0, verbose=False)
        annotated = res[0].plot()  # BGR

        # Asegura tamaño (algunos RTSP cambian resolución)
        if annotated.shape[1] != W or annotated.shape[0] != H:
            annotated = cv2.resize(annotated, (W, H), interpolation=cv2.INTER_AREA)

        # Escribe al stdin de ffmpeg (rawvideo BGR)
        try:
            proc.stdin.write(annotated.tobytes())
        except BrokenPipeError:
            print("FFmpeg terminó. Revisa logs/config.", file=sys.stderr)
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.wait(timeout=2)
