#!/usr/bin/env python3
import os, sys, cv2, time, subprocess, shlex, argparse
from ultralytics import YOLO

def have_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT).decode()
        return name in out
    except Exception:
        return False

def build_ffmpeg_cmd(W, H, FPS, name, bitrate, encoder):
    # Entrada (raw BGR por stdin) + PTS válidos
    common_in = (
        f"-f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {FPS} "
        f"-use_wallclock_as_timestamps 1 -fflags +genpts "
        f"-i - -an "
    )
    # Ritmo y GOP estables (I-frame cada 1s), sin B-frames
    gop = f"-vsync cfr -r {FPS} -g {FPS} -keyint_min {FPS} -sc_threshold 0 -bf 0 "
    # Rate control
    rate = f"-b:v {bitrate} -maxrate {bitrate} -bufsize {bitrate} "
    # Salida RTSP (publish a MediaMTX)
    out = f"-force_key_frames expr:gte(t,n_forced*1) -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/{name}"

    if encoder == "h264_v4l2m2m":
        # Jetson HW: NO usar -profile/-level (provoca error)
        enc = "-c:v h264_v4l2m2m -pix_fmt yuv420p "
        cmd = f"ffmpeg -loglevel error {common_in}{enc}{gop}{rate}{out}"
    elif encoder == "libx264":
        # CPU x264: sí podemos usar perfil baseline para compatibilidad
        enc = "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -profile:v baseline -level 3.1 "
        cmd = f"ffmpeg -loglevel error {common_in}{enc}{gop}{rate}{out}"
    else:
        # Fallback prudente a libx264
        enc = "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -profile:v baseline -level 3.1 "
        cmd = f"ffmpeg -loglevel error {common_in}{enc}{gop}{rate}{out}"
    return cmd

ap = argparse.ArgumentParser()
ap.add_argument("--in_rtsp", required=True, help="RTSP de la cámara")
ap.add_argument("--model", required=True, help=".pt o .engine")
ap.add_argument("--name", default="retail", help="nombre del stream en MediaMTX (/NAME)")
ap.add_argument("--imgw", type=int, default=640, help="ancho de inferencia (ej. 640)")
ap.add_argument("--imgh", type=int, default=640, help="alto de inferencia (ej. 640)")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--fps", type=int, default=25, help="FPS de salida")
ap.add_argument("--w", type=int, default=0, help="ancho fijo de salida (0=auto)")
ap.add_argument("--h", type=int, default=0, help="alto fijo de salida (0=auto)")
ap.add_argument("--bitrate", default="4M", help="bitrate de video (p. ej. 3M, 4M)")
ap.add_argument("--force-encoder", choices=["h264_v4l2m2m","libx264"], help="forzar encoder")
ap.add_argument("--logfps", action="store_true", help="imprimir FPS efectivos del loop")
args = ap.parse_args()

# 1) Modelo
model = YOLO(args.model, task="detect")
print("[INFO] Modelo:", ("TensorRT" if args.model.endswith(".engine") else "PyTorch"), args.model)

# 2) Cámara
cap = cv2.VideoCapture(args.in_rtsp)
if not cap.isOpened():
    print("No se pudo abrir la cámara RTSP", file=sys.stderr); sys.exit(1)

# Dimensiones y FPS
W = args.w or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
H = args.h or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
FPS = args.fps
print(f"[INFO] Input {W}x{H} -> Output {W}x{H}@{FPS}  publish rtsp://127.0.0.1:8554/{args.name}")

# 3) Selección encoder
if args.force_encoder:
    encoder = args.force_encoder
else:
    if have_encoder("h264_v4l2m2m"):
        encoder = "h264_v4l2m2m"
    elif have_encoder("libx264"):
        encoder = "libx264"
    else:
        encoder = "libx264"
print(f"[INFO] Encoder: {encoder}")

# 4) FFmpeg
ffmpeg_cmd = build_ffmpeg_cmd(W, H, FPS, args.name, args.bitrate, encoder)
proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE)

# 5) Loop con ritmo fijo
frame_period = 1.0 / FPS
next_t = time.time()
frames = 0
last_log = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        # Inferencia (imgsz = (ancho, alto))
        res = model.predict(
            frame,
            imgsz=(args.imgh, args.imgw),
            conf=args.conf,
            device=0,
            verbose=False
        )
        annotated = res[0].plot()  # BGR

        # Salida con tamaño estable
        if annotated.shape[1] != W or annotated.shape[0] != H:
            annotated = cv2.resize(annotated, (W, H), interpolation=cv2.INTER_AREA)

        # Pace exacto a FPS
        now = time.time()
        if now < next_t:
            time.sleep(max(0, next_t - now))
        next_t += frame_period

        # Envía raw BGR a FFmpeg
        try:
            proc.stdin.write(annotated.tobytes())
        except BrokenPipeError:
            print("FFmpeg terminó. Revisa logs/config.", file=sys.stderr)
            break

        if args.logfps:
            frames += 1
            if now - last_log >= 2.0:
                print(f"[DEBUG] loop FPS ~ {frames/2.0:.1f}")
                frames = 0
                last_log = now

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass

