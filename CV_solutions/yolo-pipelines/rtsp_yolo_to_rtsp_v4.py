#!/usr/bin/env python3
import os
import sys
import cv2
import time
import threading
from queue import Queue

# --- GStreamer / RTSP Server ---
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GstRtspServer

# --- Ultralytics ---
from ultralytics import YOLO

"""
Uso:
  python3 rtsp_yolo_lowlat.py \
    --in_rtsp "rtsp://usuario:pass@CAM_IP:554/stream" \
    --model /ruta/a/modelo.pt \
    --out_name stream

Luego, para ver:
  ffplay -fflags nobuffer -flags low_delay -rtsp_transport tcp rtsp://127.0.0.1:8554/stream
"""

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--in_rtsp", required=True, help="URL RTSP de la cámara IP")
ap.add_argument("--model", required=True, help="Ruta a modelo YOLO (.pt o .engine)")
ap.add_argument("--out_name", default="stream", help="Nombre del mount RTSP (rtsp://host:8554/<out_name>)")
ap.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
ap.add_argument("--conf", type=float, default=0.25, help="Confianza mínima")
args = ap.parse_args()

# --- 1) Carga modelo YOLO (acepta .pt o .engine) ---
# En Jetson puedes usar FP16 (half=True) si es .pt; si es .engine ya viene optimizado
model = YOLO(args.model)

# --- 2) GStreamer init ---
Gst.init(None)
GObject.threads_init()

# Cola de frames BGR -> la usa el servidor RTSP (appsrc)
frame_queue = Queue(maxsize=10)

# --- 3) Lector RTSP (entrada) con GStreamer a OpenCV (appsink) ---
# Pipeline con decodificación por hardware y latencia mínima
gst_in = (
    f'rtspsrc location="{args.in_rtsp}" latency=0 protocols=tcp ! '
    'rtph264depay ! h264parse ! nvv4l2decoder ! '
    'nvvidconv ! video/x-raw,format=BGRx ! '
    'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
)

cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("ERROR: No se pudo abrir el RTSP de entrada.", file=sys.stderr)
    sys.exit(1)

# --- 4) RTSP Server con appsrc -> nvv4l2h264enc -> rtph264pay ---
class Factory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, width, height, fps):
        super(Factory, self).__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.number_frames = 0
        self.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        self.launch_string = (
            'appsrc name=src is-live=true block=true format=time do-timestamp=true '
            f'caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! '
            'videoconvert ! video/x-raw,format=NV12 ! '
            # Encoder HW Jetson con baja latencia
            'nvv4l2h264enc preset-level=1 insert-sps-pps=true iframeinterval=30 control-rate=1 bitrate=4000000 '
            'tune=zerolatency maxperf-enable=1 EnableTwopassCBR=0 insert-vui=1 ! '
            'h264parse config-interval=1 ! rtph264pay name=pay0 pt=96 config-interval=1'
        )

    def do_create_element(self, _url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name("src")
        appsrc.connect("need-data", self.on_need_data)

    def on_need_data(self, src, length):
        # Saca frames ya inferidos de la cola
        try:
            frame = frame_queue.get(timeout=1.0)  # BGR np.ndarray
        except:
            return
        # Empaqueta a GstBuffer
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = timestamp
        self.number_frames += 1
        src.emit("push-buffer", buf)

# Lee primer frame para dimensiones
ok, test_frame = cap.read()
if not ok:
    print("ERROR: No se pueden leer frames de la cámara.", file=sys.stderr)
    sys.exit(1)
H, W = test_frame.shape[:2]
FPS = 30  # ajustar si conoces el FPS real; 25–30 va bien en la mayoría

# --- 5) Servidor RTSP ---
server = GstRtspServer.RTSPServer.new()
server.props.service = "8554"
mounts = server.get_mount_points()
factory = Factory(W, H, FPS)
factory.set_shared(True)
mounts.add_factory(f"/{args.out_name}", factory)
server.attach(None)
print(f"[OK] Publicando RTSP en rtsp://<host>:8554/{args.out_name}")

# --- 6) Hilo de inferencia: RTSP in -> YOLO -> draw -> queue ---
def infer_loop():
    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        # Inferencia (Ultralytics pinta si show=False; usamos pred y dibujamos rápido)
        results = model.predict(
            source=frame,  # ndarray BGR
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            device=0  # GPU
        )
        # Tomamos la primera predicción
        pred = results[0]
        # Dibujado rápido en BGR
        annotated = pred.plot()  # devuelve BGR con cajas

        # empuja a la cola (descarta si llena para mantener latencia baja)
        try:
            if frame_queue.full():
                _ = frame_queue.get_nowait()
            frame_queue.put_nowait(annotated)
        except:
            pass

        # (opcional) medir FPS inferencia
        now = time.time()
        if now - last_t > 2:
            last_t = now
            # print("tick")  # descomenta si quieres logs

infer_thread = threading.Thread(target=infer_loop, daemon=True)
infer_thread.start()

# --- 7) Main loop GObject (RTSP server) ---
loop = GObject.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
