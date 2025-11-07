#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP in -> YOLO (.pt o .engine) -> RTSP out (low-latency)
Base de pipeline para proyectos retail/CPG.

Requisitos:
  pip install ultralytics opencv-python
  FFmpeg instalado
  MediaMTX en ejecución (rtsp://host:8554)

Autor: tú :)
"""

import os, sys, time, json, shlex, signal, subprocess
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import cv2
import shlex, subprocess
import numpy as np
from ultralytics import YOLO

# =========================
#  A) CONFIGURACIÓN (CLI)
# =========================
def build_cli():
    p = argparse.ArgumentParser(description="RTSP -> YOLO -> RTSP (low latency)")
    # Entrada
    p.add_argument("--rtsp_in", required=True, help="rtsp://user:pass@ip:554/stream")
    p.add_argument("--reconnect_sec", type=float, default=2.0, help="espera entre reconexiones")
    p.add_argument("--drop_old_frames", action="store_true", help="intenta minimizar buffer en captura")

    # Modelo
    p.add_argument("--model", required=True, help="ruta a .pt o .engine")
    p.add_argument("--device", default="", help="'' (auto), 'cpu', '0', '1'...")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou",  type=float, default=0.7)
    p.add_argument("--half", action="store_true")
    p.add_argument("--classes", nargs="+", type=int, help="filtra por clases: --classes 0 2 3")

    # Overlay
    p.add_argument("--overlay", action="store_true", default=True, help="dibuja cajas/labels")
    p.add_argument("--no-overlay", dest="overlay", action="store_false")
    p.add_argument("--show_labels", action="store_true", default=True)
    p.add_argument("--show_conf",   action="store_true", default=True)
    p.add_argument("--line_width",  type=int, default=2)

    # Salida RTSP
    p.add_argument("--rtsp_out", required=True, help="rtsp://host:8554/nombre_stream (MediaMTX)")
    p.add_argument("--fps_out", type=int, default=25)
    p.add_argument("--bitrate", default="4M")
    p.add_argument("--gop", type=int, default=25, help="keyint (menor = menos latencia, más bitrate)")
    p.add_argument("--encoder", choices=["auto","nvenc","jetson","cpu"], default="auto",
                   help="elige NVENC/Jetson/CPU; 'auto' detecta por bandera")
    p.add_argument("--use_nvenc", action="store_true", help="sugerir NVENC si encoder=auto")
    p.add_argument("--jetson", action="store_true", help="sugerir Jetson si encoder=auto")

    # Miscelánea
    p.add_argument("--print_fps", action="store_true", help="muestra FPS procesados")
    return p

# ==================================
#  B) CAPTURA RTSP con reconexión
# ==================================
class ReconnectingRTSP:
    """Captura RTSP con reconexión y buffer mínimo."""
    def __init__(self, url: str, reconnect_sec: float = 2.0, drop_old_frames: bool = True):
        self.url = url
        self.reconnect_sec = reconnect_sec
        self.drop_old_frames = drop_old_frames
        self.cap = None
        self.open()

    def open(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        # TIPOS de apertura:
        # 1) Directo: cv2.VideoCapture(url)
        # 2) Con opciones ffmpeg (si tu OpenCV viene con FFmpeg): setea buffersize bajo
        self.cap = cv2.VideoCapture(self.url)
        # Minimiza el buffer interno (no todas las builds lo soportan)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            time.sleep(self.reconnect_sec)
            self.open()
            return None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            # Reintentar
            time.sleep(self.reconnect_sec)
            self.open()
            return None

        # Si hay atraso, vacía el buffer leyendo frames rápidos
        if self.drop_old_frames:
            dropped = 0
            while True:
                ok2, frame2 = self.cap.read()
                if not ok2 or frame2 is None:
                    break
                # Si lees demasiado rápido, te comes frames; limitamos drops
                dropped += 1
                if dropped > 3:
                    frame = frame2
                    break

        return frame

    def release(self):
        if self.cap:
            self.cap.release()

# ===============================
#  C) INFERENCIA con YOLO
# ===============================
@dataclass
class YOLOConfig:
    model_path: str
    device: str = ""
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    half: bool = False
    classes: Optional[list] = None
    overlay: bool = True
    show_labels: bool = True
    show_conf: bool = True
    line_width: int = 2

class YOLOInfer:
    def __init__(self, cfg: YOLOConfig):
        self.cfg = cfg
        # Carga .pt o .engine (TensorRT)
        self.model = YOLO(cfg.model_path)
        if cfg.device:
            self.model.to(cfg.device)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Devuelve (frame_out, meta). Si overlay=False, frame_out == frame original."""
        r = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            half=self.cfg.half,
            device=self.cfg.device if self.cfg.device else None,
            classes=self.cfg.classes,
            verbose=False
        )[0]

        # Extrae metadatos por si quieres usarlos (JSON a futuro)
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cls = int(b.cls[0].item()) if b.cls is not None else -1
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                dets.append({"xyxy":[x1,y1,x2,y2],"cls":cls,"conf":conf})

        if self.cfg.overlay:
            out = r.plot(
                labels=self.cfg.show_labels,
                conf=self.cfg.show_conf,
                line_width=self.cfg.line_width
            )
        else:
            out = frame  # sin overlay para latencia mínima

        return out, {"detections": dets}

# =========================================
#  D) PUBLICADOR RTSP (FFmpeg -> MediaMTX)
# =========================================
class RTSPPublisher:
    """
    Publica frames por RTSP usando GStreamer en un proceso externo (gst-launch-1.0).
    No depende de OpenCV con GStreamer. Empuja frames BGR por stdin.
    """
    def __init__(self, rtsp_url, size, fps=25, bitrate="4M",
                 encoder="auto", suggest_nvenc=False, suggest_jetson=True, gop=25):
        self.w, self.h = size
        self.fps = fps
        self.gop = gop
        self.proc = None

        # kbps desde "4M", "3M", etc.
        if bitrate.lower().endswith("m"):
            kbps = int(float(bitrate[:-1]) * 1000)
        elif bitrate.lower().endswith("k"):
            kbps = int(float(bitrate[:-1]))
        else:
            # número “crudo” en bps
            kbps = max(1000, int(bitrate) // 1000)

        use_jetson = (encoder == "jetson") or suggest_jetson
        # Encoder HW (Jetson) si está disponible; si no, x264 (CPU)
        if use_jetson:
            enc = (
                f"nvv4l2h264enc iframeinterval={gop} idrinterval={gop} control-rate=1 "
                f"bitrate={kbps} preset-level=1 insert-sps-pps=true"
            )
        else:
            enc = (
                f"x264enc tune=zerolatency key-int-max={gop} bframes=0 "
                f"bitrate={kbps} speed-preset=veryfast"
            )

        # Usamos fdsrc (stdin = fd 0) + videoparse para declarar el formato BGR continuo
        # Luego convertimos a I420, codificamos a H264 y empujamos por RTSP.
        pipeline = (
            "gst-launch-1.0 -q "
            "fdsrc fd=0 ! "
            f"videoparse width={self.w} height={self.h} framerate={self.fps}/1 format=bgr ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"{enc} ! h264parse config-interval=1 ! "
            f"rtspclientsink location={rtsp_url} protocols=tcp"
        )

        self.proc = subprocess.Popen(
            shlex.split(pipeline),
            stdin=subprocess.PIPE
        )
        if self.proc.poll() is not None:
            raise RuntimeError("No se pudo lanzar gst-launch-1.0. Revisa que GStreamer y plugins estén instalados.")

    def write(self, frame_bgr):
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            raise RuntimeError("GStreamer se cerró (Broken pipe). Verifica el pipeline y la URL RTSP del servidor.")

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

# ======================
#  E) BUCLE PRINCIPAL
# ======================
def main():
    args = build_cli().parse_args()

    # ----- Entrada RTSP -----
    src = ReconnectingRTSP(args.rtsp_in, args.reconnect_sec, args.drop_old_frames)

    # ----- YOLO -----
    yolo_cfg = YOLOConfig(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        classes=args.classes,
        overlay=args.overlay,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        line_width=args.line_width,
    )
    infer = YOLOInfer(yolo_cfg)

    # ----- Warm-up: toma 1 frame para tamaños -----
    frame0 = None
    while frame0 is None:
        frame0 = src.read()
    h, w = frame0.shape[:2]

    # ----- Publicador RTSP (salida) -----
    pub = RTSPPublisher(
        rtsp_url=args.rtsp_out,
        size=(w, h),
        fps=args.fps_out,
        bitrate=args.bitrate,
        encoder=args.encoder,
        suggest_nvenc=args.use_nvenc,
        suggest_jetson=args.jetson,
        gop=args.gop
    )

    # Control de salida limpia
    def handle_sigint(sig, frame):
        pub.close()
        src.release()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # ----- Bucle de procesamiento -----
    last_t = time.time()
    frames = 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                continue

            # Inferencia
            out_frame, meta = infer.process(frame)

            # Publica al RTSP out
            pub.write(out_frame)

            # Métrica (opcional)
            if args.print_fps:
                frames += 1
                now = time.time()
                if now - last_t >= 1.0:
                    print(f"FPS ~ {frames/(now-last_t):.1f}")
                    frames = 0
                    last_t = now

    finally:
        pub.close()
        src.release()

if __name__ == "__main__":
    main()
