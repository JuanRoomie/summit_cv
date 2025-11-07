#!/usr/bin/env python3
# tsp_yolo_to_rtsp.py  (Jetson-ready)
# YOLO (Ultralytics) -> tracking -> overlay -> RTSP (MediaMTX) + Self-Checkout ROI events
# Requiere: pip install ultralytics opencv-python requests
#            y el módulo zone_events.py (ZoneEventEmitterSC)
# Nota: pensado para Jetson (Orin/Nano) usando encoder HW v4l2m2m vía FFmpeg.

import os, sys, cv2, time, shlex, argparse, subprocess, math, json
from typing import Dict, Tuple, List
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
if not hasattr(DEFAULT_CFG, "fuse_score"):
    setattr(DEFAULT_CFG, "fuse_score", False)

# módulo de eventos Self-Checkout (lo que escribimos antes)
from zone_events import ZoneEvents
from sc_backend_client import SCClient, SCConfig, _slug

GENERIC_PRICE = 1000

def resolver_por_label(label: str):
    # product_code = slug(label), name = label, price = genérico
    return _slug(label), (label or "producto"), GENERIC_PRICE


def parse_roi_norm(s: str):
    # "x1,y1;x2,y2;..."
    return [(float(x), float(y)) for x,y in (pair.split(",") for pair in s.split(";"))]

def draw_roi(frame, poly_px, color=(0,255,255), alpha=0.25, thickness=2, name="ROI"):
    if not poly_px or len(poly_px) < 3: return frame
    overlay = frame.copy()
    pts = np.array(poly_px, dtype=np.int32).reshape((-1,1,2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    x0,y0 = pts[0,0]; cv2.putText(frame, name, (x0+6, max(0,y0-6)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def parse_roi_norm(s: str) -> List[Tuple[float, float]]:
    """
    Convierte "x1,y1;x2,y2;..." en lista [(x,y), ...] con valores 0..1
    """
    pts = []
    for pair in s.split(";"):
        x, y = pair.split(",")
        pts.append((float(x), float(y)))
    return pts


def parse_map_inline(s: str) -> Dict:
    """
    Convierte "label1:SKU1,label2:SKU2,3:SKUforClass3" a dict
    - keys pueden ser label (str) o class_id (int)
    """
    if not s:
        return {}
    out = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k.isdigit():
            out[int(k)] = v
        else:
            out[k] = v
            out[k.lower()] = v  # comodidad
    return out


def build_ffmpeg_cmd(width: int, height: int, fps: int, name: str, use_tcp: bool = True) -> str:
    # H.264 exige dimensiones pares
    width  = width  - (width  % 2)
    height = height - (height % 2)
    transport = "tcp" if use_tcp else "udp"
    return f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - -an \
      -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
      -f rtsp -rtsp_transport {transport} rtsp://127.0.0.1:8554/{name}
    """.strip()


def main():
    ap = argparse.ArgumentParser("YOLO -> RTSP + Self-Checkout ROI (Jetson)")
    # Entrada / modelo
    ap.add_argument("--in_rtsp", required=True, help="RTSP de la cámara (input)")
    ap.add_argument("--model", required=True, help="Ruta .pt o .engine")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgw", type=int, default=640)
    ap.add_argument("--imgh", type=int, default=640)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--w", type=int, default=0, help="force width (0=auto de la cámara)")
    ap.add_argument("--h", type=int, default=0, help="force height (0=auto de la cámara)")

    # Salida RTSP (MediaMTX)
    ap.add_argument("--name", default="retail", help="Nombre del stream RTSP en MediaMTX")
    ap.add_argument("--rtsp-udp", action="store_true", help="Usar RTSP/UDP (por defecto TCP)")

    # ROI + eventos SC
    ap.add_argument("--roi", type=str, required=True, help="Polígono normalizado 'x1,y1;x2,y2;...'")
    ap.add_argument("--roi-name", type=str, default="zona_productos")
    ap.add_argument("--min-inside", type=int, default=2)
    ap.add_argument("--send-exit", action="store_true")
    ap.add_argument("--exit-grace", type=int, default=12)
    ap.add_argument("--exit-action", type=str, default="decrement", choices=["decrement", "remove"])

    # Backend SC + mapeo de productos
    #http://ec2-44-223-62-175.compute-1.amazonaws.com:9094
    #
    ap.add_argument("--api-base", type=str, default=os.getenv("API_BASE", "http://ec2-44-223-62-175.compute-1.amazonaws.com:9094"))
    ap.add_argument("--api-key", type=str, default=os.getenv("API_KEY", ""))
    ap.add_argument("--station", type=str, default=os.getenv("STATION_ID", "caja-01"))
    ap.add_argument("--session", type=str, default=os.getenv("SESSION_ID", None))
    ap.add_argument("--session-refresh-frames", type=int, default=30)
    ap.add_argument("--map", type=str, default="", help='Inline mapping "coke:SKU1,7:SKUclass7,chips:SKU2"')
    ap.add_argument("--map-json", type=str, default="", help="Ruta a JSON con mapping label/cls -> product_code")
    ap.add_argument("--price-cents", type=int, default=None, help="Precio por defecto (opcional)")

    args = ap.parse_args()

    # 1) Modelo
    model_path = args.model
    model = YOLO(model_path, task="detect")
    print("[INFO] Cargando modelo:", ("TensorRT" if model_path.endswith(".engine") else "PyTorch"), model_path)

    roi_norm = parse_roi_norm(args.roi)   # ROI normalizado 0..1
    ze = ZoneEvents(
        roi_polygon_norm=roi_norm,
        fps=10,                  # muy importante para que los segundos funcionen
        enter_dwell_sec= 1,     # ← 1 segundo para confirmar entrada
        exit_absence_sec= 1     # ← 2 segundos de ausencia para salida
    )
    

    # 2) Cámara
    cap = cv2.VideoCapture(args.in_rtsp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS reportado por cámara:", fps)
    if not cap.isOpened():
        print("No se pudo abrir la cámara RTSP", file=sys.stderr)
        sys.exit(1)

    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    W = args.w or W_in
    H = args.h or H_in
    FPS = args.fps
    print(f"[INFO] Input {W_in}x{H_in} -> Proc/Out {W}x{H}@{FPS}  rtsp://127.0.0.1:8554/{args.name}")

    # 3) ROI + eventos Self-Checkout
    roi_norm = parse_roi_norm(args.roi)

    # mapeo productos
    prod_map = parse_map_inline(args.map)
    if args.map_json and os.path.isfile(args.map_json):
        try:
            with open(args.map_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            # merge con prioridad al JSON
            prod_map.update(j)
        except Exception as e:
            print(f"[WARN] No se pudo leer map-json: {e}")

    # 4) FFmpeg (Jetson HW encoder)
    ffmpeg_cmd = build_ffmpeg_cmd(W, H, FPS, args.name, use_tcp=(not args.rtsp_udp))
    proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE)


    sc = SCClient(SCConfig(
        api_base=args.api_base,
        api_key=getattr(args, "api_key", ""),
        station_id=args.station,
        session_id=getattr(args, "session", None),   # None => se resuelve por estación
        default_price_cents=GENERIC_PRICE,
        exit_action=args.exit_action,                 # "decrement" o "remove"
        products_map=resolver_por_label,              # <<< usa class name
        slugify_fallback=False,                       # no hace falta, ya slugificamos arriba
    ))

    # 5) Bucle principal
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame_idx += 1

            # Reescalar si es necesario
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            # Tracking con IDs persistentes (ByteTrack)
            res = model.track(
                frame,
                imgsz=(args.imgh, args.imgw),
                conf=args.conf,           # prueba 0.20–0.35
                persist=True,
                tracker="botsort.yaml",
                device=0,
                verbose=False,
            )
            r = res[0]
            annotated = r.plot()  # BGR
            poly_px = [(int(x*W), int(y*H)) for (x,y) in roi_norm]
            annotated = draw_roi(annotated, poly_px, name=args.roi_name)


            # Extraer detecciones con ids
            dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
                conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                ids = None
                if hasattr(r.boxes, "id") and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy()

                for i in range(xyxy.shape[0]):
                    tid = -1
                    if ids is not None:
                        val = ids[i]
                        if not (isinstance(val, float) and math.isnan(val)):
                            tid = int(val)
                    dets.append({
                        "track_id": tid,
                        "cls": int(cls[i]) if cls is not None else -1,
                        "label": r.names[int(cls[i])] if cls is not None else None,
                        "conf": float(conf[i]) if conf is not None else None,
                        "xyxy": tuple(map(float, xyxy[i])),
                    })
            entered, entries, exited, exits = ze.handle_frame_with_exits(dets, W, H)
            
            if entered:
                for tid, cls_name in entries:
                    print(f"[ZONE] ENTER -> tid={tid}, class={cls_name}")

            if exited:
                for tid, cls_name in exits:
                    print(f"[ZONE] EXIT  -> tid={tid}, class={cls_name}")

            # manda al backend con class name
            if entered or exited:
                sc.handle_events(entries, exits)

            # Enviar a FFmpeg
            try:
                proc.stdin.write(annotated.tobytes())
            except (BrokenPipeError, OSError):
                print("[ERR] FFmpeg terminó o pipe roto. Saliendo...")
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            emitter.shutdown()
        except Exception:
            pass
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
