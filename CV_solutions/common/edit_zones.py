#!/usr/bin/env python3
# edit_zone_once.py
# Captura un stream RTSP, permite dibujar un polígono con clicks y devuelve
# la lista de puntos normalizados [0..1] en stdout (JSON). Opcionalmente guarda en archivo.
#
# Uso:
#   python edit_zone_once.py --rtsp rtsp://user:pass@ip:554/stream --out nueva_zona.json
#
# Controles:
#   - Click izquierdo: agregar vértice
#   - Z: deshacer último punto
#   - C: limpiar puntos
#   - Enter: terminar, imprimir JSON y cerrar
#   - Esc: abortar sin guardar/imprimir

import argparse
import json
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

def normalize_points(points_px: List[Tuple[int, int]], w: int, h: int) -> List[List[float]]:
    out = []
    for (x, y) in points_px:
        nx = max(0.0, min(1.0, x / max(1, w)))
        ny = max(0.0, min(1.0, y / max(1, h)))
        out.append([float(nx), float(ny)])
    return out

def draw_current_poly(frame, pts):
    # Segmentos
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 180, 255), 2, cv2.LINE_AA)
    # Puntos
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    # Si hay 3+ puntos, dibuja relleno translúcido de referencia
    if len(pts) >= 3:
        overlay = frame.copy()
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (0, 160, 0))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 200, 0), 2, cv2.LINE_AA)

def put_hud(frame, pts_count: int):
    lines = [
        "Edit Zone (single): Click=add  Z=undo  C=clear  Enter=finish  Esc=cancel",
        f"Puntos: {pts_count}"
    ]
    x, y = 12, 24
    for line in lines:
        cv2.rectangle(frame, (x - 6, y - 18), (x + 8 * len(line) + 6, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

def main():
    ap = argparse.ArgumentParser(description="Dibuja un polígono sobre un RTSP y devuelve puntos normalizados [0..1] en JSON.")
    ap.add_argument("--rtsp", required=True, help="URL RTSP del stream")
    ap.add_argument("--out", default=None, help="Ruta de archivo para guardar la lista JSON (opcional)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir RTSP: {args.rtsp}", file=sys.stderr)
        sys.exit(2)

    cv2.namedWindow("Edit Zone (single)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Edit Zone (single)", 1280, 720)

    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    cv2.setMouseCallback("Edit Zone (single)", on_mouse)

    print("[INFO] Controles: Click=añadir punto | Z=deshacer | C=limpiar | Enter=terminar | Esc=cancelar")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        draw_current_poly(frame, points)
        put_hud(frame, len(points))

        cv2.imshow("Edit Zone (single)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc
            print("[INFO] Cancelado por el usuario. No se generó salida.", file=sys.stderr)
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if key in (ord('z'), ord('Z')):
            if points:
                points.pop()

        if key in (ord('c'), ord('C')):
            points.clear()

        if key == 13:  # Enter
            if len(points) < 3:
                print("[WARN] Necesitas al menos 3 puntos para un polígono.", file=sys.stderr)
                continue

            norm = normalize_points(points, w, h)
            json_str = json.dumps(norm, indent=2)
            print(json_str)  # salida principal por stdout

            if args.out:
                try:
                    with open(args.out, "w") as f:
                        f.write(json_str + "\n")
                    print(f"[OK] Guardado también en: {args.out}", file=sys.stderr)
                except Exception as e:
                    print(f"[ERR] No se pudo guardar en {args.out}: {e}", file=sys.stderr)

            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

if __name__ == "__main__":
    main()
