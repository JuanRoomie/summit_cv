# antitheft_core.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import cv2
import json
import math

Point = Tuple[float, float]   # (x, y) en píxeles
NPoint = Tuple[float, float]  # normalizado 0..1

# ---------------- Geometría ----------------

def _w_h(frame) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    return w, h

def denorm_points(points: List[NPoint], w: int, h: int) -> List[Point]:
    return [(int(px * w), int(py * h)) for (px, py) in points]

def bbox_center_xyxy(xyxy) -> Point:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_bottom_center_xyxy(xyxy) -> Point:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return ((x1 + x2) / 2.0, y2)

def point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    # ray casting
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside

def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    # helper cross product
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

# ---------------- Zonas & Config ----------------

@dataclass
class StreamZoneConfig:
    name: str
    exit_line_norm: List[NPoint]               # 2 puntos [(x,y), (x,y)]
    must_pass_polygon_norm: List[NPoint]       # polígono normalizado

@dataclass
class ZoneManager:
    per_stream: Dict[str, StreamZoneConfig] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "ZoneManager":
        with open(path, "r") as f:
            data = json.load(f)
        per_stream = {}
        for stream_name, cfg in data.get("streams", {}).items():
            per_stream[stream_name] = StreamZoneConfig(
                name=cfg.get("name", stream_name),
                exit_line_norm=cfg["exit_line"],                       # [[x1,y1],[x2,y2]]
                must_pass_polygon_norm=cfg["must_pass_zone"]["points"] # [[x,y]...]
            )
        return cls(per_stream=per_stream)

# ---------------- Estado por objeto & Detector ----------------

@dataclass
class TrackState:
    visited_zone: bool = False
    last_point: Optional[Point] = None
    last_seen_ts: float = field(default_factory=time.time)

class SuspiciousDetector:
    """
    Regla: si el objeto cruza la línea de salida y NUNCA ha estado dentro de la zona obligatoria -> evento sospechoso.
    """
    def __init__(self, stream_name: str, zones: ZoneManager, decay_seconds: float = 10.0):
        self.stream_name = stream_name
        self.zones = zones
        self.decay = decay_seconds
        self.state: Dict[int, TrackState] = {}

    def _get_cfg_px(self, frame) -> Tuple[List[Point], List[Point]]:
        w, h = _w_h(frame)
        s_cfg = self.zones.per_stream[self.stream_name]
        line_px = denorm_points(s_cfg.exit_line_norm, w, h)
        poly_px = denorm_points(s_cfg.must_pass_polygon_norm, w, h)
        return line_px, poly_px

    def update(self, frame, tracks_xyxy: List[Tuple[int, List[float]]]) -> List[Dict]:
        """
        tracks_xyxy: lista de (track_id, [x1,y1,x2,y2]) para el frame actual.
        Devuelve una lista de eventos sospechosos detectados en este frame.
        """
        now = time.time()
        # Limpieza por inactividad
        rm = [tid for tid, st in self.state.items() if now - st.last_seen_ts > self.decay]
        for tid in rm:
            self.state.pop(tid, None)

        if self.stream_name not in self.zones.per_stream:
            return []

        line_px, poly_px = self._get_cfg_px(frame)
        p1, p2 = line_px[0], line_px[1]

        events = []
        for tid, xyxy in tracks_xyxy:
            st = self.state.setdefault(tid, TrackState())
            st.last_seen_ts = now

            # Punto representativo del objeto (centro o “bottom-center”)
            pt = bbox_bottom_center_xyxy(xyxy)

            # Marcar si entra a la zona
            if point_in_polygon(pt, poly_px):
                st.visited_zone = True

            # Detección de cruce de línea (segmento de movimiento entre last_point -> pt)
            if st.last_point is not None:
                if segments_intersect(st.last_point, pt, p1, p2):
                    if not st.visited_zone:
                        events.append({
                            "stream": self.stream_name,
                            "track_id": tid,
                            "when": now,
                            "reason": "line_cross_without_zone",
                            "point": pt
                        })
            st.last_point = pt

        # Render opcional (línea y polígono)
        cv2.line(frame, p1, p2, (80, 230, 30), 2)
        cv2.polylines(frame, [cv2.convexHull(cv2.UMat(cv2.UMat(np.array(poly_px, dtype='int32')))).get()], True, (20, 180, 255), 2)
        # Si tu OpenCV no soporta UMat/convexHull así, usa directamente:
        # cv2.polylines(frame, [np.array(poly_px, dtype='int32')], True, (20, 180, 255), 2)

        return events

# ---------------- Dibujos utilitarios (opcional) ----------------
import numpy as np

def draw_track_id(frame, xyxy, track_id: int, color=(255, 255, 255)):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID {track_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
