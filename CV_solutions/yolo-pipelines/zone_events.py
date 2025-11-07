# zone_events.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable

Point = Tuple[float, float]

def _point_in_poly(pt: Point, poly: List[Point]) -> bool:
    if not poly or len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if xin >= x:
                inside = not inside
    return inside

def _poly_scale_about_centroid(poly: List[Point], scale: float) -> List[Point]:
    """Contrae/expande el polígono respecto a su centroide (scale<1 = contrae)."""
    if not poly:
        return poly
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for (x, y) in poly]

class ZoneEvents:
    """
    ENTRADA: requiere dwell (tiempo) + confianza mínima + estar dentro del ROI *interno*.
    SALIDA: si un track confirmado no vuelve a verse dentro del ROI normal por >= exit_absence_sec.
    """

    def __init__(
        self,
        roi_polygon_norm: List[Point],
        fps: int = 25,
        enter_dwell_sec: float = 1.0,
        exit_absence_sec: float = 2.0,
        conf_enter_min: float = 0.35,         # confianza mínima para confirmar entrada
        conf_keep_min: float = 0.30,          # confianza mínima para “mantener” dentro
        min_box_area_px: int = 80 * 80,       # ignora cajas demasiado pequeñas
        enter_roi_scale: float = 0.92,        # ROI interno (contraído) para ENTRADA
    ):
        self.roi_polygon_norm: List[Point] = list(roi_polygon_norm or [])
        self.fps = max(1, int(fps))
        self.enter_dwell_frames = max(1, int(round(self.fps * float(enter_dwell_sec))))
        self.exit_absence_frames = max(1, int(round(self.fps * float(exit_absence_sec))))
        self.conf_enter_min = float(conf_enter_min)
        self.conf_keep_min = float(conf_keep_min)
        self.min_box_area_px = int(min_box_area_px)
        self.enter_roi_scale = float(enter_roi_scale)

        self._state: Dict[int, Dict] = {}  # por track_id
        self._frame_idx = 0

    # -------- ROI helpers --------
    def update_roi(self, roi_polygon_norm: List[Point]) -> None:
        self.roi_polygon_norm = list(roi_polygon_norm or [])

    def set_fps(self, fps: int, enter_dwell_sec: float = None, exit_absence_sec: float = None):
        self.fps = max(1, int(fps))
        if enter_dwell_sec is not None:
            self.enter_dwell_frames = max(1, int(round(self.fps * float(enter_dwell_sec))))
        if exit_absence_sec is not None:
            self.exit_absence_frames = max(1, int(round(self.fps * float(exit_absence_sec))))

    def _poly_abs(self, W: int, H: int) -> List[Point]:
        return [(max(0.0, min(1.0, x)) * W, max(0.0, min(1.0, y)) * H) for (x, y) in self.roi_polygon_norm]

    # -------- núcleo --------
    def _process_det(self, det: dict, W: int, H: int, poly_enter: List[Point], poly_keep: List[Point]):
        tid = int(det.get("track_id", -1))
        if tid < 0:
            return None
        label = det.get("label") or "item"
        x1, y1, x2, y2 = det.get("xyxy", (0, 0, 0, 0))
        conf = float(det.get("conf", 0.0) or 0.0)

        # filtros de tamaño
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if area < self.min_box_area_px:
            return None

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        inside_enter = _point_in_poly((cx, cy), poly_enter)    # más estricto para ENTRAR
        inside_keep  = _point_in_poly((cx, cy), poly_keep)     # normal para “mantener”

        st = self._state.get(tid)
        if st is None:
            st = self._state[tid] = dict(
                label=label, inside_count=0, confirmed=False,
                last_inside_frame=None, exit_emitted=False
            )
        else:
            st["label"] = label or st["label"]

        # ENTRADA (dwell + conf + ROI interno)
        if inside_enter and conf >= self.conf_enter_min:
            st["inside_count"] += 1
        else:
            # si no pasó por el filtro de entrada, resetea acumulador
            st["inside_count"] = 0

        entered = False
        if not st["confirmed"] and st["inside_count"] >= self.enter_dwell_frames:
            st["confirmed"] = True
            st["exit_emitted"] = False
            st["last_inside_frame"] = self._frame_idx
            entered = True

        # MANTENER (solo si está dentro del ROI normal y tiene conf suficiente)
        if st["confirmed"] and inside_keep and conf >= self.conf_keep_min:
            st["last_inside_frame"] = self._frame_idx

        return (tid, st["label"], entered)

    def _eval_absence(self) -> List[Tuple[int, str]]:
        exits: List[Tuple[int, str]] = []
        for tid, st in list(self._state.items()):
            if not st["confirmed"]:
                continue
            last_in = st.get("last_inside_frame")
            if last_in is None:
                continue
            if (self._frame_idx - last_in) >= self.exit_absence_frames and not st["exit_emitted"]:
                exits.append((tid, st["label"]))
                st["exit_emitted"] = True
                st["confirmed"] = False
                st["last_inside_frame"] = None
                st["inside_count"] = 0
        return exits

    # -------- API pública --------
    def handle_frame_with_exits(
        self, detections: Iterable[dict], W: int, H: int
    ) -> Tuple[bool, List[Tuple[int, str]], bool, List[Tuple[int, str]]]:
        self._frame_idx += 1
        entries: List[Tuple[int, str]] = []

        poly_keep = self._poly_abs(W, H)
        if not poly_keep:
            return (False, [], False, [])
        poly_enter = _poly_scale_about_centroid(poly_keep, self.enter_roi_scale)

        for det in detections:
            out = self._process_det(det, W, H, poly_enter, poly_keep)
            if out is None:
                continue
            tid, label, entered = out
            if entered:
                entries.append((tid, label))

        exits = self._eval_absence()
        return (len(entries) > 0, entries, len(exits) > 0, exits)

    def handle_frame(self, detections: Iterable[dict], W: int, H: int):
        entered, entries, _, _ = self.handle_frame_with_exits(detections, W, H)
        return (entered, entries)

    def handle_frame_exit_only(self, detections: Iterable[dict], W: int, H: int):
        _, _, exited, exits = self.handle_frame_with_exits(detections, W, H)
        return (exited, exits)

    def reset(self):
        self._state.clear()
        self._frame_idx = 0
