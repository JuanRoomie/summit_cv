# analytics_engine.py
import time
import cv2
import numpy as np

class AnalyticsEngine:
    """
    Motor de analíticas por-ROI:
    - ROIs normalizados (0..1) -> se denormalizan a cada frame (tamaño variable)
    - Trackea por stream_name y object_id:
        * ocupación actual
        * entradas / salidas (cruce de borde)
        * dwell time (tiempo acumulado dentro del ROI)
    - ROIs soportados:
        * {"name": "...", "type": "poly", "points": [(x,y), ...]}  # normalizados
        * {"name": "...", "type": "rect", "xywh": [x,y,w,h]}       # normalizados
    """

    def __init__(self):
        # ROIs por stream: dict[str, list[roi_dict_normalizado]]
        self.rois_norm = {}
        # Estado por stream:
        # per_stream = {
        #   "roi_px": {roi_name: np.array puntos int},
        #   "id_state": {id: {"inside": set(roi_names), "enter_ts": {roi: t0}, "dwell": {roi: secs}}},
        #   "stats": {roi_name: {"entries": int, "exits": int}}
        # }
        self.state = {}

    # ---------- API de configuración ----------
    def set_rois(self, stream_name, roi_list_norm):
        """
        roi_list_norm: lista de dicts normalizados (poly o rect)
        """
        self.rois_norm[stream_name] = roi_list_norm
        # reset parcial (se recalcularán a px por frame)
        if stream_name not in self.state:
            self.state[stream_name] = {"roi_px": {}, "id_state": {}, "stats": {}}

    def add_roi(self, stream_name, roi_norm):
        self.rois_norm.setdefault(stream_name, []).append(roi_norm)

    # ---------- Helpers de geometría ----------
    @staticmethod
    def _rect_to_poly(x, y, w, h):
        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)

    @staticmethod
    def _denorm_point(p, W, H):
        # p = (x_norm, y_norm)
        return int(round(p[0] * W)), int(round(p[1] * H))

    def _denorm_rois(self, stream_name, W, H):
        """
        Convierte los ROIs normalizados a coordenadas de píxel (por frame).
        """
        roi_px = {}
        for roi in self.rois_norm.get(stream_name, []):
            name = roi.get("name", f"roi_{len(roi_px)}")
            if roi.get("type", "poly") == "rect":
                x, y, w, h = roi["xywh"]
                x, y, w, h = int(round(x * W)), int(round(y * H)), int(round(w * W)), int(round(h * H))
                poly = self._rect_to_poly(x, y, w, h)
            else:
                # poly
                pts = [self._denorm_point(p, W, H) for p in roi["points"]]
                poly = np.array(pts, dtype=np.int32)
            roi_px[name] = poly
        self.state[stream_name]["roi_px"] = roi_px

        # inicializa stats si no existen
        stats = self.state[stream_name]["stats"]
        for name in roi_px.keys():
            stats.setdefault(name, {"entries": 0, "exits": 0})

    @staticmethod
    def _center_of_bbox(x1, y1, x2, y2):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        return float(cx), float(cy)

    @staticmethod
    def _point_in_poly(pt, poly):
        # poly: Nx2 int32, pt = (x,y)
        # cv2.pointPolygonTest: >0 inside, 0 on edge, <0 outside
        res = cv2.pointPolygonTest(poly, (pt[0], pt[1]), False)
        return res >= 0  # consideramos borde como inside

    # ---------- Update por frame ----------
    def update(
        self,
        stream_name: str,
        frame_idx: int,
        frame_shape,   # (H, W, C)
        detections     # lista de dicts: [{"id": int, "bbox": (x1,y1,x2,y2), "class_id": int, "class_name": str}, ...]
    ):
        """
        Actualiza el estado de analíticas para un frame y retorna un snapshot:
        {
          "timestamp": float,
          "stream": str,
          "frame": int,
          "per_roi": {
             roi_name: {
               "occupancy": int,
               "entries": int,
               "exits": int,
               "dwell_avg": float (segundos, promedio por id actual), 
             }, ...
          }
        }
        """
        H, W = frame_shape[:2]
        if stream_name not in self.state:
            self.state[stream_name] = {"roi_px": {}, "id_state": {}, "stats": {}}

        # Denormaliza ROIs (por si el tamaño de frame cambia)
        self._denorm_rois(stream_name, W, H)
        roi_px = self.state[stream_name]["roi_px"]
        id_state = self.state[stream_name]["id_state"]
        stats = self.state[stream_name]["stats"]

        now = time.time()

        # 1) Construye mapa id -> set(rois dentro)
        current_inside = {}
        for det in detections:
            oid = int(det["id"])
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = self._center_of_bbox(x1, y1, x2, y2)
            inside = set()
            for roi_name, poly in roi_px.items():
                if self._point_in_poly((cx, cy), poly):
                    inside.add(roi_name)
            current_inside[oid] = {"inside": inside, "center": (cx, cy)}

        # 2) Actualiza transiciones por ID (entradas/salidas + dwell time)
        for oid, cur in current_inside.items():
            st = id_state.setdefault(oid, {"inside": set(), "enter_ts": {}, "dwell": {}})
            prev_inside = st["inside"]
            now_inside = cur["inside"]

            # Entradas = en now_inside pero no en prev_inside
            for roi_name in now_inside - prev_inside:
                st["enter_ts"][roi_name] = now
                stats[roi_name]["entries"] += 1

            # Salidas = en prev_inside pero no en now_inside
            for roi_name in prev_inside - now_inside:
                t0 = st["enter_ts"].pop(roi_name, None)
                if t0 is not None:
                    st["dwell"][roi_name] = st["dwell"].get(roi_name, 0.0) + (now - t0)
                stats[roi_name]["exits"] += 1

            # Mantiene flag de inside
            st["inside"] = now_inside

        # 3) Cierra dwell parcial (promedio momentáneo de los que están dentro)
        per_roi = {}
        for roi_name, poly in roi_px.items():
            # ocupación actual = cuantos IDs están inside
            occ = sum(1 for st in id_state.values() if roi_name in st["inside"])

            # dwell avg = promedio entre:
            # (tiempo acumulado de los que ya salieron) + (tiempo parcial de los que están dentro)
            dwell_sum = 0.0
            count_ids = 0
            for oid, st in id_state.items():
                acc = st["dwell"].get(roi_name, 0.0)
                if roi_name in st["inside"]:
                    t0 = st["enter_ts"].get(roi_name, None)
                    if t0 is not None:
                        acc = acc + (now - t0)
                if acc > 0.0:
                    dwell_sum += acc
                    count_ids += 1
            dwell_avg = dwell_sum / max(count_ids, 1) if (occ > 0 or count_ids > 0) else 0.0

            per_roi[roi_name] = {
                "occupancy": occ,
                "entries": stats[roi_name]["entries"],
                "exits": stats[roi_name]["exits"],
                "dwell_avg": dwell_avg
            }

        return {
            "timestamp": now,
            "stream": stream_name,
            "frame": frame_idx,
            "per_roi": per_roi
        }

    # ---------- Overlay (opcional) ----------
    def draw_overlays(self, stream_name, frame, color=(0, 200, 0)):
        """
        Dibuja ROIs y sus nombres. Llamar después de denormalizar (update hace eso).
        """
        if stream_name not in self.state:
            return frame
        roi_px = self.state[stream_name].get("roi_px", {})
        out = frame
        for name, poly in roi_px.items():
            cv2.polylines(out, [poly], isClosed=True, color=color, thickness=2)
            # etiqueta
            x, y = int(poly[:,0].mean()), int(poly[:,1].mean())
            cv2.putText(out, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
