# zone_alerts.py
# Uso: desde tu pipeline -> from zone_alerts import check_and_post_zone_alerts
#      check_and_post_zone_alerts(zones_for_cam, detections, frame.shape[:2], publish_name, api_base=API_BASE)

from typing import List, Tuple, Dict, Any
import time
import requests
import numpy as np
import cv2

# anti-spam muy simple (por objeto y zona)
_LAST_SENT: Dict[Tuple[int, str], float] = {}
_COOLDOWN_S = 2.0  # segundos

_INSIDE_STATE: Dict[Tuple[str, str], bool] = {}

def _zones_to_polys(zones: List[dict], frame_shape: Tuple[int, int]) -> List[Tuple[dict, np.ndarray]]:
    """Convierte zonas (normalizadas o en px) a polígonos en píxeles."""
    h, w = frame_shape
    out = []
    for z in zones or []:
        pts = z.get("points") or []
        if not pts:
            continue
        is_norm = True
        for p in pts:
            try:
                if not (0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0):
                    is_norm = False
                    break
            except Exception:
                is_norm = False
                break
        if is_norm:
            poly = np.array([(int(float(px)*w), int(float(py)*h)) for px,py in pts], dtype=np.int32)
        else:
            poly = np.array([(int(float(px)), int(float(py))) for px,py in pts], dtype=np.int32)
        out.append((z, poly))
    return out

def _inside(point_xy: Tuple[int,int], poly: np.ndarray) -> bool:
    """True si el punto (x,y) está dentro o en borde del polígono."""
    x, y = float(point_xy[0]), float(point_xy[1])
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0.0

def _maybe_post(api_base: str, camera_id: str, zone: dict, class_name: str, cxcy: Tuple[int,int], obj_id: int):
    """Envía POST si no está en cooldown."""
    zid = zone.get("id", "zone")
    key = (int(obj_id) if obj_id is not None else -1, str(zid))
    now = time.time()
    last = _LAST_SENT.get(key, 0.0)
    if now - last < _COOLDOWN_S:
        return  # en cooldown

    payload = {
        "camera_id": camera_id,
        "zone_id": zone.get("id"),
        "zone_name": zone.get("name"),
        "class_name": class_name,
        "action": "enter",
        "type": "Unauthorized Access",
        "is_prohibited": True,
        "meta": {
            "prohibited": True,
            "center": {"x": int(cxcy[0]), "y": int(cxcy[1])},
            "obj_id": int(obj_id) if obj_id is not None else -1
        }
    }
    url = f"{api_base.rstrip('/')}/warehouse/zone-event"
    try:
        r = requests.post(url, json=payload, timeout=2.5)
        print(f"[ZONE_ALERT] {url} -> {r.status_code} {r.text[:180]}")
        _LAST_SENT[key] = now
    except Exception as e:
        print(f"[ZONE_ALERT] POST error: {e}")

def check_and_post_zone_alerts(
    zones_for_cam: List[dict],
    detections: List[Dict[str, Any]],
    frame_shape: Tuple[int,int],
    publish_name: str,
    api_base: str = "http://192.168.1.72:8000"
):
    """
    Revisa cada bbox de 'detections' y si el CENTRO del bbox cae en una zona prohibida,
    envía un POST al backend.
    - zones_for_cam: lista de zonas (con 'points' normalizados o en px, y 'prohibited': bool)
    - detections: [{'xyxy': (x1,y1,x2,y2), 'class_name': str, 'obj_id': int}, ...]
    - frame_shape: (h, w)
    - publish_name: se usa como camera_id
    """
    polys = _zones_to_polys(zones_for_cam, frame_shape)
    if not polys or not detections:
        return

    for det in detections:
        x1, y1, x2, y2 = det.get("xyxy", (0,0,0,0))
        cx = int((float(x1) + float(x2)) / 2.0)
        cy = int((float(y1) + float(y2)) / 2.0)
        cls = str(det.get("class_name", "unknown"))
        obj_id = int(det.get("obj_id", -1))

        for zone, poly in polys:
            if not zone.get("prohibited", False):
                continue

            inside_now = _inside((cx, cy), poly)

            # --- Estado de entrada/salida ---
            if obj_id is not None and obj_id >= 0:
                k_obj = f"obj:{obj_id}"
            else:
                k_obj = f"pt:{int(cx/16)},{int(cy/16)}"  # cuantiza para evitar jitter

            key_state = (k_obj, str(zone.get("id", "zone")))
            was_inside = _INSIDE_STATE.get(key_state, False)

            # Solo dispara en flanco de subida (outside → inside)
            if inside_now and not was_inside:
                _maybe_post(api_base, publish_name, zone, cls, (cx, cy), obj_id)

            # Actualiza el estado
            _INSIDE_STATE[key_state] = inside_now
                # no hacemos break: si por diseño puede caer en múltiples zonas, se envían todas
