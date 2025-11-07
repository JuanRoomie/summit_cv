# damage_persistence.py
from __future__ import annotations
import threading, time, queue, requests, math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict, Any, List, Set

# ===== Tipos =====
@dataclass
class Detection:
    track_id: Optional[int]
    cls_name: str
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]

@dataclass
class AlertPayload:
    product_name: str
    damage_type: str
    station: str
    detected_at_iso: str

# ===== Utilidades =====
def _now_mono() -> float:
    return time.monotonic()

def _iso_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _centroid_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _quant_key(b: Tuple[float, float, float, float], bin_size: int = 32) -> Tuple[int, int]:
    cx, cy = _centroid_xyxy(b)
    return (int(cx // bin_size), int(cy // bin_size))

# ===== Envío asíncrono (no bloquea inferencia) =====
class _Sender(threading.Thread):
    def __init__(self, api_base: str, api_token: Optional[str] = None, timeout: float = 4.0):
        super().__init__(daemon=True)
        self.api_base = api_base.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.q: "queue.Queue[AlertPayload]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()

    def run(self):
        s = requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        url = f"{self.api_base}/damaged/events"
        while not self._stop.is_set():
            try:
                p = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                body = {
                    "product_name": p.product_name,
                    "damage_type": p.damage_type,
                    "station": p.station,
                    "detected_at": p.detected_at_iso,
                }
                s.post(url, json=body, headers=headers, timeout=self.timeout)
            except Exception:
                pass
            finally:
                self.q.task_done()

    def enqueue(self, payload: AlertPayload):
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

# ===== Núcleo: persistencia de daño =====
class DamagePersistenceEmitter:
    """
    Dispara una alerta si una detección con clase de daño persiste >= min_duration_sec.
    - Ignora clase "no defect"
    - Acepta clase "torn" y "wrinkle"
    - Si cambia la clase (p.ej. torn->wrinkle) reinicia el temporizador
    - Usa track_id si está disponible; si no, centro cuantizado.
    - En esta versión: sólo una alerta por track_id (no repite)
    """
    def __init__(
        self,
        api_base: str,
        api_token: Optional[str] = None,
        product_name: str = "Caja de empaquetado",
        min_duration_sec: float = 3.0,
        conf_threshold: float = 0.40,
        damage_classes: Optional[Set[str]] = None,
        ignore_class: str = "no defect",
        debounce_sec: float = 5.0,
    ):
        self.api_base = api_base
        self.product_name = product_name
        self.min_dur = float(min_duration_sec)
        self.conf_thr = float(conf_threshold)
        self.damage_classes = {c.strip().lower() for c in (damage_classes or {"torn", "wrinkle"})}
        self.ignore_class = ignore_class.strip().lower()
        self.debounce = float(debounce_sec)

        # Estado por objeto
        self._first_seen: Dict[Any, float] = {}
        self._last_seen: Dict[Any, float] = {}
        self._label_at_first: Dict[Any, str] = {}
        self._alert_sent: Set[Any] = set()  # ✅ nuevo: IDs que ya enviaron alerta

        self._lock = threading.Lock()
        self._sender = _Sender(api_base=api_base, api_token=api_token)
        self._sender.start()

    def _key_for(self, det: Detection) -> Any:
        if det.track_id is not None:
            return ("tid", int(det.track_id))
        return ("cent",) + _quant_key(det.bbox_xyxy, bin_size=32)

    def _is_damage(self, cls_name: str, conf: float) -> bool:
        if conf < self.conf_thr:
            return False
        name = cls_name.strip().lower()
        if name == self.ignore_class:
            return False
        return name in self.damage_classes

    def process(self, detections: Iterable[Detection], station: str, now_ts: Optional[float] = None):
        t = _now_mono() if now_ts is None else now_ts
        seen: Set[Any] = set()

        for det in detections:
            if not self._is_damage(det.cls_name, det.conf):
                continue

            k = self._key_for(det)
            seen.add(k)
            label = det.cls_name.strip().lower()

            with self._lock:
                # Si ya se mandó alerta para este ID → no hacer nada más
                if k in self._alert_sent:
                    self._last_seen[k] = t
                    continue

                # Nueva pista o cambio de clase → reinicia temporizador
                if (k not in self._first_seen) or (self._label_at_first.get(k) != label):
                    self._first_seen[k] = t
                    self._label_at_first[k] = label

                self._last_seen[k] = t
                dur = t - self._first_seen[k]

                # Si persiste suficiente tiempo y no ha alertado antes → alerta única
                if dur >= self.min_dur:
                    self._alert_sent.add(k)  # ✅ marcar como ya alertado
                    self._sender.enqueue(AlertPayload(
                        product_name=self.product_name,
                        damage_type=det.cls_name,
                        station=station,
                        detected_at_iso=_iso_utc(),
                    ))

        # Limpieza de objetos que desaparecen
        self._prune(seen, t)

    def _prune(self, seen_now: Set[Any], t: float, grace_sec: float = 1.0):
        with self._lock:
            to_del = []
            for k, last in self._last_seen.items():
                if k not in seen_now and (t - last) > grace_sec:
                    to_del.append(k)
            for k in to_del:
                self._first_seen.pop(k, None)
                self._last_seen.pop(k, None)
                self._label_at_first.pop(k, None)

# ===== Parser para Ultralytics Results =====
def parse_ultralytics_result(r: Any, names: Optional[Dict[int, str]] = None) -> List[Detection]:
    out: List[Detection] = []
    if names is None:
        names = getattr(r, "names", None)

    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return out

    ids   = getattr(boxes, "id", None)
    confs = getattr(boxes, "conf", None)
    clss  = getattr(boxes, "cls", None)
    xyxy  = getattr(boxes, "xyxy", None)

    if xyxy is None or clss is None or confs is None:
        return out

    n = len(xyxy)
    for i in range(n):
        tid  = int(ids[i].item()) if ids is not None and ids[i] is not None else None
        conf = float(confs[i].item())
        cidx = int(clss[i].item())
        cname = str(names[cidx]) if names and cidx in names else f"class_{cidx}"
        b = xyxy[i].tolist()
        out.append(Detection(
            track_id=tid,
            cls_name=cname,
            conf=conf,
            bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        ))
    return out
