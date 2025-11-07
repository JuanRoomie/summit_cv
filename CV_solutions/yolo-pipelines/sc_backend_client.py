# sc_backend_client.py
# -*- coding: utf-8 -*-
"""
Cliente de Self-Checkout para integrarlo con tu pipeline de zonas.

Objetivo:
- Recibir listas de ENTRADAS y SALIDAS detectadas por ZoneEvents.
- Llamar al backend con el mismo formato que usaba el ds_selfcheckout_probe:
    POST /selfcheckout/session/{id}/scan
        {
          "product_code": "SKU_X",
          "name": "Nombre legible",
          "unit_price_cents": 1000,
          "detection_id": "SESSION:RUN:TRACK:SEQ",
          "source": "vision"
        }
    POST /selfcheckout/session/{id}/item/{product_code}/decrement
    DELETE /selfcheckout/session/{id}/item/{product_code}

Uso mínimo en el pipeline:
    from sc_backend_client import SCClient, SCConfig

    sc = SCClient(SCConfig(
        api_base="http://127.0.0.1:8000",
        station_id="caja-01",
        # session_id=None -> auto-resuelve por estación
        default_price_cents=1000,
        exit_action="decrement",   # o "remove"
        products_map={"coke": {"code": "SKU_COCA", "name": "Coca Cola", "price": 1500}}
    ))

    # Dentro del loop:
    entered_any, entries, exited_any, exits = ze.handle_frame_with_exits(dets, W, H)
    if entered_any or exited_any:
        sc.handle_events(entries, exits)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable, Union
import threading
import queue
import os, time, uuid
import requests
import re


# ---------------- Config ----------------

@dataclass
class SCConfig:
    api_base: str = "http://127.0.0.1:8000"
    api_key: str = ""
    station_id: str = "caja-01"
    session_id: Optional[str] = None          # si None, se auto-resuelve por estación
    session_refresh_sec: float = 5.0          # refresh periódico si no hay session fijada
    default_price_cents: int = 1000
    exit_action: str = "decrement"            # "decrement" | "remove"
    source: str = "vision"
    # Mapa de productos: label -> {code, name, price}
    #   también puedes pasar un callable(label:str)->(code,str_name,price_cents)
    products_map: Union[Dict[str, Dict], Callable[[str], Tuple[str, str, int]]] = field(default_factory=dict)
    # Opcional: normalizar/slug de labels a product_code cuando no exista en map
    slugify_fallback: bool = True
    # Cola HTTP
    http_timeout_s: float = 1.5
    http_queue_max: int = 2000
    http_workers: int = 2


# ---------------- Util ----------------

_slug_re = re.compile(r"[^a-z0-9]+")

def _slug(s: str) -> str:
    s = (s or "").lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or "item"

def _truthy(v) -> bool:
    if v is None: return False
    if isinstance(v, (bool, int, float)): return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "in")


# ---------------- Worker HTTP async ----------------

class _HttpPool:
    def __init__(self, headers: Dict[str, str], timeout_s: float, workers: int, qmax: int):
        self.headers = headers
        self.timeout_s = timeout_s
        self.q: queue.Queue = queue.Queue(maxsize=qmax)
        self.workers = []
        self._stop = threading.Event()
        for _ in range(max(1, workers)):
            t = threading.Thread(target=self._run, daemon=True)
            t.start()
            self.workers.append(t)

    def _run(self):
        while not self._stop.is_set():
            try:
                method, url, body = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if method == "POST":
                    requests.post(url, json=body or {}, headers=self.headers, timeout=self.timeout_s)
                elif method == "DELETE":
                    requests.delete(url, headers=self.headers, timeout=self.timeout_s)
                elif method == "GET":
                    requests.get(url, headers=self.headers, timeout=self.timeout_s)
            except Exception as e:
                print(f"[SC-HTTP] {method} {url} ERROR: {e}")

    def submit(self, method: str, url: str, body: Optional[dict] = None):
        try:
            self.q.put_nowait((method, url, body))
        except queue.Full:
            print("[SC-HTTP] WARN: cola llena, descartando evento")

    def close(self, join_timeout: float = 1.0):
        self._stop.set()
        for t in self.workers:
            t.join(timeout=join_timeout)


# ---------------- Cliente principal ----------------

class SCClient:
    def __init__(self, cfg: SCConfig):
        self.cfg = cfg
        self._headers = {"X-API-KEY": cfg.api_key} if cfg.api_key else {}
        self._http = _HttpPool(self._headers, timeout_s=max(2.5, cfg.http_timeout_s),
                               workers=cfg.http_workers, qmax=cfg.http_queue_max)

        self._enter_seq: Dict[int, int] = {}

        # 🔹 ID único por ejecución (sobrevive mientras el proceso viva)
        # corto y legible: 8 chars de un uuid4
        self._run_id = uuid.uuid4().hex[:8]

        self._last_session_refresh: float = 0.0
        if not self.cfg.session_id:
            self._refresh_session(blocking=True)

    # ---------- API pública ----------

    def close(self):
        self._http.close()

    def handle_events(
        self,
        entries: List[Tuple[int, str]],
        exits: List[Tuple[int, str]],
    ):
        """
        entries: [(track_id, class_name)] para OUT->IN
        exits:   [(track_id, class_name)] para IN->OUT (o ausencia si así lo reporta ZoneEvents)
        """
        # refresco suave de sesión si no hay fija
        if not self.cfg.session_id:
            self._refresh_session(blocking=False)

        # ENTRADAS -> scan
        for tid, label in entries:
            self._on_enter(tid, label)

        # SALIDAS -> decrement/remove
        for tid, label in exits:
            self._on_exit(tid, label)

    # ---------- Internos: enter/exit ----------

    def _on_enter(self, track_id: int, label: str):
        sid = self.cfg.session_id
        if not sid:
            print("[SC] skip ENTER: sin session_id")
            return

        enter_seq = self._enter_seq.get(track_id, 0) + 1
        self._enter_seq[track_id] = enter_seq

        # 🔹 detection_id ahora incluye run_id para evitar choques tras reinicios
        detection_id = f"{sid}:{self._run_id}:{track_id}:{enter_seq}"

        code, name, price = self._resolve_product(label)

        payload = {
            "product_code": code,
            "name": name,
            "unit_price_cents": int(price),
            "detection_id": detection_id,
            "source": self.cfg.source,
        }
        url_scan = f"{self.cfg.api_base}/selfcheckout/session/{sid}/scan"

        print(f"[SC] ENTER scan track={track_id} label='{label}' code={code} idem={detection_id}")

        # envío síncrono para ver respuesta (puedes volver a async cuando confirmes)
        try:
            r = requests.post(url_scan, json=payload, headers=self._headers, timeout=max(3, self.cfg.http_timeout_s))
            txt = (r.text or "").replace("\n"," ").replace("\r"," ")
            print(f"[SC-HTTP] POST /scan -> {r.status_code} {txt[:200]}")
        except Exception as e:
            print(f"[SC-HTTP] scan ERROR: {e}")
            return

        # (opcional) lectura del carrito para verificar
        try:
            g = requests.get(f"{self.cfg.api_base}/selfcheckout/session/{sid}", headers=self._headers, timeout=max(3, self.cfg.http_timeout_s))
            print(f"[SC-HTTP] GET /session -> {g.status_code}")
            if g.ok:
                data = g.json()
                items = data.get("items") or data.get("cart") or data.get("lines") or []
                print("[SC] CART:", items)
        except Exception as e:
            print(f"[SC-HTTP] get-cart ERROR: {e}")

    def _on_exit(self, track_id: int, label: str):
        sid = self.cfg.session_id
        if not sid:
            print("[SC] skip EXIT: sin session_id")
            return

        code, name, _ = self._resolve_product(label)

        if self.cfg.exit_action == "remove":
            url = f"{self.cfg.api_base}/selfcheckout/session/{sid}/item/{code}"
            print(f"[SC] EXIT REMOVE track={track_id} code={code}")
            self._http.submit("DELETE", url, None)
        else:
            url = f"{self.cfg.api_base}/selfcheckout/session/{sid}/item/{code}/decrement"
            print(f"[SC] EXIT DECREMENT track={track_id} code={code}")
            self._http.submit("POST", url, {})

    # ---------- Resolución de producto ----------

    def _resolve_product(self, label: str) -> Tuple[str, str, int]:
        """
        Devuelve (code, name, unit_price_cents) usando:
          - callable personalizado
          - dict {label: {code,name,price}}
          - fallback slug (si no hay map)
        """
        pm = self.cfg.products_map
        if callable(pm):
            try:
                code, name, price = pm(label)
                return str(code), str(name), int(price)
            except Exception as e:
                print(f"[SC] WARN resolver callable falló para '{label}': {e}")

        if isinstance(pm, dict) and label in pm:
            entry = pm[label] or {}
            code = entry.get("code") or _slug(entry.get("name", label))
            name = entry.get("name", label)
            price = int(entry.get("price", self.cfg.default_price_cents))
            return code, name, price

        # fallback genérico: usa class name como SKU (slug)
        code = _slug(label) if self.cfg.slugify_fallback else label
        return code, label, self.cfg.default_price_cents

    # ---------- Sesión ----------

    def _refresh_session(self, blocking: bool):
        now = time.time()
        if blocking or (now - self._last_session_refresh) >= self.cfg.session_refresh_sec:
            self._last_session_refresh = now
            try:
                url = f"{self.cfg.api_base}/selfcheckout/station/{self.cfg.station_id}/session"
                r = requests.get(url, headers=self._headers, timeout=self.cfg.http_timeout_s)
                if r.ok:
                    data = r.json()
                    sid = data.get("session_id") or data.get("id") or data.get("session")
                    if sid:
                        if sid != self.cfg.session_id:
                            print(f"[SC] session_id={sid}")
                        self.cfg.session_id = str(sid)
                else:
                    print(f"[SC] WARN station session GET -> {r.status_code} {(r.text or '')[:120]}")
            except Exception as e:
                print(f"[SC] WARN refresh session: {e}")
