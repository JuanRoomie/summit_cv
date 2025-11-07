#!/usr/bin/env python3
from zone_alerts import check_and_post_zone_alerts

# Simula una zona restringida (como en tu zones_config.json)
zones_for_cam = [
    {
        "id": "restricted_dock",
        "name": "Restricted Dock",
        "prohibited": True,
        "points": [
            [0.1, 0.2],
            [0.3, 0.2],
            [0.3, 0.4],
            [0.1, 0.4]
        ]
    }
]

# Simula una detección (bbox centrado dentro de la zona)
detections = [
    {
        "obj_id": 1,
        "class_name": "person",
        "xyxy": (100, 100, 200, 200)  # coordenadas absolutas de bbox
    }
]

# Tamaño del frame simulado (para convertir puntos normalizados)
frame_shape = (480, 640)  # (h, w)
publish_name = "people-beh"
API_BASE = "http://192.168.1.72:8000"

# Ejecuta el chequeo (debería mandar un POST al backend)
check_and_post_zone_alerts(zones_for_cam, detections, frame_shape, publish_name, api_base=API_BASE)
