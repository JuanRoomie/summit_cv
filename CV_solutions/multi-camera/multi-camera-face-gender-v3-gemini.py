#!/usr/bin/env python3
import threading
import subprocess, shlex
import time
import numpy as np
import cv2
from ultralytics import YOLO
from metadata_writer import MetadataWriter

# ---- Recomendado: evita race conditions internas de OpenCV
cv2.setNumThreads(1)

# ===== Writer global + lock (sin cambios) =====
metadata_writer = MetadataWriter(
    base_filename="metadata_faces",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración (sin cambios) =====
MODEL_PATH = "yolov11n-face.engine"
SOURCES = [
    #"cremramirez_rostros.mp4"
    "rtsp://admin:R00m13b0t@192.168.1.40:554"
]
STREAM_NAMES = ["gender"]
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.35

# Cada cuántos frames correr edad/género (reduce CPU)
N_ATTR_EVERY = 10

# +++++ MEJORA 1: AGREGAR MARGEN/PADDING A LA CARA +++++
# Escala (1.3 = 30% de margen)
PADDING_SCALE = 1.3 

# ===== Age/Gender DNN (OpenCV) – rutas a tus ONNX (Recomendación de Modelos) =====
# NOTA: Si usas GooLeNet, esta configuración es la mejor.
# Si la precisión es crucial, considera actualizar a modelos MobileNet o ResNet.
AGE_MODEL = "age_googlenet.onnx"
GENDER_MODEL = "gender_googlenet.onnx"

AGE_BUCKETS = ["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"]
GENDER_LABELS = ["male", "female"]

ID_ATTRS_CACHE = {}

def make_age_gender_nets():
    """Crea nets locales por hilo (thread-safe)."""
    age_net = cv2.dnn.readNetFromONNX(AGE_MODEL)
    gender_net = cv2.dnn.readNetFromONNX(GENDER_MODEL)
    age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return age_net, gender_net

# +++++ MEJORA 2: NORMALIZACIÓN REFINADA +++++
def infer_age_gender(face_bgr, age_net, gender_net):
    """
    Inferencia de edad/género en un crop BGR.
    Normaliza explícitamente a [0, 1] y luego aplica estandarización ImageNet.
    """
    # 1. Redimensionar y convertir a float32, normalizando a rango [0, 1]
    # (Esto es más robusto que depender de scalefactor dentro de blobFromImage)
    face_input = cv2.resize(face_bgr, (224, 224)).astype("float32") / 255.0

    # 2. Convertir a formato NCHW (Blob) sin aplicar scaling/mean
    blob = cv2.dnn.blobFromImage(
        face_input,
        scalefactor=1.0,         # Ya normalizado a [0, 1]
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=True,             # BGR -> RGB
        crop=False
    )
    
    # 3. Normalización por canal (Mean/Std de ImageNet)
    blob[:, 0, :, :] = (blob[:, 0, :, :] - 0.485) / 0.229  # R
    blob[:, 1, :, :] = (blob[:, 1, :, :] - 0.456) / 0.224  # G
    blob[:, 2, :, :] = (blob[:, 2, :, :] - 0.406) / 0.225  # B

    # 4. Forward
    gender_net.setInput(blob)
    g_logits = gender_net.forward().squeeze()
    gender_idx = int(np.argmax(g_logits))
    gender = GENDER_LABELS[gender_idx]

    age_net.setInput(blob)
    a_logits = age_net.forward().squeeze()
    age_idx = int(np.argmax(a_logits))
    age_bucket = AGE_BUCKETS[age_idx]

    return gender, age_bucket

def ffmpeg_writer_cmd(w, h, fps, publish_url):
    cmd = f"""
    ffmpeg -loglevel error -re -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - -an \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g {fps} -b:v 4M \
    -f rtsp -rtsp_transport tcp {publish_url}
    """.strip()
    return shlex.split(cmd)

def run_tracker_and_publish(src, publish_name, source_index):
    """
    - track() en src
    - publica video anotado por FFmpeg
    - escribe metadatos (incluye edad/género en product_id)
    """
    age_net, gender_net = make_age_gender_nets()

    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    fallback_names = {0: "face"}

    try:
        for r in model.track(
            source=src,
            stream=True,
            conf=CONF,
            save=False,
            show=False,
            verbose=False,
            persist=False,
            classes=None
        ):
            if getattr(r, "orig_img", None) is None:
                continue
            
            # r.plot() dibuja las bboxes, pero necesitamos modificarlas
            # Haremos el dibujo manualmente después de la inferencia
            frame = r.orig_img.copy() 
            if frame is None:
                continue

            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls  = boxes.cls.detach().cpu().numpy().astype(int)
                obj_ids = (boxes.id.detach().cpu().numpy().astype(int)
                           if boxes.id is not None else np.full((len(boxes),), -1, dtype=int))

                ts = time.time()
                names = r.names if getattr(r, "names", None) else fallback_names
                h_img, w_img = frame.shape[:2]

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    
                    # --- CÁLCULO DE COORDENADAS PARA METADATOS (sin padding)
                    leftx  = float(x1)
                    topy   = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])
                    class_name = str(names.get(class_id, "face")).replace(" ", "_")

                    cache_key = (source_index, object_id)
                    need_update = (cache_key not in ID_ATTRS_CACHE) or ((frame_idx % N_ATTR_EVERY) == 0)

                    gender_str, age_str = None, None
                    if need_update and object_id != -1:
                        
                        # +++++ MEJORA 3: CÁLCULO DE CROP CON PADDING +++++
                        # Calcular el centro y el nuevo tamaño con padding (1.3x)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        
                        new_w = w * PADDING_SCALE
                        new_h = h * PADDING_SCALE
                        
                        # Coordenadas con padding
                        x1p = int(center_x - new_w / 2)
                        y1p = int(center_y - new_h / 2)
                        x2p = int(center_x + new_w / 2)
                        y2p = int(center_y + new_h / 2)
                        
                        # Recorte seguro (clamping a los límites de la imagen)
                        x1c, y1c = max(0, x1p), max(0, y1p)
                        x2c, y2c = min(w_img, x2p), min(h_img, y2p)

                        if x2c > x1c and y2c > y1c:
                            face_crop = frame[y1c:y2c, x1c:x2c]
                            try:
                                gender_str, age_str = infer_age_gender(face_crop, age_net, gender_net)
                                ID_ATTRS_CACHE[cache_key] = {
                                    "gender": gender_str,
                                    "age": age_str,
                                    "last_frame": frame_idx
                                }
                            except Exception:
                                pass # si falla, conserva últimos valores si existen

                    # --- Si no se actualizó, toma del caché
                    if (gender_str is None or age_str is None) and cache_key in ID_ATTRS_CACHE:
                        cached = ID_ATTRS_CACHE[cache_key]
                        gender_str = cached["gender"]
                        age_str = cached["age"]

                    # Defaults si aún no hay atributos
                    if gender_str is None: gender_str = "unknown"
                    if age_str is None:    age_str = ""

                    # ===== DIBUJAR bbox + etiqueta (Añadido: dibuja la bbox de YOLO) =====
                    # Dibuja la bbox original de YOLO
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

                    # etiqueta
                    label_txt = f"{gender_str}, {age_str}".strip().rstrip(",")
                    (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    ty1 = max(0, y1i - th - baseline - 6)
                    cv2.rectangle(frame,
                                  (x1i, ty1),
                                  (x1i + tw + 8, ty1 + th + baseline + 6),
                                  (0, 255, 0), thickness=-1)
                    cv2.putText(frame, label_txt, (x1i + 4, ty1 + th),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                    # ===== METADATOS (persisten con el mismo ID) =====
                    with write_lock:
                        metadata_writer.write(
                            time=f"{ts}",
                            source=f"{source_index}",
                            frame=frame_idx,
                            object_id=object_id,
                            class_id=class_id,
                            product_id=f"{class_name}_{gender_str}_{age_str}",
                            leftx=leftx,
                            topy=topy,
                            width=width,
                            height=height,
                            conf=1.0
                        )

            # ===== PUBLICAR POR FFMPEG (sin cambios) =====
            h, w = frame.shape[:2]
            if ffmpeg_proc is None:
                cmd = ffmpeg_writer_cmd(w, h, FPS, publish_url)
                ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError):
                print(f"[WARN] FFmpeg cerró stdin para {publish_name}")
                break

            frame_idx += 1

    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ffmpeg_proc.wait(timeout=3)

def main():
    threads = []
    for idx, (src, name) in enumerate(zip(SOURCES, STREAM_NAMES)):
        t = threading.Thread(
            target=run_tracker_and_publish,
            args=(src, name, idx),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    with write_lock:
        metadata_writer.close()

if __name__ == "__main__":
    main()