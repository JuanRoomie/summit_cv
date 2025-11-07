#!/usr/bin/env python3
import threading
import subprocess, shlex
import time
import numpy as np
import cv2
from collections import Counter
from ultralytics import YOLO
from metadata_writer import MetadataWriter

# ---- Recomendado: evita race conditions internas de OpenCV
cv2.setNumThreads(1)

# ===== Writer global + lock =====
metadata_writer = MetadataWriter(
    base_filename="metadata_faces",
    header="Time,Source,Frame,ObjectID,ClassID,ClassName,Leftx,Topy,Width,Height\n",
    output_dir="metadata"
)
write_lock = threading.Lock()

# ===== Configuración =====
MODEL_PATH = "yolov11n-face.engine"
SOURCES = [
    #"cremramirez_rostros.mp4"
    "rtsp://admin:R00m13b0t@192.168.1.40:554"
]
STREAM_NAMES = ["gender"]   # solo para la publicación
PUBLISH_BASE = "rtsp://127.0.0.1:8554"
FPS = 10
CONF = 0.35

# Cada cuántos frames correr edad/género (reduce CPU)
N_ATTR_EVERY = 5  # Reducido de 10 a 5 para más actualizaciones

# Tamaño mínimo de rostro para inferencia (evita rostros muy pequeños)
MIN_FACE_SIZE = 40  # píxeles

# Número de inferencias para suavizado temporal
TEMPORAL_WINDOW = 3

# ===== Age/Gender DNN (OpenCV) – rutas a tus ONNX =====
# Descárgalos y ajústalos a tu ruta (ver notas)
AGE_MODEL = "age_googlenet.onnx"
GENDER_MODEL = "gender_googlenet.onnx"

AGE_BUCKETS = ["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"]
GENDER_LABELS = ["male", "female"]

# Cache mejorado: ahora guarda historial para suavizado
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

def preprocess_face(face_bgr):
    """
    Preprocesamiento mejorado del rostro antes de la inferencia.
    - Ecualización de histograma para mejorar contraste
    - Resize con interpolación de calidad
    """
    # Convertir a YUV para ecualización
    face_yuv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YUV)
    face_yuv[:,:,0] = cv2.equalizeHist(face_yuv[:,:,0])
    face_bgr = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
    
    # Resize con interpolación de calidad
    face_resized = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    return face_resized

def infer_age_gender(face_bgr, age_net, gender_net):
    """
    Inferencia mejorada de edad/género en un crop BGR.
    Incluye mejor preprocesamiento, confianza y manejo de errores.
    """
    # Validación de tamaño mínimo
    if face_bgr.shape[0] < MIN_FACE_SIZE or face_bgr.shape[1] < MIN_FACE_SIZE:
        return None, None
    
    try:
        # Preprocesamiento mejorado
        face_processed = preprocess_face(face_bgr)
        
        # 1) Blob limpio: escala a [0,1], SIN mean aquí, y convierte a RGB
        blob = cv2.dnn.blobFromImage(
            face_processed,
            scalefactor=1/255.0,
            size=(224, 224),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # 2) Normalización por canal (mean/std en espacio [0,1])
        blob[:, 0, :, :] = (blob[:, 0, :, :] - 0.485) / 0.229  # R
        blob[:, 1, :, :] = (blob[:, 1, :, :] - 0.456) / 0.224  # G
        blob[:, 2, :, :] = (blob[:, 2, :, :] - 0.406) / 0.225  # B

        # 3) Forward Gender con softmax para probabilidades
        gender_net.setInput(blob)
        g_logits = gender_net.forward().squeeze()
        
        # Aplicar softmax para obtener probabilidades
        g_probs = np.exp(g_logits) / np.sum(np.exp(g_logits))
        gender_idx = int(np.argmax(g_probs))
        gender_conf = float(g_probs[gender_idx])
        
        # Solo usar predicción si confianza > 0.6
        if gender_conf < 0.6:
            gender = None
        else:
            gender = GENDER_LABELS[gender_idx]

        # 4) Forward Age con softmax
        age_net.setInput(blob)
        a_logits = age_net.forward().squeeze()
        
        # Aplicar softmax
        a_probs = np.exp(a_logits) / np.sum(np.exp(a_logits))
        age_idx = int(np.argmax(a_probs))
        age_conf = float(a_probs[age_idx])
        
        # Solo usar predicción si confianza > 0.5
        if age_conf < 0.5:
            age_bucket = None
        else:
            age_bucket = AGE_BUCKETS[age_idx]

        return gender, age_bucket
        
    except Exception as e:
        print(f"[ERROR] infer_age_gender: {e}")
        return None, None

def update_temporal_cache(cache_key, gender_str, age_str, frame_idx):
    """
    Actualiza el cache con suavizado temporal.
    Mantiene un historial de las últimas N inferencias y usa la moda.
    """
    if cache_key not in ID_ATTRS_CACHE:
        ID_ATTRS_CACHE[cache_key] = {
            "gender_history": [],
            "age_history": [],
            "gender": "unknown",
            "age": "",
            "last_frame": frame_idx
        }
    
    cache = ID_ATTRS_CACHE[cache_key]
    
    # Agregar nuevas inferencias si son válidas
    if gender_str is not None:
        cache["gender_history"].append(gender_str)
        if len(cache["gender_history"]) > TEMPORAL_WINDOW:
            cache["gender_history"].pop(0)
    
    if age_str is not None:
        cache["age_history"].append(age_str)
        if len(cache["age_history"]) > TEMPORAL_WINDOW:
            cache["age_history"].pop(0)
    
    # Obtener valor más común (moda) del historial
    if cache["gender_history"]:
        gender_counter = Counter(cache["gender_history"])
        cache["gender"] = gender_counter.most_common(1)[0][0]
    
    if cache["age_history"]:
        age_counter = Counter(cache["age_history"])
        cache["age"] = age_counter.most_common(1)[0][0]
    
    cache["last_frame"] = frame_idx
    
    return cache["gender"], cache["age"]

def process_detection(frame, x1i, y1i, x2i, y2i, object_id, cache_key, 
                     frame_idx, age_net, gender_net, w_img, h_img):
    """
    Procesa una detección individual para inferir género y edad.
    """
    # Necesito (re)inferir?
    need_update = (
        (cache_key not in ID_ATTRS_CACHE) or 
        ((frame_idx % N_ATTR_EVERY) == 0)
    )
    
    # Recorta cara de forma segura con padding
    padding = 10  # píxeles de padding para capturar más contexto
    x1c = max(0, x1i - padding)
    y1c = max(0, y1i - padding)
    x2c = min(w_img, x2i + padding)
    y2c = min(h_img, y2i + padding)
    
    gender_str, age_str = None, None
    
    if need_update and object_id != -1 and (x2c > x1c and y2c > y1c):
        face_crop = frame[y1c:y2c, x1c:x2c]
        
        # Verificar tamaño mínimo
        if face_crop.shape[0] >= MIN_FACE_SIZE and face_crop.shape[1] >= MIN_FACE_SIZE:
            gender_str, age_str = infer_age_gender(face_crop, age_net, gender_net)
            
            # Actualizar cache con suavizado temporal
            gender_str, age_str = update_temporal_cache(
                cache_key, gender_str, age_str, frame_idx
            )
    
    # Si no se actualizó, toma del caché
    if cache_key in ID_ATTRS_CACHE:
        cached = ID_ATTRS_CACHE[cache_key]
        if gender_str is None:
            gender_str = cached.get("gender", "unknown")
        if age_str is None:
            age_str = cached.get("age", "")
    
    # Defaults
    if gender_str is None or gender_str == "unknown":
        gender_str = "unknown"
    if age_str is None:
        age_str = ""
    
    return gender_str, age_str

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
    # Nets locales por hilo (evita 'double free or corruption')
    age_net, gender_net = make_age_gender_nets()

    model = YOLO(MODEL_PATH)
    ffmpeg_proc = None
    publish_url = f"{PUBLISH_BASE}/{publish_name}"
    frame_idx = 0

    # Fallback de nombres por si el .engine no trae labels
    fallback_names = {0: "face"}

    try:
        for r in model.track(
            source=src,
            stream=True,
            conf=CONF,
            save=False,
            show=False,
            verbose=False,
            persist=True,               # IDs consistentes
            classes=None                # Face-only model: no filtrar
        ):
            if getattr(r, "orig_img", None) is None:
                continue
            frame = r.orig_img.copy()

            frame = r.plot()
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
                    leftx  = float(x1)
                    topy   = float(y1)
                    width  = float(max(0.0, x2 - x1))
                    height = float(max(0.0, y2 - y1))
                    object_id = int(obj_ids[i])
                    class_id  = int(cls[i])
                    class_name = str(names.get(class_id, "face")).replace(" ", "_")

                    # --- Clave única por cámara + ID
                    cache_key = (source_index, object_id)

                    # --- INFERENCIA MEJORADA CON LA NUEVA FUNCIÓN
                    gender_str, age_str = process_detection(
                        frame, x1i, y1i, x2i, y2i, object_id, cache_key,
                        frame_idx, age_net, gender_net, w_img, h_img
                    )

                    # ===== DIBUJAR bbox + etiqueta (SUSTITUYE "FACE" por "genero, edad") =====
                    # etiqueta
                    label_txt = f"{gender_str}, {age_str}".strip().rstrip(",")
                    (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # caja de fondo para texto (arriba de la bbox)
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

            # ===== PUBLICAR POR FFMPEG =====
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