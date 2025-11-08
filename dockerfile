# Imagen base oficial para Jetson (JetPack 5.1.2) con Ultralytics y wheels ARM64 de Torch/TV/ORT ya resueltos
FROM ultralytics/ultralytics:latest-jetson-jetpack5

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Paquetes del sistema adicionales (ajusta si te faltara algo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Requisitos Python "seguros" (evitamos re-instalar torch/vision/onnxruntime-gpu/tensorrt/tensorflow/ultralytics)
COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip uv && \
    uv pip install --system -r /tmp/requirements-additions.txt && \
    rm -rf /root/.cache/pip

# ===========================
# OPCIONAL: TensorFlow 2.11.0
# ===========================
# TIP: En Jetson, TF se instala con wheels específicos (aarch64).
# Si tienes la URL del wheel correcto (cp38 aarch64 para JetPack 5.1.2),
# descomenta y ajusta las líneas de abajo:
#
# RUN uv pip install --system \
#     https://<TU_URL_WHEEL>/tensorflow-2.11.0-cp38-cp38-linux_aarch64.whl \
#     tensorflow-estimator==2.11.0 \
#     tensorflow-hub==0.12.0 \
#     tensorflow-io-gcs-filesystem==0.35.0 \
#     tensorflowjs==3.18.0
#
# Nota: si TF ya viene instalado en tu entorno base o lo gestionas fuera,
# puedes dejar este bloque comentado.

WORKDIR /workspace
