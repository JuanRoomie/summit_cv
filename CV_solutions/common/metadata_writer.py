import os
import re
import glob
from datetime import datetime

class MetadataWriter:
    def __init__(self, base_filename="metadata_productos", header="", output_dir="metadata_logs",
                 fps_hint=None):
        """
        Crea un escritor de metadatos que:
        - Busca en output_dir los archivos del día con patrón base_ddmmyyyy_####.txt
        - Encuentra el mayor #### y crea el siguiente (####+1)
        - Si cambia la fecha, reinicia a 0001

        Args:
            base_filename (str): Prefijo del archivo (sin fecha ni índice).
            header (str): Línea de encabezado (se escribe una vez al crear el archivo).
            output_dir (str): Carpeta donde se guardan los .txt.
            fps_hint (float|None): Solo informativo si quieres guardarlo como parte de tu metadata, opcional.
        """
        self.base = base_filename
        self.header = header or ""
        self.output_dir = output_dir
        self.fps_hint = fps_hint
        os.makedirs(self.output_dir, exist_ok=True)

        self.date_str = datetime.now().strftime("%d%m%Y")
        self.filepath, self.file = self._open_next_file()

        # Escribe encabezado si aplica
        if self.header:
            self.file.write(self.header)

    def _open_next_file(self):
        """Abre en modo exclusivo ('x') el siguiente archivo disponible con nombre correlativo."""
        # Buscar el índice máximo existente hoy
        pattern = os.path.join(self.output_dir, f"{self.base}_{self.date_str}_*.txt")
        existing = glob.glob(pattern)

        max_idx = 0
        rx = re.compile(rf"^{re.escape(self.base)}_{self.date_str}_(\d{{4}})\.txt$")
        for p in existing:
            m = rx.search(os.path.basename(p))
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx

        # Intento de creación exclusiva para evitar colisiones
        idx = max_idx + 1
        while True:
            filename = f"{self.base}_{self.date_str}_{idx:04d}.txt"
            path = os.path.join(self.output_dir, filename)
            try:
                f = open(path, "x", buffering=1)  # 'x' = crear fallo si existe; buffering=1 = line-buffered
                return path, f
            except FileExistsError:
                idx += 1  # si alguien ya lo creó, probamos el siguiente

    def write(self, time, source, frame, object_id, class_id, product_id,
              leftx, topy, width, height, conf):
        """Escribe una línea CSV con los metadatos."""
        line = f"{time},{source},{frame},{object_id},{class_id},{product_id}," \
               f"{leftx},{topy},{width},{height},{conf:.2f}\n"
        self.file.write(line)

    def close(self):
        """Cierra el archivo."""
        if self.file and not self.file.closed:
            self.file.close()
