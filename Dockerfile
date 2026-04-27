FROM python:3.11-slim

# Sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt constraints.txt ./

# 1. NumPy önce ve sabit
RUN pip install --no-cache-dir numpy==1.26.4

# 2. PyTorch — numpy 1.26 ile uyumlu sürüm
RUN pip install --no-cache-dir "torch>=2.0,<2.5" "torchvision<0.20"

# 3. Diğer bağımlılıklar
RUN pip install --no-cache-dir -r requirements.txt

# 4. cpuinfo
RUN pip install --no-cache-dir py-cpuinfo

# 5. Ultralytics (yedek pose model için)
RUN pip install --no-cache-dir ultralytics==8.3.40 --no-deps

# 6. SpinePose
RUN pip install --no-cache-dir spinepose

# 7. KRİTİK: Eğer ultralytics/spinepose full opencv kurmuşsa kaldır
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true

# 8. SADECE headless'i tekrar kur
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80

# 9. Doğrula
RUN python -c "import numpy; print('[OK] numpy', numpy.__version__)"
RUN python -c "import torch; print('[OK] torch', torch.__version__)"
RUN python -c "import cv2; print('[OK] cv2', cv2.__version__)"
RUN python -c "import cpuinfo; print('[OK] cpuinfo OK')"
RUN python -c "from ultralytics import YOLO; print('[OK] ultralytics OK')"
RUN python -c "from spinepose import SpinePoseEstimator; print('[OK] spinepose OK')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python download_models.py && gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 180 app:app"]
