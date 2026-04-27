FROM python:3.11-slim

# Sistem kütüphaneleri (libGL, libstdc++ vs.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt constraints.txt ./

# 1. Headless opencv'yi önce kur
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80

# 2. Diğer bağımlılıkları kur
RUN pip install --no-cache-dir -r requirements.txt

# 3. Ultralytics — yeni sürüm (yolo26 modelleri için C3k2 sınıfı gerekli)
#    py-cpuinfo de açıkça ekleniyor (No module named 'cpuinfo' fix)
RUN pip install --no-cache-dir py-cpuinfo
RUN pip install --no-cache-dir ultralytics==8.3.40 --no-deps

# 4. Full opencv geldiyse temizle
RUN pip uninstall -y opencv-python || true

# 5. Doğrula
RUN python -c "import cv2; print('[OK] cv2', cv2.__version__)"
RUN python -c "import cpuinfo; print('[OK] cpuinfo OK')"
RUN python -c "from ultralytics import YOLO; print('[OK] ultralytics OK')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 app:app"]
