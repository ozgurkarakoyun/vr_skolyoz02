"""
download_models.py
──────────────────────────────────────────────────
Railway deploy sırasında çalıştırılır.
SpinePose modeli ilk inference'ta kendi otomatik indirir.
YOLO yedek için ekstra indirme yapar.
"""
import os
import sys
import urllib.request

YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"

def download_models():
    pose_path = os.environ.get('MODEL_PATH', 'models/yolov8n-pose.pt')
    scol_path = os.environ.get('SCOL_MODEL_PATH', 'models/model_point4.pt')

    pose_dir = os.path.dirname(pose_path)
    if pose_dir:
        os.makedirs(pose_dir, exist_ok=True)

    # YOLO yedek pose model
    print(f"[MODEL] YOLO yedek model kontrol: {pose_path}")
    if os.path.exists(pose_path):
        size_mb = os.path.getsize(pose_path) / 1024 / 1024
        print(f"[MODEL] ✅ Mevcut: {pose_path} ({size_mb:.1f} MB)")
    else:
        try:
            print(f"[MODEL] İndiriliyor: {YOLO_URL}")
            urllib.request.urlretrieve(YOLO_URL, pose_path)
            size_mb = os.path.getsize(pose_path) / 1024 / 1024
            print(f"[MODEL] ✅ İndirildi: {pose_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[MODEL] ⚠ YOLO model indirilemedi: {e}", file=sys.stderr)

    # Skolyoz model
    print(f"[MODEL] Skolyoz model kontrol: {scol_path}")
    if os.path.exists(scol_path):
        size_mb = os.path.getsize(scol_path) / 1024 / 1024
        print(f"[MODEL] ✅ Mevcut: {scol_path} ({size_mb:.1f} MB)")
    else:
        print(f"[MODEL] ℹ Skolyoz model bulunamadı, pose fallback kullanılır")

    # SpinePose info
    print("[MODEL] ℹ SpinePose modeli ilk inference'ta otomatik indirilecek")

if __name__ == "__main__":
    download_models()
