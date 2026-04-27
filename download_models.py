"""
download_models.py
──────────────────────────────────────────────────
Railway deploy sırasında çalıştırılır.
yolov8n-pose.pt modelini Ultralytics resmi GitHub'dan indirir.
"""
import os
import sys
import urllib.request

POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"

def download_pose_model():
    pose_path = os.environ.get('MODEL_PATH', 'models/yolov8n-pose.pt')
    scol_path = os.environ.get('SCOL_MODEL_PATH', 'models/model_point4.pt')

    # Klasör oluştur
    pose_dir = os.path.dirname(pose_path)
    if pose_dir:
        os.makedirs(pose_dir, exist_ok=True)

    # Pose model
    print(f"[MODEL] Pose model kontrol: {pose_path}")
    if os.path.exists(pose_path):
        size_mb = os.path.getsize(pose_path) / 1024 / 1024
        print(f"[MODEL] ✅ Zaten mevcut: {pose_path} ({size_mb:.1f} MB)")
    else:
        try:
            print(f"[MODEL] İndiriliyor: {POSE_URL}")
            urllib.request.urlretrieve(POSE_URL, pose_path)
            size_mb = os.path.getsize(pose_path) / 1024 / 1024
            print(f"[MODEL] ✅ İndirildi: {pose_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[MODEL] ⚠ Pose model indirilemedi: {e}", file=sys.stderr)

    # Skolyoz model
    print(f"[MODEL] Skolyoz model kontrol: {scol_path}")
    if os.path.exists(scol_path):
        size_mb = os.path.getsize(scol_path) / 1024 / 1024
        print(f"[MODEL] ✅ Mevcut: {scol_path} ({size_mb:.1f} MB)")
    else:
        print(f"[MODEL] ℹ Skolyoz model bulunamadı: {scol_path}")
        print("[MODEL] ℹ Pose keypoint tahmini fallback kullanılacak")

if __name__ == "__main__":
    download_pose_model()
