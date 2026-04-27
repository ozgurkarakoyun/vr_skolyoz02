"""
marker_engine.py
─────────────────────────────────────────────────────────────────
Marker Tabanlı Schroth Analiz Motoru

Bu motor, hastanın sırtına yapıştırılan 9 fiziksel markerı tespit eder
ve anatomik konumlarına göre otomatik atama yapar.

Marker yerleşimi (sırtta, hasta dik durumda):

       T1 (üst omurga)
   R-AKRO ●     ● L-AKRO
            
            ● T-APEX (torakal apex)
            
            ● TL-APEX (torakolumbar apex)
            
            ● L-APEX (lumbar apex)
            
   R-PSIS ●  L5  ● L-PSIS
            (sağ kalça | omurga sonu | sol kalça)

Modelin tek sınıfı vardır ('back'), atamaları kod yapar.
"""
import math
import cv2
import numpy as np
import base64
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# ─── Klinik eşikler ─────────────────────────────────────────
COLOR_GREEN  = (0, 200, 0)
COLOR_BLUE   = (255, 140, 0)
COLOR_RED    = (0, 0, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_YELLOW = (0, 220, 220)

ANGLE_HIGH = 25.0   # KRİTİK eşiği (derece)
ANGLE_MID  = 10.0   # ORTA eşiği

REQUIRED_MARKERS = 9

# Anatomik nokta isimleri (sıralama için)
MARKER_NAMES = [
    't1',              # 0  Üst omurga
    'right_acromion',  # 1  Sağ omuz
    'left_acromion',   # 2  Sol omuz
    't_apex',          # 3  Torakal apex
    'tl_apex',         # 4  Torakolumbar apex
    'l_apex',          # 5  Lumbar apex
    'l5',              # 6  Alt omurga
    'right_psis',      # 7  Sağ iliak kanat
    'left_psis',       # 8  Sol iliak kanat
]

# ─── Model yükleme ──────────────────────────────────────────
_marker_model = None

def get_marker_model():
    global _marker_model
    if _marker_model is None:
        import torch
        _orig_load = torch.load
        try:
            from ultralytics import YOLO
            torch.load = lambda *a, **kw: _orig_load(
                *a, **{**kw, 'weights_only': False}
            )
            model_path = os.environ.get('MARKER_MODEL_PATH',
                                        os.environ.get('SCOL_MODEL_PATH',
                                                       'models/model_point4.pt'))
            if os.path.exists(model_path):
                _marker_model = YOLO(model_path)
                logger.info(f"Marker model loaded: {model_path}")
            else:
                logger.warning(f"Marker model bulunamadı: {model_path}")
        except Exception as e:
            logger.error(f"Marker model load failed: {e}")
        finally:
            torch.load = _orig_load
    return _marker_model

# ─── Anatomik Atama ─────────────────────────────────────────

def assign_anatomical_positions(points):
    """
    9 markerı anatomik konumlarına göre otomatik ata.

    Hasta dik durmalı.

    Args:
        points: list of (x, y) — model tarafından tespit edilen marker merkezleri
    Returns:
        dict: {anatomik_isim: (x, y)} veya None (yetersiz nokta)
    """
    if len(points) < REQUIRED_MARKERS:
        return None

    # En çok 9 marker al (fazlaysa Y'ye göre yayılan ana 9'u seç)
    pts = sorted(points, key=lambda p: p[1])  # Y'ye göre sırala
    if len(pts) > REQUIRED_MARKERS:
        # Çok fazla marker → en güvenli 9'unu al (TODO: konfigürasyon)
        # Şimdilik ilk 9'u alıyoruz (en üstten 9 nokta)
        pts = pts[:REQUIRED_MARKERS]

    # Y'ye göre 3 gruba böl: üst (3), orta (3), alt (3)
    upper = sorted(pts[0:3], key=lambda p: p[0])  # X'e göre sırala (sol→sağ)
    middle = sorted(pts[3:6], key=lambda p: p[1])  # Y'ye göre (üst→alt)
    lower = sorted(pts[6:9], key=lambda p: p[0])  # X'e göre (sol→sağ)

    # ─── ÜST GRUP (omuz kuşağı) ─────────────────────────────
    # 3 nokta: 2 omuz (sağ-sol) + 1 T1 (orta)
    # T1 = X'i orta olan, akromion'lar = ekstrem X'lerde
    # Sıralı: [en sol X, orta X, en sağ X]
    # Kameraya göre: sol=patient'ın sağı, sağ=patient'ın solu
    upper_xs = [p[0] for p in upper]
    upper_x_min, upper_x_mid, upper_x_max = upper_xs

    # T1: 3 noktanın X-ortalamasına en yakın olan nokta
    upper_avg_x = sum(upper_xs) / 3
    upper_with_dist = [(p, abs(p[0] - upper_avg_x)) for p in upper]
    upper_with_dist.sort(key=lambda x: x[1])
    t1 = upper_with_dist[0][0]

    # Diğer 2 nokta omuzlar
    shoulders = [p for p in upper if p != t1]
    shoulders.sort(key=lambda p: p[0])  # X küçük (kamera solu) → sağ omuz
    right_acromion = shoulders[0]  # Kamera solu = hastanın sağı
    left_acromion = shoulders[1]   # Kamera sağı = hastanın solu

    # ─── ORTA GRUP (omurga apex'leri) ───────────────────────
    # 3 nokta: yukarıdan aşağıya T-apex, TL-apex, L-apex
    t_apex = middle[0]
    tl_apex = middle[1]
    l_apex = middle[2]

    # ─── ALT GRUP (pelvik kuşak) ────────────────────────────
    # 3 nokta: 2 PSIS (sağ-sol) + 1 L5 (orta)
    lower_xs = [p[0] for p in lower]
    lower_avg_x = sum(lower_xs) / 3
    lower_with_dist = [(p, abs(p[0] - lower_avg_x)) for p in lower]
    lower_with_dist.sort(key=lambda x: x[1])
    l5 = lower_with_dist[0][0]

    psis_pts = [p for p in lower if p != l5]
    psis_pts.sort(key=lambda p: p[0])
    right_psis = psis_pts[0]  # Kamera solu
    left_psis = psis_pts[1]   # Kamera sağı

    return {
        't1': t1,
        'right_acromion': right_acromion,
        'left_acromion': left_acromion,
        't_apex': t_apex,
        'tl_apex': tl_apex,
        'l_apex': l_apex,
        'l5': l5,
        'right_psis': right_psis,
        'left_psis': left_psis,
    }

# ─── Açı Hesaplamaları ──────────────────────────────────────

def angle_between_segments(p1, p2, p3, p4):
    """
    İki segment (p1-p2 ve p3-p4) arasındaki açıyı hesapla.
    Sonuç: 0-90 derece arası mutlak açı.
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_a)
    angle_deg = math.degrees(angle_rad)
    
    # 0-90 aralığına döndür
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg

def angle_label(angle):
    if angle > ANGLE_HIGH:
        return "KRİTİK"
    elif angle > ANGLE_MID:
        return "ORTA"
    return "NORMAL"

def angle_color(angle):
    if angle > ANGLE_HIGH:
        return COLOR_RED
    elif angle > ANGLE_MID:
        return COLOR_BLUE
    return COLOR_GREEN

# ─── Schroth Analiz ─────────────────────────────────────────

def analyze_markers(frame):
    """
    Frame üzerinde marker tespiti yap, anatomik atama yap,
    Schroth açılarını hesapla.
    """
    model = get_marker_model()
    if model is None:
        return None

    tmp_path = None
    try:
        # Geçici dosya
        tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        cv2.imwrite(tmp_path, frame)

        # Model inference
        results = model(tmp_path, verbose=False)
        all_points = []
        for r in results:
            if r.boxes is None:
                continue
            boxes_xywh = r.boxes.xywh.cpu().numpy()
            for box in boxes_xywh:
                cx, cy = float(box[0]), float(box[1])
                all_points.append((cx, cy))

        if len(all_points) < REQUIRED_MARKERS:
            logger.debug(f"Yetersiz marker: {len(all_points)}/{REQUIRED_MARKERS}")
            return {
                'detected_markers': len(all_points),
                'required_markers': REQUIRED_MARKERS,
                'status': 'insufficient_markers',
                'detected_points': [(int(p[0]), int(p[1])) for p in all_points],
            }

        # Anatomik atama
        anatomy = assign_anatomical_positions(all_points)
        if anatomy is None:
            return {
                'detected_markers': len(all_points),
                'status': 'assignment_failed',
            }

        # ─── Cobb açıları ─────────────────────────────────────
        # Torakal Cobb: T1-TApex segmenti vs TApex-TLApex segmenti
        thoracic_cobb = angle_between_segments(
            anatomy['t1'], anatomy['t_apex'],
            anatomy['t_apex'], anatomy['tl_apex']
        )
        # Torakolomber Cobb
        tl_cobb = angle_between_segments(
            anatomy['t_apex'], anatomy['tl_apex'],
            anatomy['tl_apex'], anatomy['l_apex']
        )
        # Lumbar Cobb
        lumbar_cobb = angle_between_segments(
            anatomy['tl_apex'], anatomy['l_apex'],
            anatomy['l_apex'], anatomy['l5']
        )

        # ─── Klinik metrikler ────────────────────────────────
        # Omuz asimetrisi (Y farkı, +ise sol yüksek)
        shoulder_diff_y = anatomy['left_acromion'][1] - anatomy['right_acromion'][1]
        shoulder_dx = abs(anatomy['left_acromion'][0] - anatomy['right_acromion'][0]) or 1
        shoulder_angle = math.degrees(math.atan2(shoulder_diff_y, shoulder_dx))

        # Pelvik tilt (PSIS Y farkı)
        pelvis_diff_y = anatomy['left_psis'][1] - anatomy['right_psis'][1]
        pelvis_dx = abs(anatomy['left_psis'][0] - anatomy['right_psis'][0]) or 1
        pelvic_tilt = math.degrees(math.atan2(pelvis_diff_y, pelvis_dx))

        # Lateral trunk shift: T1 X - PSIS merkezi X
        psis_center_x = (anatomy['left_psis'][0] + anatomy['right_psis'][0]) / 2
        lateral_shift = anatomy['t1'][0] - psis_center_x

        # Pelvis genişliği (% normalleştirme için)
        pelvis_width = abs(anatomy['left_psis'][0] - anatomy['right_psis'][0]) or 1
        lateral_shift_pct = (lateral_shift / pelvis_width) * 100

        # ─── Skolyoz sınıflaması ─────────────────────────────
        # 3c: tek major (genelde torakal)
        # 4c: double major (torakal + lumbar)
        # +p: pelvik kayma > %5
        thoracic_dominant = thoracic_cobb > tl_cobb and thoracic_cobb > lumbar_cobb
        double_major = (thoracic_cobb > 10 and lumbar_cobb > 10)
        has_pelvic_shift = abs(lateral_shift_pct) > 5

        if double_major:
            curve_type = '4cp' if has_pelvic_shift else '4c'
            dominant_curve = 'thoracic+lumbar'
        else:
            curve_type = '3cp' if has_pelvic_shift else '3c'
            dominant_curve = 'thoracic' if thoracic_dominant else 'lumbar'

        # RAB nefes tarafı: konkav taraf
        # Eğer torakal eğri sağa konveks ise → konkav sol → RAB sol
        # Apex X'i merkezden sağda mı solda mı?
        spine_center_x = (anatomy['t1'][0] + anatomy['l5'][0]) / 2
        t_apex_offset = anatomy['t_apex'][0] - spine_center_x
        rab_side = 'left' if t_apex_offset > 0 else 'right'  # sağa konveks → sol konkav

        # ─── Görüntüye çiz ───────────────────────────────────
        image = frame.copy()
        # Omurga hattı
        spine_pts = [anatomy['t1'], anatomy['t_apex'],
                     anatomy['tl_apex'], anatomy['l_apex'], anatomy['l5']]
        for i in range(len(spine_pts) - 1):
            cv2.line(image,
                     (int(spine_pts[i][0]), int(spine_pts[i][1])),
                     (int(spine_pts[i+1][0]), int(spine_pts[i+1][1])),
                     COLOR_YELLOW, 3, cv2.LINE_AA)

        # Omuz çizgisi
        cv2.line(image,
                 (int(anatomy['right_acromion'][0]), int(anatomy['right_acromion'][1])),
                 (int(anatomy['left_acromion'][0]), int(anatomy['left_acromion'][1])),
                 angle_color(abs(shoulder_angle)), 2, cv2.LINE_AA)

        # PSIS çizgisi
        cv2.line(image,
                 (int(anatomy['right_psis'][0]), int(anatomy['right_psis'][1])),
                 (int(anatomy['left_psis'][0]), int(anatomy['left_psis'][1])),
                 angle_color(abs(pelvic_tilt)), 2, cv2.LINE_AA)

        # Marker noktaları (renkli, etiketli)
        marker_colors = {
            't1': (255, 200, 0),
            'right_acromion': (0, 100, 255),
            'left_acromion': (0, 100, 255),
            't_apex': (255, 255, 0),
            'tl_apex': (255, 255, 0),
            'l_apex': (255, 255, 0),
            'l5': (255, 200, 0),
            'right_psis': (0, 255, 100),
            'left_psis': (0, 255, 100),
        }
        marker_labels = {
            't1': 'T1',
            'right_acromion': 'R-A',
            'left_acromion': 'L-A',
            't_apex': 'T-Ap',
            'tl_apex': 'TL-Ap',
            'l_apex': 'L-Ap',
            'l5': 'L5',
            'right_psis': 'R-P',
            'left_psis': 'L-P',
        }
        for name, pt in anatomy.items():
            x, y = int(pt[0]), int(pt[1])
            color = marker_colors.get(name, COLOR_WHITE)
            cv2.circle(image, (x, y), 8, COLOR_WHITE, -1)
            cv2.circle(image, (x, y), 8, COLOR_BLACK, 2)
            cv2.circle(image, (x, y), 5, color, -1)
            label = marker_labels.get(name, name)
            cv2.putText(image, label, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Cobb açıları üst kutu
        info_lines = [
            f"T:  {thoracic_cobb:5.1f}d  [{angle_label(thoracic_cobb)}]",
            f"TL: {tl_cobb:5.1f}d  [{angle_label(tl_cobb)}]",
            f"L:  {lumbar_cobb:5.1f}d  [{angle_label(lumbar_cobb)}]",
            f"Tip: {curve_type} | RAB: {rab_side}",
        ]
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (340, 130), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(image, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Base64 encode
        _, buffer = cv2.imencode('.jpg', image)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        max_cobb = max(thoracic_cobb, tl_cobb, lumbar_cobb)

        return {
            'status': 'ok',
            'detected_markers': len(all_points),
            'required_markers': REQUIRED_MARKERS,

            # Schroth açıları (gerçek 2D Cobb proxy)
            'angles': {
                'thoracic':      round(thoracic_cobb, 2),
                'thoracolumbar': round(tl_cobb, 2),
                'lumbar':        round(lumbar_cobb, 2),
            },
            'labels': {
                'thoracic':      angle_label(thoracic_cobb),
                'thoracolumbar': angle_label(tl_cobb),
                'lumbar':        angle_label(lumbar_cobb),
            },
            'severity': angle_label(max_cobb),
            'max_angle': round(max_cobb, 1),

            # Klinik metrikler
            'shoulder_angle': round(shoulder_angle, 1),
            'pelvic_tilt': round(pelvic_tilt, 1),
            'lateral_shift_px': round(lateral_shift, 1),
            'lateral_shift_pct': round(lateral_shift_pct, 1),

            # Sınıflama
            'curve_type': curve_type,
            'dominant_curve': dominant_curve,
            'rab_side': rab_side,

            # Anatomik noktalar (UI için)
            'anatomy': {k: [round(v[0], 1), round(v[1], 1)] for k, v in anatomy.items()},
            'result_image_b64': img_b64,
        }

    except Exception as e:
        logger.error(f"Marker analiz hatası: {e}", exc_info=True)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
