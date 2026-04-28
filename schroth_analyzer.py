"""
schroth_analyzer.py
─────────────────────────────────────────────────────────────────
Schroth Egzersiz Faz Yönetimi ve Seans Takibi

NOT: Bu dosya artık postür analizi YAPMAZ.
Tüm analiz (Cobb açıları, omuz/kalça simetrisi, eğri sınıflaması)
marker_engine.py içinde marker tabanlı yapılır.

Bu modül sadece şunları sağlar:
  - 4 fazlı Schroth egzersiz döngüsü (Elongation → Derotation → RAB → Stabilizasyon)
  - Faz zamanlaması ve geçişler
  - Tekrar (rep) sayımı
  - Seans skoru istatistikleri (best/avg)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Schroth Egzersiz Fazları ────────────────────────────────
PHASES = [
    {
        'id': 'elongation',
        'name': 'Elongasyon',
        'name_tr': '1. Faz — Boyun Uzatma',
        'duration': 30,
        'instruction_tr': 'Başınızı tavana doğru uzatın. Omurgayı dik tutun.',
        'breath_tr': 'Normal nefes alın, boyunuzu hissedin',
        'color': '#00e5ff',
    },
    {
        'id': 'derotation',
        'name': 'Derotasyon',
        'name_tr': '2. Faz — Derotasyon',
        'duration': 45,
        'instruction_tr': 'Konkav tarafı açın, konveks tarafı bastırın.',
        'breath_tr': 'Konkav tarafa doğru RAB nefesi alın',
        'color': '#7b2fff',
    },
    {
        'id': 'rab_breathing',
        'name': 'RAB Nefes',
        'name_tr': '3. Faz — RAB Nefes',
        'duration': 60,
        'instruction_tr': 'Çökmüş tarafı nefesle doldurun. 4 sayı al, 6 sayı ver.',
        'breath_tr': 'Derin — Tut — Yavaş ver',
        'color': '#2ed573',
    },
    {
        'id': 'stabilization',
        'name': 'Stabilizasyon',
        'name_tr': '4. Faz — İzometrik Stabilizasyon',
        'duration': 30,
        'instruction_tr': 'Düzeltilmiş pozisyonu 10 saniye tutun. Kasları sıkın.',
        'breath_tr': 'Nefesi tutun — Kas kontraksiyonu',
        'color': '#ffa502',
    },
]


# ─── Seans İstatistikleri ────────────────────────────────────
@dataclass
class SessionStats:
    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    valid_detections: int = 0
    scores: List[float] = field(default_factory=list)
    best_score: float = 0.0
    current_phase_idx: int = 0
    phase_start_time: float = field(default_factory=time.time)
    rep_count: int = 0


# ─── Ana Sınıf ───────────────────────────────────────────────
class SchrothAnalyzer:
    """
    Schroth seans/faz yöneticisi.

    Kullanım:
        analyzer = SchrothAnalyzer()
        analyzer.tick(score=85)             # Her frame'de çağır
        phase = analyzer.get_current_phase()
        summary = analyzer.get_session_summary()
    """

    def __init__(self):
        self.session = SessionStats()

    # ─── Tick — her frame için çağrılır ──────────────────────
    def tick(self, score: float = 0.0, valid: bool = True):
        """
        Her frame'de çağrılır. Faz geçişlerini ve istatistikleri günceller.

        Args:
            score: O frame'in skoru (0-100)
            valid: Marker tespiti başarılı mı
        """
        self.session.frame_count += 1
        if valid:
            self.session.valid_detections += 1
            self.session.scores.append(score)
            if score > self.session.best_score:
                self.session.best_score = score
        self._update_phase()

    # ─── Faz Yönetimi ────────────────────────────────────────
    def _update_phase(self):
        phase = PHASES[self.session.current_phase_idx]
        elapsed = time.time() - self.session.phase_start_time
        if elapsed >= phase['duration']:
            next_idx = (self.session.current_phase_idx + 1) % len(PHASES)
            if next_idx == 0:
                self.session.rep_count += 1
            self.session.current_phase_idx = next_idx
            self.session.phase_start_time = time.time()

    def get_current_phase(self) -> dict:
        phase = PHASES[self.session.current_phase_idx]
        elapsed = time.time() - self.session.phase_start_time
        remaining = max(0, phase['duration'] - elapsed)
        progress = min(1.0, elapsed / phase['duration'])
        return {
            **phase,
            'elapsed_sec': round(elapsed),
            'remaining_sec': round(remaining),
            'progress': round(progress, 2),
        }

    # ─── Seans Özeti ─────────────────────────────────────────
    def get_session_summary(self) -> dict:
        s = self.session
        elapsed = time.time() - s.start_time
        avg_score = sum(s.scores) / len(s.scores) if s.scores else 0
        detection_rate = s.valid_detections / max(s.frame_count, 1) * 100
        return {
            'duration_sec': round(elapsed),
            'frame_count': s.frame_count,
            'detection_rate_pct': round(detection_rate, 1),
            'avg_score': round(avg_score, 1),
            'best_score': round(s.best_score, 1),
            'rep_count': s.rep_count,
            'current_phase': PHASES[s.current_phase_idx]['name_tr'],
        }

    # ─── Sıfırlama ───────────────────────────────────────────
    def reset_session(self):
        self.session = SessionStats()
