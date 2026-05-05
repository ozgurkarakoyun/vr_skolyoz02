/**
 * audio_engine.js
 * ─────────────────────────────────────────────────────────────
 * Schroth VR / TV — Türkçe ses geri bildirim motoru
 *
 * Öncelik sırası:
 *   1) /static/voice/tr/*.mp3 veya *.wav dosyaları varsa doğal kayıtları çalar.
 *   2) Ses dosyası yoksa tarayıcının Türkçe Web Speech API sesini kullanır.
 *   3) Türkçe ses bulunamazsa yine tr-TR diliyle deneme yapar ve ekrana uyarı verir.
 *
 * Not: Web Speech API sesi cihaz/tarayıcıya bağlıdır. En doğal sonuç için
 *      static/voice/tr klasörüne profesyonel kayıt MP3/WAV dosyaları konulabilir.
 */

class SchrothAudioEngine {
  constructor() {
    this.enabled = false;
    this.ttsEnabled = false;
    this.beepEnabled = true;
    this.preferRecordedAudio = true;

    // Web Speech API
    this.synth = window.speechSynthesis || null;
    this.turkishVoice = null;
    this.voiceReady = false;
    this.voiceWarning = '';

    // AudioContext
    this.audioCtx = null;

    // Kuyruk & debounce
    this.queue = [];
    this.speaking = false;
    this.lastSpoken = {};
    this.DEBOUNCE_MS = 9000;

    // Nefes metronom
    this.breathInterval = null;
    this.breathPhase = 'idle';

    // Skor / motivasyon
    this.lastMilestone = 0;
    this.bestScore = 0;
    this.lastMotivationAt = 0;
    this.motivationPhrases = [
      'Harikasınız, böyle devam edin.',
      'Çok güzel, duruşunuz belirgin şekilde iyileşiyor.',
      'Mükemmel gidiyorsunuz, aynı şekilde devam edin.',
      'Tebrikler, kontrolünüz daha iyi.',
      'Çok iyi çalışıyorsunuz, pozisyonunuzu koruyun.'
    ];

    // Opsiyonel insan sesi dosyaları. Dosyalar yoksa otomatik TTS fallback olur.
    this.recordedAudioMap = {
      session_start: '/static/voice/tr/session_start.mp3',
      excellent: '/static/voice/tr/excellent.mp3',
      continue: '/static/voice/tr/continue.mp3',
      posture_better: '/static/voice/tr/posture_better.mp3',
      very_good: '/static/voice/tr/very_good.mp3',
      breathe_in: '/static/voice/tr/breathe_in.mp3',
      hold: '/static/voice/tr/hold.mp3',
      breathe_out: '/static/voice/tr/breathe_out.mp3',
      warning: '/static/voice/tr/warning.mp3'
    };
    this.recordedCache = {};

    this._initVoice();
    this._initAudioCtx();
  }

  // ─── Init ─────────────────────────────────────────────────

  _initVoice() {
    if (!this.synth) {
      this.voiceWarning = 'Bu tarayıcı sesli okuma desteklemiyor.';
      this._emitStatus();
      return;
    }

    const scoreVoice = (v) => {
      const lang = String(v.lang || '').toLowerCase();
      const name = String(v.name || '').toLowerCase();
      let score = 0;
      if (lang === 'tr-tr') score += 100;
      else if (lang.startsWith('tr')) score += 80;
      if (name.includes('turkish') || name.includes('türk') || name.includes('turk')) score += 20;
      // Bazı sistemlerde bu isimler daha doğal duyulur.
      if (name.includes('google')) score += 15;
      if (name.includes('microsoft')) score += 12;
      if (name.includes('apple')) score += 10;
      if (name.includes('natural') || name.includes('neural') || name.includes('online')) score += 8;
      if (v.localService === false) score += 4;
      return score;
    };

    const loadVoices = () => {
      const voices = this.synth.getVoices() || [];
      const turkishVoices = voices
        .filter(v => String(v.lang || '').toLowerCase().startsWith('tr') || /turkish|türk|turk/i.test(v.name || ''))
        .sort((a, b) => scoreVoice(b) - scoreVoice(a));

      this.turkishVoice = turkishVoices[0] || null;
      this.voiceReady = true;
      this.ttsEnabled = true;

      if (this.turkishVoice) {
        this.voiceWarning = '';
        console.log('[Audio] Türkçe TTS sesi:', this.turkishVoice.name, this.turkishVoice.lang);
      } else {
        this.voiceWarning = 'Cihazda Türkçe ses bulunamadı. Türkçe ses için Chrome/Edge kullanın veya cihaza Türkçe ses paketi yükleyin.';
        console.warn('[Audio] Türkçe ses bulunamadı. tr-TR diliyle varsayılan TTS deneniyor.');
      }
      this._emitStatus();
    };

    // Bazı tarayıcılarda ses listesi geç gelir.
    loadVoices();
    this.synth.addEventListener?.('voiceschanged', loadVoices);
    setTimeout(loadVoices, 600);
    setTimeout(loadVoices, 1800);
  }

  _initAudioCtx() {
    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch(e) {
      console.warn('[Audio] AudioContext not available');
    }
  }

  _emitStatus() {
    window.dispatchEvent(new CustomEvent('schrothAudioStatus', { detail: this.getStatus() }));
  }

  // ─── Public API ───────────────────────────────────────────

  enable() {
    this.enabled = true;
    if (this.audioCtx?.state === 'suspended') this.audioCtx.resume();
    if (this.synth?.paused) this.synth.resume();
    this.beep('start');
    setTimeout(() => this._speak('Sesli Türkçe yönlendirme açıldı.', 'high', 'continue'), 180);
    this._emitStatus();
  }

  disable() {
    this.enabled = false;
    this.stopBreath();
    if (this.synth) this.synth.cancel();
    this.queue = [];
    this.speaking = false;
    this._emitStatus();
  }

  toggle() {
    this.enabled ? this.disable() : this.enable();
    return this.enabled;
  }

  update(analysis, prevAnalysis) {
    if (!this.enabled || !analysis) return;

    const score = analysis.score || 0;
    const scol = analysis.scoliosis;
    const phase = analysis.phase;

    this._checkScoreMilestone(score);
    this._checkMotivation(score);
    if (scol) this._checkCriticalAngle(scol);
    this._speakPostureInstruction(analysis);

    if (phase && prevAnalysis?.phase && phase.name !== prevAnalysis.phase.name) {
      this._onPhaseChange(phase);
    }

    if (phase?.id === 'rab_breathing') this._startBreathMetronome();
    else this._stopBreathMetronome();
  }

  speakSessionStart() {
    this._speak('Seans başladı. Hazırsanız başlayalım.', 'high', 'session_start');
  }

  // ─── Skor Milestone ───────────────────────────────────────

  _checkScoreMilestone(score) {
    const milestones = [50, 70, 85];
    for (const m of milestones) {
      if (score >= m && this.lastMilestone < m) {
        this.lastMilestone = m;
        if (m === 85) {
          this.beep('milestone_high');
          this._speak('Harika. Mükemmel pozisyon.', 'high', 'excellent');
        } else if (m === 70) {
          this.beep('milestone_mid');
          this._speak('Çok iyi, devam edin.', 'normal', 'very_good');
        } else {
          this.beep('milestone_low');
        }
        return;
      }
    }
    if (score < 45 && this.lastMilestone > 0) {
      this.lastMilestone = 0;
    }
  }

  _checkMotivation(score) {
    const now = Date.now();
    if (score >= 70 && score >= this.bestScore + 5 && now - this.lastMotivationAt > 20000) {
      this.bestScore = score;
      this.lastMotivationAt = now;
      const text = this.motivationPhrases[Math.floor(Math.random() * this.motivationPhrases.length)];
      this.beep('milestone_mid');
      const key = text.includes('duruşunuz') ? 'posture_better' : text.includes('Mükemmel') ? 'excellent' : 'continue';
      this._speak(text, 'high', key);
    } else if (score > this.bestScore) {
      this.bestScore = score;
    }
  }

  // ─── Uyarılar / talimatlar ────────────────────────────────

  _checkCriticalAngle(scol) {
    if (scol.severity === 'KRİTİK' && scol.max_angle > 40) {
      this._speak(`Dikkat. Eğrilik değeri ${scol.max_angle.toFixed(0)} dereceye yükseldi. Pozisyonu kontrol edin.`, 'urgent', 'warning');
      this.beep('warning');
    }
  }

  _speakPostureInstruction(analysis) {
    const instructions = analysis.instructions || [];
    if (!instructions.length) return;

    const raw = instructions[0];
    const clean = this._cleanTurkishText(raw);
    if (!clean) return;

    const now = Date.now();
    const last = this.lastSpoken[clean] || 0;
    if (now - last < this.DEBOUNCE_MS) return;

    this._speak(clean, 'normal');
  }

  _onPhaseChange(phase) {
    this.beep('phase_change');
    const text = this._cleanTurkishText(phase.name_tr || phase.name || 'Yeni faz');
    setTimeout(() => {
      this._speak(text, 'high');
      if (phase.instruction_tr) {
        setTimeout(() => this._speak(this._cleanTurkishText(phase.instruction_tr), 'normal'), 1600);
      }
    }, 350);
  }

  // ─── Nefes Metronom ───────────────────────────────────────

  _startBreathMetronome() {
    if (this.breathInterval) return;
    const sequence = [
      { phase: 'in',   duration: 4000, text: 'Nefes alın.', key: 'breathe_in' },
      { phase: 'hold', duration: 1000, text: 'Tutun.', key: 'hold' },
      { phase: 'out',  duration: 6000, text: 'Nefes verin.', key: 'breathe_out' },
      { phase: 'rest', duration: 1000, text: null, key: null },
    ];
    let idx = 0;

    const tick = () => {
      const step = sequence[idx % sequence.length];
      this.breathPhase = step.phase;
      if (step.text) {
        this.beep(step.phase === 'in' ? 'breath_in' : step.phase === 'out' ? 'breath_out' : 'breath_hold');
        if (idx % 3 === 0) this._speak(step.text, 'breath', step.key);
      }
      window.dispatchEvent(new CustomEvent('breathPhase', { detail: step }));
      idx++;
    };

    tick();
    const scheduleNext = () => {
      const step = sequence[idx % sequence.length];
      this.breathInterval = setTimeout(() => { tick(); scheduleNext(); }, step.duration);
    };
    scheduleNext();
  }

  _stopBreathMetronome() {
    if (!this.breathInterval) return;
    clearTimeout(this.breathInterval);
    this.breathInterval = null;
    this.breathPhase = 'idle';
    window.dispatchEvent(new CustomEvent('breathPhase', { detail: { phase: 'idle' } }));
  }

  startBreath() { this._startBreathMetronome(); }
  stopBreath() { this._stopBreathMetronome(); }

  // ─── Seslendirme ──────────────────────────────────────────

  _speak(text, priority = 'normal', audioKey = null) {
    if (!this.enabled) return;
    const clean = this._cleanTurkishText(text);
    if (!clean) return;

    const now = Date.now();
    this.lastSpoken[clean] = now;

    // Varsa doğal kayıtları çal. Yoksa TTS'e düş.
    if (this.preferRecordedAudio && audioKey && this.recordedAudioMap[audioKey]) {
      this._playRecorded(audioKey)
        .then((played) => { if (!played) this._queueTTS(clean, priority); })
        .catch(() => this._queueTTS(clean, priority));
    } else {
      this._queueTTS(clean, priority);
    }
  }

  async _playRecorded(audioKey) {
    const url = this.recordedAudioMap[audioKey];
    if (!url) return false;

    try {
      let audio = this.recordedCache[audioKey];
      if (!audio) {
        audio = new Audio(url);
        audio.preload = 'auto';
        this.recordedCache[audioKey] = audio;
      }
      audio.pause();
      audio.currentTime = 0;
      await audio.play();
      return true;
    } catch(e) {
      // Dosya yoksa veya tarayıcı otomatik oynatmayı engellerse sessizce TTS'e düş.
      return false;
    }
  }

  _queueTTS(text, priority = 'normal') {
    if (!this.ttsEnabled || !this.synth) return;

    const utterance = new SpeechSynthesisUtterance(text);
    if (this.turkishVoice) utterance.voice = this.turkishVoice;
    utterance.lang = 'tr-TR';

    // Daha doğal ve anlaşılır Türkçe için hız/pitch düşürüldü.
    if (priority === 'urgent') {
      utterance.rate = 0.88;
      utterance.pitch = 0.95;
      utterance.volume = 1.0;
    } else if (priority === 'breath') {
      utterance.rate = 0.78;
      utterance.pitch = 0.92;
      utterance.volume = 0.72;
    } else if (priority === 'high') {
      utterance.rate = 0.82;
      utterance.pitch = 0.96;
      utterance.volume = 1.0;
    } else {
      utterance.rate = 0.80;
      utterance.pitch = 0.94;
      utterance.volume = 0.95;
    }

    if (priority === 'urgent') {
      this.synth.cancel();
      this.queue = [];
      this.speaking = false;
      this.synth.speak(utterance);
    } else {
      this.queue.push(utterance);
      if (!this.speaking) this._processQueue();
    }
  }

  _processQueue() {
    if (!this.queue.length) { this.speaking = false; return; }
    this.speaking = true;
    const utt = this.queue.shift();
    utt.onend = () => setTimeout(() => this._processQueue(), 450);
    utt.onerror = () => setTimeout(() => this._processQueue(), 250);
    this.synth.speak(utt);
  }

  // ─── Beep Sesleri ─────────────────────────────────────────

  beep(type = 'tick') {
    if (!this.enabled || !this.beepEnabled || !this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

    const ctx = this.audioCtx;
    const now = ctx.currentTime;
    const configs = {
      start:          { freq: [440, 550, 660], dur: 0.10, vol: 0.20, type: 'sine' },
      phase_change:   { freq: [330, 440],      dur: 0.12, vol: 0.25, type: 'sine' },
      milestone_low:  { freq: [523],           dur: 0.12, vol: 0.20, type: 'sine' },
      milestone_mid:  { freq: [523, 659],      dur: 0.12, vol: 0.23, type: 'sine' },
      milestone_high: { freq: [523, 659, 784], dur: 0.16, vol: 0.28, type: 'sine' },
      warning:        { freq: [220, 220],      dur: 0.22, vol: 0.35, type: 'sine' },
      breath_in:      { freq: [330],           dur: 0.08, vol: 0.13, type: 'sine' },
      breath_out:     { freq: [220],           dur: 0.08, vol: 0.13, type: 'sine' },
      breath_hold:    { freq: [280],           dur: 0.06, vol: 0.10, type: 'sine' },
      tick:           { freq: [720],           dur: 0.04, vol: 0.10, type: 'sine' },
    };
    const cfg = configs[type] || configs.tick;

    cfg.freq.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = cfg.type;
      osc.frequency.setValueAtTime(freq, now + i * cfg.dur * 0.85);
      gain.gain.setValueAtTime(0.0001, now + i * cfg.dur * 0.85);
      gain.gain.linearRampToValueAtTime(cfg.vol, now + i * cfg.dur * 0.85 + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * cfg.dur * 0.85 + cfg.dur);
      osc.start(now + i * cfg.dur * 0.85);
      osc.stop(now + i * cfg.dur * 0.85 + cfg.dur + 0.05);
    });
  }

  // ─── Utils ────────────────────────────────────────────────

  _cleanTurkishText(text) {
    return String(text || '')
      .replace(/[⬇️⬆️⬅️➡️↔️↙️↘️✅👍🫁📋⚠️]/gu, '')
      .replace(/Cobb proxy/gi, 'postür ölçümü')
      .replace(/Cobb/gi, 'postür açısı')
      .replace(/T:/g, 'Torasik')
      .replace(/TL:/g, 'Torakolomber')
      .replace(/L:/g, 'Lomber')
      .replace(/\s+/g, ' ')
      .trim();
  }

  getStatus() {
    return {
      enabled: this.enabled,
      ttsEnabled: this.ttsEnabled,
      voiceReady: this.voiceReady,
      voiceName: this.turkishVoice?.name || '',
      voiceLang: this.turkishVoice?.lang || 'tr-TR',
      warning: this.voiceWarning,
      recordedAudio: this.preferRecordedAudio
    };
  }

  get isEnabled() { return this.enabled; }
  get hasTTS() { return this.ttsEnabled; }
  get breathState() { return this.breathPhase; }
}

window.schrothAudio = new SchrothAudioEngine();
