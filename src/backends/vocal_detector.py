# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Lightweight vocal-activity detector for instrumental generation quality control.

Uses only scipy + numpy (already available as transitive dependencies of
stable-audio-tools) to estimate whether a rendered WAV contains human
vocal sounds.  Returns a confidence score in [0.0, 1.0] or None when the
detection cannot run (missing deps, unreadable file, too short).

The detector combines two signals known to distinguish human voice
from instruments:

1. **Spectral formant pattern** – human vocal tracts produce a
   descending formant energy profile (F1 > F2 > F3 in the 300–2500 Hz
   band).  Guitar harmonics and synth lead lines don't replicate this.

2. **Temporal modulation** – speech and singing have energy modulated at
   3–8 Hz (the typical phonetic/syllable rate).  This shows up as a
   distinct autocorrelation peak that instruments rarely produce.

The final score is a weighted blend of the two; a threshold comparison
decides whether to retry the generation with a different seed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_wav(path: Path) -> Optional[tuple]:
    """Load a PCM WAV file as mono float32 samples + sample rate.

    Returns ``(samples, sample_rate)`` or ``None`` on failure.
    """
    try:
        import numpy as np
        from scipy import signal as sig  # noqa: F401  -- imported for availability
    except ImportError as exc:
        logger.debug("vocal_detector: numpy/scipy unavailable: %s", exc)
        return None

    try:
        import wave as _wave

        with _wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
            samples /= 128.0
        elif sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            samples /= 32768.0
        else:
            return None

        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        if framerate == 0 or len(samples) < 4096:
            return None

        return samples, framerate
    except Exception as exc:
        logger.debug("vocal_detector: failed to load %s: %s", path, exc)
        return None


def detect_vocal_activity(
    wav_path: Path,
    threshold: float = 0.55,
) -> Optional[bool]:
    """Detect whether *wav_path* contains human vocal sounds.

    Args:
        wav_path: Path to a PCM WAV file.
        threshold: Confidence score above which vocals are declared present.
            Range 0.0–1.0.

    Returns:
        ``True`` if vocals are detected, ``False`` if not, ``None`` if the
        detection could not be performed (missing dependencies, unreadable
        file, clip too short).
    """
    loaded = _load_wav(wav_path)
    if loaded is None:
        return None

    samples, framerate = loaded

    try:
        import numpy as np
        from scipy import signal as sig
    except ImportError:
        return None

    # --- Spectral formant analysis ---
    nperseg = min(8192, len(samples) // 4)
    noverlap = min(4096, len(samples) // 8)
    if nperseg < 256 or noverlap >= nperseg:
        return None

    f, psd = sig.welch(samples, framerate, nperseg=nperseg, noverlap=noverlap)
    total = np.sum(psd)
    if total <= 0:
        return False

    # Human vocal formant bands (F1-F3 in the speech-frequency range).
    f1 = np.sum(psd[(f > 300) & (f < 800)]) / total
    f2 = np.sum(psd[(f > 800) & (f < 1500)]) / total
    f3 = np.sum(psd[(f > 1500) & (f < 2500)]) / total

    # Formant pattern: vocals have significant energy in ALL three bands
    # simultaneously (the vocal tract shape produces multiple formants).
    # A sine wave or single-harmonic instrument triggers only F1 — requiring
    # F2 and F3 above small thresholds eliminates most false positives.
    has_formant_pattern = f1 > 0.03 and f2 > 0.01 and f3 > 0.003

    # --- Temporal modulation analysis ---
    frame_len = int(0.025 * framerate)
    hop = int(0.010 * framerate)
    if hop < 1:
        hop = 1
    n_frames = 1 + (len(samples) - frame_len) // hop
    if n_frames < 5:
        return None

    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        seg = samples[i * hop : i * hop + frame_len]
        if len(seg) < frame_len:
            break
        envelope[i] = np.sqrt(np.mean(seg ** 2))

    env = envelope[:n_frames]
    if len(env) > 4:
        env = env - env.mean()
        ac = np.correlate(env, env, mode="full")[len(env) - 1:]
        ac0 = float(ac[0])
        # Guard: if the envelope has near-zero variance (constant amplitude
        # like a pure sine wave), the autocorrelation is dominated by noise
        # and the normalized peak is meaningless — treat as no modulation.
        if ac0 > 1e-10:
            ac /= ac0
            min_lag = max(1, int(framerate / 8.0 / hop))
            max_lag = min(len(ac) - 1, int(framerate / 2.0 / hop))
            mod_peak = float(np.max(ac[min_lag:max_lag])) if max_lag > min_lag else 0.0
        else:
            mod_peak = 0.0
    else:
        mod_peak = 0.0

    # Speech modulation: 3-8 Hz peak.
    has_speech_mod = mod_peak > 0.4

    # Require BOTH formant pattern AND speech modulation for a high
    # confidence vocal detection.  Formant pattern alone is not
    # sufficient — sine waves and guitar harmonics can trigger it
    # without the temporal modulation characteristic of human speech.
    if has_formant_pattern and has_speech_mod:
        score = 0.7
    elif has_formant_pattern or has_speech_mod:
        score = 0.3
    else:
        score = 0.0

    return score >= threshold
