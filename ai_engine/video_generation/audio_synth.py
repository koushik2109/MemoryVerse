"""
Procedural Ambient Audio Synthesizer
Generates 4-chord ambient harmonic background music using pure NumPy waveforms
with ADSR envelopes, warm lowpass harmonics, and WAV export.
"""
import io
import math
import struct
import wave
from typing import List, Dict, Any, Optional
import numpy as np


# Scale frequencies in Hz (Root: C4 = 261.63 Hz)
NOTE_FREQS = {
    "C3": 130.81, "E3": 164.81, "G3": 196.00, "B3": 246.94,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.25,
}

# 4-Chord Progression Presets (4 bars per cycle)
CHORD_PROGRESSIONS = {
    "calm": [["C4", "E4", "G4"], ["G3", "B3", "D4"], ["A3", "C4", "E4"], ["F3", "A3", "C4"]],  # I - V - vi - IV
    "energetic": [["A3", "C4", "E4"], ["F3", "A3", "C4"], ["C4", "E4", "G4"], ["G3", "B3", "D4"]],  # vi - IV - I - V
    "nostalgic": [["C4", "E4", "G4", "B4"], ["A3", "C4", "E4", "G4"], ["F3", "A3", "C4", "E4"], ["G3", "B3", "D4", "F4"]],
    "joyful": [["C4", "E4", "G4"], ["F3", "A3", "C4"], ["G3", "B3", "D4"], ["C4", "E4", "G4"]],
    "dramatic": [["A3", "C4", "E4"], ["D4", "F4", "A4"], ["F3", "A3", "C4"], ["E3", "G#3", "B3"]],
    "serene": [["F3", "A3", "C4"], ["G3", "B3", "D4"], ["E3", "G3", "B3"], ["A3", "C4", "E4"]],
    "neutral": [["C4", "E4", "G4"], ["A3", "C4", "E4"], ["F3", "A3", "C4"], ["G3", "B3", "D4"]],
    "acoustic": [["G3", "B3", "D4"], ["D4", "F#4", "A4"], ["E4", "G4", "B4"], ["C4", "E4", "G4"]],
    "warm": [["C4", "E4", "G4"], ["A3", "C4", "E4"], ["F3", "A3", "C4"], ["G3", "B3", "D4"]],
    "emotional": [["A3", "C4", "E4"], ["F3", "A3", "C4"], ["C4", "E4", "G4"], ["E3", "G#3", "B3"]],
    "lofi": [["C4", "E4", "G4", "B4"], ["A3", "C4", "E4", "G4"], ["D4", "F4", "A4", "C5"], ["G3", "B3", "D4", "F4"]],
}


def generate_chord_tone(freq: float, duration_sec: float, sample_rate: int = 44100) -> np.ndarray:
    """Generates a warm synthesized pad tone with harmonic overtones and ADSR envelope."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, False)

    # Fundamental + 2nd harmonic + 3rd harmonic for warm analog pad feel
    signal = 0.6 * np.sin(2 * np.pi * freq * t)
    signal += 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
    signal += 0.15 * np.sin(2 * np.pi * (freq * 3) * t)

    # ADSR Envelope: 15% Attack, 15% Decay, 50% Sustain (0.8), 20% Release
    attack_len = int(0.15 * n_samples)
    decay_len = int(0.15 * n_samples)
    release_len = int(0.20 * n_samples)
    sustain_len = n_samples - (attack_len + decay_len + release_len)

    envelope = np.ones(n_samples, dtype=np.float32)
    if attack_len > 0:
        envelope[:attack_len] = np.linspace(0, 1.0, attack_len)
    if decay_len > 0:
        envelope[attack_len:attack_len + decay_len] = np.linspace(1.0, 0.8, decay_len)
    envelope[attack_len + decay_len:attack_len + decay_len + sustain_len] = 0.8
    if release_len > 0:
        envelope[-release_len:] = np.linspace(0.8, 0.0, release_len)

    return signal * envelope


def synthesize_ambient_soundtrack(
    mood: Any = "calm",
    duration_seconds: float = 30.0,
    sample_rate: int = 44100,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Synthesizes a complete looping 4-chord ambient soundtrack.
    Optionally saves to WAV file path.
    """
    if hasattr(mood, "get_audio_mood"):
        mood_str = str(getattr(mood, "get_audio_mood")())
    elif hasattr(mood, "overall_mood"):
        mood_str = str(getattr(mood, "overall_mood"))
    elif hasattr(mood, "dominant_emotion"):
        mood_str = str(getattr(mood, "dominant_emotion"))
    else:
        mood_str = str(mood)

    progression = CHORD_PROGRESSIONS.get(mood_str.lower(), CHORD_PROGRESSIONS["calm"])
    chord_duration = 3.5  # seconds per chord
    cycle_duration = chord_duration * len(progression)
    n_cycles = int(np.ceil(duration_seconds / cycle_duration))

    track_samples = int(duration_seconds * sample_rate)
    full_audio = np.zeros(track_samples, dtype=np.float32)

    current_idx = 0
    for cycle in range(n_cycles):
        for chord in progression:
            chord_signal = np.zeros(int(chord_duration * sample_rate), dtype=np.float32)
            for note in chord:
                freq = NOTE_FREQS.get(note, 261.63)
                tone = generate_chord_tone(freq, chord_duration, sample_rate)
                chord_signal += tone / len(chord)

            end_idx = min(track_samples, current_idx + len(chord_signal))
            slice_len = end_idx - current_idx
            if slice_len > 0:
                full_audio[current_idx:end_idx] += chord_signal[:slice_len]
            current_idx += int(chord_duration * sample_rate)
            if current_idx >= track_samples:
                break
        if current_idx >= track_samples:
            break

    # Normalize audio to peak at -3dB (0.7)
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = (full_audio / peak) * 0.7

    if output_path:
        write_wav_file(output_path, full_audio, sample_rate)

    return full_audio


def write_wav_file(file_path: str, audio: np.ndarray, sample_rate: int = 44100):
    """Encodes float32 numpy array into 16-bit PCM WAV file."""
    int16_audio = np.int16(audio * 32767)
    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_audio.tobytes())
