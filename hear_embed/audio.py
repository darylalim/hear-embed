"""Audio loading, resampling, and windowing for the HeAR pipeline.

HeAR consumes fixed 2-second, 16 kHz, mono clips (exactly 32,000 samples). This
module turns an arbitrary-length recording into a batch of such clips. It has no
torch/model dependency, so the windowing logic is cheap to unit-test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # Hz; HeAR's required input rate.
CLIP_DURATION = 2  # seconds.
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32,000 samples per clip.

# Containers libsndfile (via soundfile) reads reliably. mp3/m4a depend on the
# libsndfile build; convert to wav/flac first, or load with librosa, if needed.
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


def load_and_resample(path: str | Path) -> np.ndarray:
    """Loads an audio file as a mono, 16 kHz, float32 waveform in [-1, 1].

    soundfile returns float already scaled from the source PCM format, which
    sidesteps the int16 -> /2**15 scaling bug that silently corrupts embeddings
    when raw integer samples are fed to the model. Mono-mixing and resampling
    mirror Google's reference helper (mean across channels, then
    ``scipy.signal.resample``) but without its torch import, so the data path
    stays torch-free.

    soundfile and scipy are imported here rather than at module load so the
    windowing logic stays usable with only numpy installed.
    """
    import soundfile as sf
    from scipy import signal

    # always_2d -> (frames, channels) for both mono and multi-channel inputs.
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        new_count = int(round(mono.shape[0] * (SAMPLE_RATE / source_rate)))
        mono = signal.resample(mono, new_count)
    return np.asarray(mono, dtype=np.float32)


def window_audio(
    audio: np.ndarray,
    clip_length: int = CLIP_LENGTH,
    overlap: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Slices a 1-D waveform into fixed-length clips.

    Windows step forward by ``clip_length * (1 - overlap)`` samples and the final
    clip is zero-padded so every clip is exactly ``clip_length`` long (the model
    requires it, and equal lengths let us batch them into one array).

    Args:
        audio: 1-D float waveform (already mono, 16 kHz).
        clip_length: Samples per clip (32,000 for HeAR).
        overlap: Fractional overlap between consecutive clips, in [0, 1).

    Returns:
        ``(clips, offsets)`` where ``clips`` is a ``(n, clip_length)`` float32
        array and ``offsets[i]`` is the start sample of clip ``i`` in ``audio``.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    n = audio.shape[0]
    if n == 0:
        return np.empty((0, clip_length), dtype=np.float32), []

    step = clip_length - int(round(clip_length * overlap))
    step = max(1, step)  # guard against overlap rounding to a zero-length step.

    clips: list[np.ndarray] = []
    offsets: list[int] = []
    start = 0
    while True:
        clip = audio[start : start + clip_length]
        pad = clip_length - clip.shape[0]
        if pad > 0:
            clip = np.pad(clip, (0, pad))
        clips.append(clip)
        offsets.append(start)
        # Stop once this window already reaches the end of the recording, so we
        # don't emit a redundant, mostly-padding trailing clip.
        if start + clip_length >= n:
            break
        start += step

    return np.stack(clips), offsets


def iter_audio_files(
    root: str | Path,
    extensions: tuple[str, ...] = AUDIO_EXTENSIONS,
) -> list[Path]:
    """Returns audio files under ``root`` (a single file or a directory tree)."""
    root = Path(root)
    if root.is_file():
        return [root]
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
