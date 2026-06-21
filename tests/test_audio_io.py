"""Unit tests for audio I/O — load_and_resample and iter_audio_files.

These exercise the torch-free data path: reading/decoding files via soundfile,
mono-mixing, resampling with scipy, and walking a directory tree for audio.
They use real on-disk files written with ``soundfile.write`` into ``tmp_path``,
so no model or torch import is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from hear_embed.audio import (
    AUDIO_EXTENSIONS,
    SAMPLE_RATE,
    iter_audio_files,
    load_and_resample,
)


def test_load_mono_16k_returns_float32_1d_in_range(tmp_path: Path) -> None:
    # A 16 kHz mono signal should round-trip unchanged in shape/rate.
    n = SAMPLE_RATE  # 1 second
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    wav = tmp_path / "mono.wav"
    sf.write(wav, audio, SAMPLE_RATE)

    out = load_and_resample(wav)
    assert out.ndim == 1  # mono waveform, not (frames, channels)
    assert out.dtype == np.float32
    assert out.shape[0] == n  # no resampling at 16 kHz
    assert np.all(np.abs(out) <= 1.0)  # amplitude 0.5 stays in [-1, 1]
    np.testing.assert_allclose(out, audio, atol=1e-4)


def test_stereo_is_downmixed_by_channel_mean(tmp_path: Path) -> None:
    n = 1000
    left = np.full(n, 0.8, dtype=np.float32)
    right = np.full(n, -0.2, dtype=np.float32)
    # soundfile expects (frames, channels) for multichannel writes.
    stereo = np.stack([left, right], axis=1)
    wav = tmp_path / "stereo.wav"
    sf.write(wav, stereo, SAMPLE_RATE)

    out = load_and_resample(wav)
    assert out.ndim == 1  # collapsed to mono
    assert out.shape[0] == n
    # mean(0.8, -0.2) == 0.3 for every sample.
    np.testing.assert_allclose(out, 0.3, atol=1e-4)


def test_resampled_from_8k_to_16k_doubles_length(tmp_path: Path) -> None:
    src_rate = 8_000
    src_count = 4_000  # 0.5 s at 8 kHz
    t = np.linspace(0.0, 0.5, src_count, endpoint=False, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
    wav = tmp_path / "low_rate.wav"
    sf.write(wav, audio, src_rate)

    out = load_and_resample(wav)
    assert out.dtype == np.float32
    # 16 kHz / 8 kHz == 2x the source sample count (exact here, but allow rounding).
    expected = int(round(src_count * (SAMPLE_RATE / src_rate)))
    assert expected == 2 * src_count
    assert out.shape[0] == expected


def test_int16_pcm_is_scaled_to_float_not_raw_integers(tmp_path: Path) -> None:
    # Write a half-scale square wave as 16-bit PCM.
    n = 256
    audio = np.concatenate(
        [
            np.full(n // 2, 0.5, dtype=np.float32),
            np.full(n // 2, -0.5, dtype=np.float32),
        ]
    )
    wav = tmp_path / "pcm16.wav"
    sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")

    out = load_and_resample(wav)
    assert out.dtype == np.float32
    # Must come back as floats near +-0.5, NOT raw int16 (~+-16384).
    assert np.all(np.abs(out) <= 1.0)
    np.testing.assert_allclose(np.abs(out), 0.5, atol=1e-3)
    assert out[0] > 0.0 and out[-1] < 0.0  # sign/order preserved


def test_iter_single_file_returns_that_file(tmp_path: Path) -> None:
    wav = tmp_path / "one.wav"
    sf.write(wav, np.zeros(10, dtype=np.float32), SAMPLE_RATE)
    assert iter_audio_files(wav) == [wav]


def test_iter_nested_tree_sorted_and_ignores_non_audio(tmp_path: Path) -> None:
    silence = np.zeros(10, dtype=np.float32)
    # Audio files scattered across a nested directory tree.
    a = tmp_path / "a.wav"
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    b = nested / "b.flac"
    c = tmp_path / "sub" / "c.ogg"
    for p in (a, b, c):
        sf.write(p, silence, SAMPLE_RATE)
    # Non-audio files that must be ignored.
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "sub" / "data.json").write_text("{}")

    result = iter_audio_files(tmp_path)
    assert result == sorted([a, b, c])  # only audio, returned sorted
    assert result == sorted(result)  # explicitly sorted order


def test_iter_honors_custom_extensions(tmp_path: Path) -> None:
    silence = np.zeros(10, dtype=np.float32)
    wav = tmp_path / "keep.wav"
    flac = tmp_path / "drop.flac"
    sf.write(wav, silence, SAMPLE_RATE)
    sf.write(flac, silence, SAMPLE_RATE)

    # Restricting to (.wav,) excludes the .flac even though it is real audio.
    result = iter_audio_files(tmp_path, extensions=(".wav",))
    assert result == [wav]
    assert flac not in result
    # Sanity: the default extension set would have included the .flac.
    assert ".flac" in AUDIO_EXTENSIONS
