"""Unit tests for windowing — pure numpy, no model/torch/soundfile required."""

import numpy as np
import pytest

from hear_pipeline.audio import CLIP_LENGTH, window_audio


def test_short_audio_yields_one_padded_clip():
    audio = np.ones(CLIP_LENGTH // 2, dtype=np.float32)  # 1 second.
    clips, offsets = window_audio(audio)
    assert clips.shape == (1, CLIP_LENGTH)
    assert offsets == [0]
    # The tail is zero-padded out to a full clip.
    assert np.all(clips[0, CLIP_LENGTH // 2 :] == 0.0)
    assert np.all(clips[0, : CLIP_LENGTH // 2] == 1.0)


def test_exact_multiple_no_overlap():
    audio = np.arange(3 * CLIP_LENGTH, dtype=np.float32)
    clips, offsets = window_audio(audio, overlap=0.0)
    assert clips.shape == (3, CLIP_LENGTH)
    assert offsets == [0, CLIP_LENGTH, 2 * CLIP_LENGTH]
    # No padding needed: clips reconstruct the signal exactly.
    assert np.array_equal(clips.reshape(-1), audio)


def test_overlap_halves_the_step():
    audio = np.zeros(2 * CLIP_LENGTH, dtype=np.float32)
    _, offsets = window_audio(audio, overlap=0.5)
    # Step = 16000 -> starts at 0, 16000, 32000 to cover the end.
    assert offsets == [0, CLIP_LENGTH // 2, CLIP_LENGTH]


def test_last_window_is_padded_when_not_aligned():
    audio = np.ones(CLIP_LENGTH + 100, dtype=np.float32)
    clips, offsets = window_audio(audio, overlap=0.0)
    assert offsets == [0, CLIP_LENGTH]
    # Second clip has 100 real samples then padding.
    assert np.all(clips[1, :100] == 1.0)
    assert np.all(clips[1, 100:] == 0.0)


def test_empty_audio_returns_no_clips():
    clips, offsets = window_audio(np.array([], dtype=np.float32))
    assert clips.shape == (0, CLIP_LENGTH)
    assert offsets == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        window_audio(np.zeros(CLIP_LENGTH, dtype=np.float32), overlap=1.0)


def test_clips_are_exactly_clip_length():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(5 * CLIP_LENGTH + 777).astype(np.float32)
    clips, offsets = window_audio(audio, overlap=0.25)
    assert clips.shape[1] == CLIP_LENGTH
    assert clips.shape[0] == len(offsets)
    assert offsets[0] == 0
