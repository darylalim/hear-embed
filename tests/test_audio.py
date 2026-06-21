"""Unit tests for windowing — pure numpy, no model/torch/soundfile required."""

import numpy as np
import pytest

from hear_embed.audio import CLIP_LENGTH, window_audio


@pytest.mark.parametrize(
    "n_samples, overlap, expected_offsets",
    [
        (0, 0.0, []),  # empty -> no clips
        (CLIP_LENGTH // 2, 0.0, [0]),  # 1s -> one padded clip
        (CLIP_LENGTH, 0.0, [0]),  # exactly one full clip, no extra
        (CLIP_LENGTH + 100, 0.0, [0, CLIP_LENGTH]),  # spills into a padded 2nd clip
        (3 * CLIP_LENGTH, 0.0, [0, CLIP_LENGTH, 2 * CLIP_LENGTH]),  # 3 aligned clips
        (2 * CLIP_LENGTH, 0.5, [0, CLIP_LENGTH // 2, CLIP_LENGTH]),  # 50% overlap
    ],
    ids=[
        "empty",
        "half-clip",
        "one-clip",
        "spill-pad",
        "three-aligned",
        "half-overlap",
    ],
)
def test_window_offsets_and_shape(n_samples, overlap, expected_offsets):
    clips, offsets = window_audio(
        np.arange(n_samples, dtype=np.float32), overlap=overlap
    )
    assert offsets == expected_offsets
    # One row per offset, and every clip is padded/cropped to exactly CLIP_LENGTH.
    assert clips.shape == (len(expected_offsets), CLIP_LENGTH)


def test_short_audio_is_padded_not_truncated():
    audio = np.ones(CLIP_LENGTH // 2, dtype=np.float32)  # 1 second of ones
    clips, _ = window_audio(audio)
    assert np.all(clips[0, : CLIP_LENGTH // 2] == 1.0)  # real samples kept
    assert np.all(clips[0, CLIP_LENGTH // 2 :] == 0.0)  # tail zero-padded


def test_last_window_is_zero_padded():
    audio = np.ones(CLIP_LENGTH + 100, dtype=np.float32)
    clips, _ = window_audio(audio, overlap=0.0)
    assert np.all(clips[1, :100] == 1.0)  # 100 real samples
    assert np.all(clips[1, 100:] == 0.0)  # then padding


def test_no_overlap_reconstructs_signal_exactly():
    audio = np.arange(3 * CLIP_LENGTH, dtype=np.float32)
    clips, _ = window_audio(audio, overlap=0.0)
    assert np.array_equal(clips.reshape(-1), audio)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        window_audio(np.zeros(CLIP_LENGTH, dtype=np.float32), overlap=1.0)


def test_all_clips_are_full_length_for_unaligned_input():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(5 * CLIP_LENGTH + 777).astype(np.float32)
    clips, offsets = window_audio(audio, overlap=0.25)
    assert clips.shape[1] == CLIP_LENGTH
    assert clips.shape[0] == len(offsets)
    assert offsets[0] == 0
