"""Unit tests for pipeline pooling — torch-free (a fake embedder, no model).

These exercise :func:`hear_embed.pipeline.embed_file` end to end through real
audio loading/windowing, but stub the model with a deterministic FakeEmbedder so
the suite runs in CI's ``-m "not model"`` job without torch/transformers.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from hear_embed.audio import CLIP_LENGTH, SAMPLE_RATE
from hear_embed.embedder import EMBEDDING_DIM, HearEmbedder
from hear_embed.pipeline import ClipMetadata, embed_file


class FakeEmbedder(HearEmbedder):
    """Deterministic stand-in for :class:`HearEmbedder` (no torch/model).

    Subclasses :class:`HearEmbedder` so it satisfies ``embed_file``'s parameter
    type, but skips the real model-loading ``__init__`` — importing
    ``HearEmbedder`` does not import torch (the real ``__init__`` imports it
    lazily), so this stays torch-free.

    Row ``i``, column ``j`` of the output is ``i + j`` — values vary along both
    axes, so a wrong reduction axis (or a global mean) is detectable — while row
    ``i`` still starts at ``i``. Tests assert (a) ``embed_file`` preserves row
    order against the metadata, and (b) the exact per-column pooled mean.
    Mirrors the real embedder's empty-input contract: ``(0, 512)`` for no clips.
    """

    def __init__(self) -> None:
        pass  # Intentionally skip HearEmbedder.__init__ (loads torch + the model).

    def embed_clips(self, clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
        n = clips.shape[0]
        if n == 0:
            # Match HearEmbedder.embed_clips: empty in -> empty (0, 512) out.
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        # Row i, column j = i + j: values vary along BOTH axes, so a mean over
        # the wrong axis (or a global mean) differs from the correct per-column
        # mean — letting the pooling tests tell a correct reduction from a buggy one.
        rows = np.arange(n, dtype=np.float32)[:, None]
        cols = np.arange(EMBEDDING_DIM, dtype=np.float32)[None, :]
        return rows + cols


def _write_wav(path, n_samples: int) -> np.ndarray:
    """Writes a deterministic mono 16 kHz wav and returns the written waveform."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    sf.write(path, audio, SAMPLE_RATE)
    return audio


def test_pool_none_shapes_and_metadata(tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav, 3 * SAMPLE_RATE)  # 3 s -> 1.5 clips -> 2 windows.

    vectors, metadata = embed_file(wav, FakeEmbedder(), pool="none")

    n_windows = vectors.shape[0]
    assert n_windows == 2  # 48000 samples, no overlap.
    assert vectors.shape == (n_windows, EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    # One metadata row per embedding row.
    assert len(metadata) == n_windows
    assert all(isinstance(m, ClipMetadata) for m in metadata)

    # clip_index runs 0..n-1 in order.
    assert [m.clip_index for m in metadata] == list(range(n_windows))

    # FakeEmbedder row i is (i + arange(512)); embed_file must not reshuffle rows.
    expected_cols = np.arange(EMBEDDING_DIM, dtype=np.float32)
    for i, m in enumerate(metadata):
        assert np.array_equal(vectors[i], float(i) + expected_cols)
        # No-overlap offsets are exact multiples of CLIP_LENGTH.
        offset = i * CLIP_LENGTH
        assert m.start_sample == offset
        assert m.start_sec == offset / SAMPLE_RATE
        assert m.end_sec == (offset + CLIP_LENGTH) / SAMPLE_RATE
        assert m.source_file == str(wav)


def test_pool_mean_averages_windows_and_spans_file(tmp_path):
    wav = tmp_path / "clip.wav"
    audio = _write_wav(wav, 3 * SAMPLE_RATE)  # 2 windows.

    # Per-window fakes are [0..511] and [1..512]; mean-pooling collapses them to
    # their column-wise mean [0.5, 1.5, ..., 511.5]. Because every column differs,
    # a wrong reduction axis or a global mean would NOT match this expectation.
    clips = np.empty((2, CLIP_LENGTH), dtype=np.float32)
    per_window = FakeEmbedder().embed_clips(clips)

    vectors, metadata = embed_file(wav, FakeEmbedder(), pool="mean")

    assert vectors.shape == (1, EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    # Pooled vector is the exact column-wise mean of the per-window fakes.
    np.testing.assert_array_equal(vectors, per_window.mean(axis=0, keepdims=True))

    # A single ClipMetadata spans the whole file.
    assert len(metadata) == 1
    (m,) = metadata
    assert m.clip_index == 0
    assert m.start_sample == 0
    assert m.start_sec == 0.0
    assert m.end_sec == len(audio) / SAMPLE_RATE  # 3.0 s.
    assert m.source_file == str(wav)


def test_invalid_pool_raises_value_error(tmp_path):
    wav = tmp_path / "clip.wav"
    _write_wav(wav, 3 * SAMPLE_RATE)
    with pytest.raises(ValueError, match="pool must be"):
        embed_file(wav, FakeEmbedder(), pool="median")


def test_sub_clip_audio_yields_single_padded_window(tmp_path):
    # 1 s of audio is shorter than one 2 s clip: windowing pads it to one window.
    wav = tmp_path / "short.wav"
    _write_wav(wav, SAMPLE_RATE // 2)  # 0.5 s.

    vectors, metadata = embed_file(wav, FakeEmbedder(), pool="none")

    assert vectors.shape == (1, EMBEDDING_DIM)
    assert len(metadata) == 1
    assert metadata[0].clip_index == 0
    assert metadata[0].start_sample == 0
    # The (single padded) clip is still reported as a full 2 s span.
    assert metadata[0].end_sec == CLIP_LENGTH / SAMPLE_RATE


def test_empty_audio_pool_none_is_empty(tmp_path):
    # An empty recording produces no windows -> empty vectors and no metadata.
    wav = tmp_path / "empty.wav"
    sf.write(wav, np.zeros(0, dtype=np.float32), SAMPLE_RATE)

    vectors, metadata = embed_file(wav, FakeEmbedder(), pool="none")

    assert vectors.shape == (0, EMBEDDING_DIM)
    assert metadata == []


def test_empty_audio_pool_mean_is_zeros(tmp_path):
    # With no windows, mean-pooling falls back to a single zero vector spanning 0 s.
    wav = tmp_path / "empty.wav"
    sf.write(wav, np.zeros(0, dtype=np.float32), SAMPLE_RATE)

    vectors, metadata = embed_file(wav, FakeEmbedder(), pool="mean")

    assert vectors.shape == (1, EMBEDDING_DIM)
    assert np.all(vectors == 0.0)
    assert len(metadata) == 1
    assert metadata[0].start_sec == 0.0
    assert metadata[0].end_sec == 0.0  # empty file -> zero-length span.
