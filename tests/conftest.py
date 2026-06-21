"""Shared torch-free test fixtures and the FakeEmbedder stand-in.

Kept in one place so the test modules that need a fake encoder, the CLI's
HearEmbedder patched out, or a throwaway wav don't drift apart. Nothing here
imports torch or loads a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hear_embed.audio import SAMPLE_RATE
from hear_embed.embedder import EMBEDDING_DIM


class FakeEmbedder:
    """Torch-free stand-in for :class:`~hear_embed.embedder.HearEmbedder`.

    Accepts the real constructor's kwargs (so the CLI tests can monkeypatch it
    onto ``HearEmbedder`` and have ``main`` build it with ``model_id``/``device``)
    and returns deterministic embeddings where row ``i``, column ``j`` is
    ``i + j`` — values vary along both axes, so callers can assert row order and
    the exact per-column pooled mean. Satisfies the ``Embedder`` protocol
    structurally; no subclassing or torch needed.
    """

    def __init__(self, model_id: str = "fake", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device

    def embed_clips(self, clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
        n = clips.shape[0]
        if n == 0:
            # Match HearEmbedder.embed_clips: empty in -> empty (0, 512) out.
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        rows = np.arange(n, dtype=np.float32)[:, None]
        cols = np.arange(EMBEDDING_DIM, dtype=np.float32)[None, :]
        return rows + cols


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A ready-to-use, torch-free embedder stand-in."""
    return FakeEmbedder()


@pytest.fixture
def use_fake_embedder(monkeypatch):
    """Patch the CLI's lazily-imported ``HearEmbedder`` with the fake."""
    monkeypatch.setattr("hear_embed.embedder.HearEmbedder", FakeEmbedder)


@pytest.fixture
def write_wav():
    """Factory writing a deterministic mono 16 kHz wav; returns the waveform.

    Defaults to 3 s (two 2-second windows, no overlap). The file format follows
    ``path``'s extension, so passing e.g. ``foo.flac`` writes real FLAC.
    """

    def _write_wav(path: Path, n_samples: int = 3 * SAMPLE_RATE) -> np.ndarray:
        audio = (np.random.default_rng(0).standard_normal(n_samples) * 0.1).astype(
            np.float32
        )
        sf.write(path, audio, SAMPLE_RATE)
        return audio

    return _write_wav
