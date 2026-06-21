"""Orchestration: file/recording -> windowed clips -> embeddings + metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import CLIP_LENGTH, SAMPLE_RATE, load_and_resample, window_audio
from .embedder import EMBEDDING_DIM, Embedder


@dataclass(frozen=True)
class ClipMetadata:
    """Where a single embedding came from in the source recording."""

    source_file: str
    clip_index: int
    start_sample: int
    start_sec: float
    end_sec: float


def embed_file(
    path: str | Path,
    embedder: Embedder,
    overlap: float = 0.0,
    batch_size: int = 64,
    pool: str = "none",
) -> tuple[np.ndarray, list[ClipMetadata]]:
    """Embeds one recording.

    Args:
        path: Audio file (any length).
        embedder: Any object satisfying :class:`~hear_embed.embedder.Embedder`
            (e.g. a loaded :class:`~hear_embed.embedder.HearEmbedder`).
        overlap: Fractional overlap between consecutive 2-second windows.
        batch_size: Clips per model forward pass.
        pool: ``"none"`` for one embedding per window, or ``"mean"`` to average
            all windows into a single ``(1, 512)`` vector for the whole file.

    Returns:
        ``(vectors, metadata)`` with ``vectors`` shaped ``(n, 512)`` and one
        :class:`ClipMetadata` per row.
    """
    # Validate up front so a bad pool fails fast, before loading/windowing the
    # file and running a (potentially GPU) forward pass.
    if pool not in ("none", "mean"):
        raise ValueError(f"pool must be 'none' or 'mean', got {pool!r}")

    audio = load_and_resample(path)
    clips, offsets = window_audio(audio, overlap=overlap)
    vectors = embedder.embed_clips(clips, batch_size=batch_size)

    if pool == "mean":
        if vectors.shape[0] == 0:
            pooled = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        else:
            pooled = vectors.mean(axis=0, keepdims=True)
        vectors = pooled
        metadata = [
            ClipMetadata(
                source_file=str(path),
                clip_index=0,
                start_sample=0,
                start_sec=0.0,
                end_sec=len(audio) / SAMPLE_RATE,
            )
        ]
        return vectors, metadata

    # pool == "none": one embedding per window.
    metadata = [
        ClipMetadata(
            source_file=str(path),
            clip_index=i,
            start_sample=offset,
            start_sec=offset / SAMPLE_RATE,
            end_sec=(offset + CLIP_LENGTH) / SAMPLE_RATE,
        )
        for i, offset in enumerate(offsets)
    ]
    return vectors, metadata
