"""hear-embed: embed human health acoustics using Google HeAR.

A CLI and Python library. Typical library use::

    from hear_embed import HearEmbedder, embed_file

    embedder = HearEmbedder()  # loads google/hear-pytorch
    vectors, metadata = embed_file("cough.wav", embedder, overlap=0.5)
    # vectors: (n_windows, 512) float32; metadata[i] locates window i in time.
"""

from __future__ import annotations

from .audio import (
    AUDIO_EXTENSIONS,
    CLIP_DURATION,
    CLIP_LENGTH,
    SAMPLE_RATE,
    iter_audio_files,
    load_and_resample,
    window_audio,
)
from .embedder import DEFAULT_MODEL_ID, EMBEDDING_DIM, HearEmbedder
from .pipeline import ClipMetadata, embed_file

__all__ = [
    "AUDIO_EXTENSIONS",
    "CLIP_DURATION",
    "CLIP_LENGTH",
    "SAMPLE_RATE",
    "EMBEDDING_DIM",
    "DEFAULT_MODEL_ID",
    "HearEmbedder",
    "ClipMetadata",
    "embed_file",
    "iter_audio_files",
    "load_and_resample",
    "window_audio",
]

__version__ = "0.1.0"
