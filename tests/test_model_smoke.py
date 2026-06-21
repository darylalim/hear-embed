"""Smoke test that loads the REAL HeAR model and runs a forward pass.

This is the one test that can catch breakage in model loading / inference
(e.g. a transformers version bump changing the output, or the gated repo
moving). It is skipped automatically unless BOTH are true:

  * ``torch`` and ``transformers`` are installed, and
  * you have authenticated access to the gated ``google/hear-pytorch`` repo
    (``huggingface-cli login`` or ``HF_TOKEN``).

So contributors without the heavy deps / gated access still get a green run,
while CI configured with an HF token exercises the real model. Run just this
test with ``pytest -m model``; skip it with ``pytest -m "not model"``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hear_embed.audio import CLIP_LENGTH
from hear_embed.embedder import DEFAULT_MODEL_ID, EMBEDDING_DIM


def _missing_deps() -> list[str]:
    return [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]


def _has_model_access(model_id: str) -> bool:
    """Cheaply verify authenticated access to the gated repo (no full download)."""
    try:
        from huggingface_hub import auth_check
    except ImportError:
        # Older huggingface_hub: fall back to "is any token configured?".
        try:
            from huggingface_hub import get_token

            return get_token() is not None
        except Exception:
            return False
    try:
        auth_check(model_id)  # raises if gated/unauthenticated/missing.
        return True
    except Exception:
        return False


_skip_reason = (
    f"needs torch+transformers and authenticated access to gated {DEFAULT_MODEL_ID} "
    "(huggingface-cli login / HF_TOKEN)"
)

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        bool(_missing_deps()) or not _has_model_access(DEFAULT_MODEL_ID),
        reason=_skip_reason,
    ),
]


@pytest.fixture(scope="module")
def embedder():
    """Load the model once on CPU (deterministic, no-GPU-CI safe)."""
    from hear_embed.embedder import HearEmbedder

    return HearEmbedder(device="cpu")


def test_embed_clips_returns_512d_finite(embedder):
    rng = np.random.default_rng(0)
    clips = (rng.standard_normal((2, CLIP_LENGTH)) * 0.1).astype(np.float32)
    vectors = embedder.embed_clips(clips, batch_size=2)
    assert vectors.shape == (2, EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()


def test_embed_file_end_to_end_and_deterministic(embedder, tmp_path):
    import soundfile as sf

    from hear_embed.pipeline import embed_file

    sr = 16_000
    audio = (np.random.default_rng(1).standard_normal(3 * sr) * 0.1).astype(np.float32)
    wav = tmp_path / "clip.wav"
    sf.write(wav, audio, sr)

    vectors, metadata = embed_file(wav, embedder, overlap=0.0)
    assert vectors.shape == (2, EMBEDDING_DIM)  # 3 s, no overlap -> 2 windows.
    assert [m.clip_index for m in metadata] == [0, 1]

    # eval-mode model on fixed input must be reproducible.
    again, _ = embed_file(wav, embedder, overlap=0.0)
    np.testing.assert_allclose(vectors, again, rtol=1e-4, atol=1e-4)
