"""HeAR PyTorch model wrapper: 2-second clips -> 512-dim embeddings."""

from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 512
DEFAULT_MODEL_ID = "google/hear-pytorch"


class HearEmbedder:
    """Loads the HeAR PyTorch encoder and embeds batches of 2-second clips.

    The model is gated on the Hugging Face Hub: accept the Health AI Developer
    Foundations terms at https://huggingface.co/google/hear-pytorch and run
    ``huggingface-cli login`` (or set ``HF_TOKEN``) before first use.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ) -> None:
        # Imported lazily so the audio/windowing code stays importable (and
        # testable) without torch/transformers installed.
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed_clips(self, clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Embeds ``(n, 32000)`` clips into an ``(n, 512)`` float32 array.

        Each clip must be exactly 32,000 samples (2 s at 16 kHz); use
        :func:`hear_pipeline.audio.window_audio` to produce them. Preprocessing
        (mel-PCEN spectrogram) runs on-device via the vendored ``preprocess_audio``.
        """
        from ._vendor.audio_utils import preprocess_audio

        torch = self._torch
        clips = np.asarray(clips, dtype=np.float32)
        if clips.ndim == 1:
            clips = clips[None, :]
        if clips.ndim != 2:
            raise ValueError(f"clips must be 1-D or 2-D, got shape {clips.shape}")
        if clips.shape[0] == 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, clips.shape[0], batch_size):
                batch = torch.from_numpy(clips[i : i + batch_size]).to(self.device)
                spectrogram = preprocess_audio(batch)
                result = self.model.forward(
                    spectrogram, return_dict=True, output_hidden_states=True
                )
                outputs.append(result.pooler_output.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)
