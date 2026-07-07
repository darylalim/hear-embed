# HeAR Embed

CLI application for embedding human health acoustics using Google
[HeAR](https://huggingface.co/google/hear-pytorch).

HeAR (Health Acoustic Representations) is a self-supervised encoder trained on
human non-speech health sounds — coughing, breathing, throat-clearing, laughing,
speaking. It turns a **2-second, 16 kHz, mono** clip into a **512-dimensional**
embedding. This package wraps the PyTorch model with the pieces a real pipeline
needs: loading arbitrary audio, resampling to the required format, windowing long
recordings into 2-second clips, batching them through the model, and writing the
vectors plus their time offsets to disk.

> **Scope note.** HeAR was trained only on *human* health acoustics. It is not
> designed for general bioacoustics (animals, birds, nature). For those, look at
> Perch / BirdNET instead.

## Install

This project uses [uv](https://docs.astral.sh/uv/). One command creates the
virtual environment and installs everything — the core deps plus the `dev` and
`model` dependency groups (both on by default) — and writes/checks `uv.lock`:

```bash
uv sync          # then prefix commands with `uv run`
```

The HeAR encoder's heavy deps (`torch`, `transformers`) live in a separate
`model` group so CI can skip them (`uv sync --no-group model`); `uv sync`
installs them by default for local use.

Without uv, a plain editable install must add the model deps explicitly, since
pip does not install dependency groups:

```bash
pip install -e . "torch>=2.1" "transformers==4.50.3" pytest
```

Requires Python ≥ 3.10 (the repo pins 3.11 via `.python-version`; `uv` will
fetch it if missing). `soundfile` needs the system `libsndfile` library
(`brew install libsndfile` on macOS; usually preinstalled on Linux).

> With uv, torch resolution is platform-aware: macOS gets the CPU/Metal wheels
> from PyPI, while Linux gets CPU-only wheels from the PyTorch CPU index
> configured under `[tool.uv]` in `pyproject.toml` (so CI avoids the multi-GB
> CUDA build); for CUDA on Linux, swap that index per uv's
> [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
> pip ignores `[tool.uv]`, so the pip fallback above pulls PyPI's default
> torch — the CUDA build on Linux. Add
> `--index-url https://download.pytorch.org/whl/cpu` there for CPU-only.

### Authenticate (the model is gated)

`google/hear-pytorch` is gated under the Health AI Developer Foundations terms:

1. Accept the terms at <https://huggingface.co/google/hear-pytorch>.
2. `uv run huggingface-cli login` (or export `HF_TOKEN=...`).

## CLI

```bash
# Embed every recording in a folder, 50% window overlap, to Parquet:
uv run hear-embed ./recordings --overlap 0.5 --out embeddings.parquet

# One averaged vector per file instead of one per window:
uv run hear-embed ./recordings --pool mean --out file_embeddings.parquet

# A single file, NumPy output (writes embeddings.npy + embeddings.csv):
uv run hear-embed cough.wav --format npz --out embeddings
```

Key flags: `--overlap` (window overlap in `[0, 1)`), `--pool` (`none` per-window /
`mean` per-file), `--batch-size`, `--device` (`cuda`/`cpu`), `--format`
(`parquet`/`npz`), `--model`, `--extensions`. Run `uv run hear-embed --help` for
the full, colorized reference with defaults.

The default Parquet output is **streamed** (one row group per file), so embedding
a large corpus never holds all vectors in memory.

## Library

```python
from hear_embed import HearEmbedder, embed_file, window_audio

embedder = HearEmbedder()  # loads google/hear-pytorch onto GPU if available

# Full recording -> per-window embeddings + metadata:
vectors, metadata = embed_file("cough.wav", embedder, overlap=0.5)
# vectors: (n_windows, 512) float32
# metadata[i]: source_file, clip_index, start_sample, start_sec, end_sec

# Or drive the pieces yourself:
import numpy as np
clips, offsets = window_audio(np.zeros(48000, dtype=np.float32))  # (n, 32000)
vecs = embedder.embed_clips(clips)                                # (n, 512)
```

## Output schema (Parquet)

| column | type | meaning |
|---|---|---|
| `source_file` | string | path of the source recording |
| `clip_index` | int32 | window index within that recording |
| `start_sample` | int64 | window start, in samples (16 kHz) |
| `start_sec` / `end_sec` | float64 | window start/end, in seconds |
| `embedding` | list<float32>[512] | the HeAR embedding |

## How it works

1. **Load + normalize** (`audio.load_and_resample`) — read via `soundfile`
   (which scales integer PCM to float correctly), downmix to mono, resample to
   16 kHz with `scipy.signal.resample`.
2. **Window** (`audio.window_audio`) — slide 2-second windows with configurable
   overlap; zero-pad the final clip to exactly 32,000 samples.
3. **Embed** (`embedder.HearEmbedder`) — apply HeAR's mel-PCEN preprocessing and
   run the PyTorch encoder, taking `pooler_output` as the 512-dim vector.

The preprocessing in `hear_embed/_vendor/audio_utils.py` is vendored
**unmodified** from [`Google-Health/hear`](https://github.com/Google-Health/hear)
(Apache-2.0) so it matches Google's reference exactly.

## Tests

```bash
uv run pytest                 # torch-free tests + auto-skipped model smoke test
uv run pytest -m "not model"  # everything except the heavy model smoke test
uv run pytest -m model        # load the real model + one forward pass (see below)
```

The bulk of the suite is **torch-free** — loading/resampling, windowing,
writers, pipeline pooling, and the CLI all run without the model or a GPU (a
fake embedder stands in), so they make up CI's `-m "not model"` job.

`tests/test_model_smoke.py` loads the real `google/hear-pytorch` and runs a
forward pass, so CI can catch model load / inference breakage. It **skips
automatically** unless `torch` + `transformers` are installed *and* you have
authenticated access to the gated repo — point CI at an `HF_TOKEN` to make it
run there.

## Development

```bash
uv run ruff check --fix    # lint + autofix (incl. import sorting)
uv run ruff format         # format
uv run ty check            # type check
uv run pre-commit install  # run ruff + ty automatically on every commit
```

`ty` is Astral's Rust type checker. It is **pre-1.0 (preview)**, so its exact
version is pinned via `uv.lock` and behavior may shift between releases. It
runs in pre-commit and in CI's `typecheck` job; both install the `model` group
(torch/transformers) so ty can resolve those imports. CI gets CPU-only torch
wheels (see the install note above), keeping the job a few hundred MB instead
of multi-GB.

## License

This repository is Apache-2.0. The HeAR **model weights** are governed
separately by the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms)
— notably, clinical/diagnostic use requires appropriate regulatory authorization.
