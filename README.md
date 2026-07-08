# HeAR Embed

[![CI](https://github.com/darylalim/hear-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/hear-embed/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

`hear-embed` is a CLI **and** Python library for embedding human health acoustics
using Google [HeAR](https://huggingface.co/google/hear-pytorch).

HeAR (Health Acoustic Representations) is a self-supervised encoder trained on
human non-speech health sounds — coughing, breathing, throat-clearing, laughing,
speaking ([Baur et al. 2024](https://arxiv.org/abs/2403.02522)). It turns a
**2-second, 16 kHz, mono** clip into a **512-dimensional** embedding. This package
wraps the PyTorch model with the pieces a real pipeline needs: loading arbitrary
audio, resampling to the required format, windowing long recordings into 2-second
clips, batching them through the model, and writing the vectors plus their time
offsets to disk.

> **Scope note.** HeAR was trained only on *human* health acoustics. It is not
> designed for general bioacoustics (animals, birds, nature). For those, look at
> Perch / BirdNET instead.

## Quickstart

```bash
uv sync                                    # create the env (core + torch + dev tools)
# then accept the HeAR terms and put your HF token in .env — see Install > Authenticate
uv run --env-file .env hear-embed cough.wav --out embeddings.parquet
```

See [Install](#install) for gated-model auth and non-uv setups, and [CLI](#cli) for
every flag.

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

### Authenticate (the model is gated)

`google/hear-pytorch` is gated under the Health AI Developer Foundations terms:

1. Accept the terms at <https://huggingface.co/google/hear-pytorch>.
2. Authenticate with a Hugging Face token — either `uv run huggingface-cli
   login`, or copy `.env.example` to `.env`, add your `HF_TOKEN`, and load it
   per command with `uv run --env-file .env …` (or `export UV_ENV_FILE=.env`
   for the whole shell). `.env` is gitignored — never commit it.

### Other install notes

**Not on PyPI.** Install from source. With uv, `uv sync` (above) is the happy
path; to add it to another project, `uv pip install
"git+https://github.com/darylalim/hear-embed" "torch>=2.1" "transformers==4.50.3"`.

**Plain pip.** pip does not install dependency groups, so add the model deps
explicitly:

```bash
pip install -e . "torch>=2.1" "transformers==4.50.3"
```

**Python & libsndfile.** Requires Python ≥ 3.10 (the repo pins 3.11 via
`.python-version`; `uv` will fetch it if missing). `soundfile` needs the system
`libsndfile` library (`brew install libsndfile` on macOS; usually preinstalled on
Linux).

**torch wheels.** With uv, torch resolution is platform-aware: macOS gets the
CPU/Metal wheels from PyPI, Linux gets CPU-only wheels from the PyTorch CPU index
(so CI avoids the multi-GB CUDA build); for CUDA on Linux, swap that index per
uv's [PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/). Plain
pip ignores uv's index config and pulls PyPI's default torch (the CUDA build on
Linux) — add `--index-url https://download.pytorch.org/whl/cpu` for CPU-only.

## CLI

```bash
# Embed every recording in a folder, 50% window overlap, to Parquet:
uv run hear-embed ./recordings --overlap 0.5 --out embeddings.parquet

# One averaged vector per file instead of one per window:
uv run hear-embed ./recordings --pool mean --out file_embeddings.parquet

# A single file, NumPy output (writes embeddings.npy + embeddings.csv):
uv run hear-embed cough.wav --format npz --out embeddings
```

Key flags: `--overlap` (window overlap in `[0, 1)`, default 0), `--pool` (`none`
per-window / `mean` per-file), `--batch-size` (default 64), `--device`
(`cuda`/`cpu`), `--format` (`parquet`/`npz`), `--model`, `--extensions`. Run
`uv run hear-embed --help` for the full, colorized reference with defaults.

Per-file errors are logged to stderr and skipped, so one unreadable recording
never aborts a batch. Exit codes:

| code | meaning |
|---|---|
| `0` | all files embedded |
| `1` | no audio files found under the input |
| `2` | model load / gating failure (accept the terms + authenticate) |
| `3` | finished, but one or more files were skipped |

The default Parquet output is **streamed** (one row group per file), so embedding
a large corpus never holds all vectors in memory.

A run reports progress on stderr:

```
Found 3 file(s) to embed.
Wrote 128 embedding(s) from 3/3 file(s) to embeddings.parquet.
```

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

## Using the embeddings

The writers put vectors on disk; here's how to load them back into a matrix.
Only `numpy` + `pyarrow` are needed — both are core deps, no pandas required:

```python
import numpy as np
import pyarrow.parquet as pq

t = pq.read_table("embeddings.parquet")
X = np.stack(t["embedding"].to_numpy(zero_copy_only=False))            # (n, 512) float32
meta = t.select(["source_file", "clip_index", "start_sec", "end_sec"])  # row-aligned with X

# npz output instead: X = np.load("embeddings.npy"); metadata is in <stem>.csv,
# whose leading `row` column is the index into X (X[row] <-> that CSV line).

# Downstream: cosine similarity / nearest-neighbour search over the matrix
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
sims = Xn @ Xn.T          # (n, n) cosine similarities; or feed X to a scikit-learn head
```

## How it works

```text
iter_audio_files → load_and_resample → window_audio → HearEmbedder.embed_clips → writer.write
```

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
fake embedder stands in), so they make up CI's `-m "not model"` job. That job
also runs the project-invariant checks: the `.claude/hooks/` guards, the
license/packaging declarations, and the CI-workflow settings.

`tests/test_model_smoke.py` loads the real `google/hear-pytorch` and runs a
forward pass, so CI can catch model load / inference breakage. It **skips
automatically** unless `torch` + `transformers` are installed *and* you have
authenticated access to the gated repo — point CI at an `HF_TOKEN` to make it
run there.

## Development

```bash
uv run ruff check --fix    # lint + autofix (incl. import sorting)
uv run ruff format         # format
uv run ty check            # type check (Astral's preview checker; version pinned via uv.lock)
uv run pre-commit install  # run ruff + ty automatically on every commit
```

CI runs three jobs: lint + tests torch-free, typecheck with the model group, and
a build that installs the wheel *without* the model group to prove it installs
and runs torch-free at the distribution level. See [CLAUDE.md](CLAUDE.md) for the
full CI/tooling rationale.

## License

This repository is Apache-2.0 (see [`LICENSE`](LICENSE)); the built distribution
ships that text plus the vendored Google license
([`hear_embed/_vendor/LICENSE.apache-2.0`](hear_embed/_vendor/LICENSE.apache-2.0),
for the unmodified `audio_utils.py`). The HeAR **model weights** are governed
separately by the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms)
— notably, clinical/diagnostic use requires appropriate regulatory authorization.
