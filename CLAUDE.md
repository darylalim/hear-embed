# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`hear-embed` (distribution name `hear-embed`, import package `hear_embed`) wraps Google's gated **HeAR** PyTorch model (`google/hear-pytorch`) in a real embedding pipeline: load arbitrary audio, resample to 16 kHz mono, window long recordings into fixed 2-second clips (32,000 samples), batch them through the encoder to 512-dim vectors, and stream the vectors + time offsets to disk. HeAR is for **human** health acoustics only (cough/breath/speech), not general bioacoustics.

## Commands

Uses [uv](https://docs.astral.sh/uv/). `uv sync` installs the `dev` + `model` groups by default (see dependency-group note below).

When working with Python, invoke the relevant `/astral:<skill>` skill — `/astral:uv`, `/astral:ty`, `/astral:ruff` — for the corresponding tool to ensure best practices are followed.

```bash
uv sync                          # full local env (dev tools + torch/transformers)
uv sync --no-group model         # core + dev only; enough for lint + the torch-free tests

uv run pytest                    # all tests (model smoke test auto-skips without deps/access)
uv run pytest -m "not model"     # everything except the heavy model smoke test
uv run pytest -m model           # only the real-model smoke test (needs gated access + HF_TOKEN)
uv run pytest tests/test_audio.py -k overlap   # a single test / subset

uv run ruff check --fix          # lint + autofix (includes import sorting)
uv run ruff format               # format
uv run ty check                  # type check (Astral's preview type checker; version pinned via uv.lock)

uv run hear-embed <input> [...]  # run the CLI (input = file or directory)
uv lock                          # regenerate the lockfile after dependency/name changes
uv run pre-commit install        # run ruff + ty on every commit
```

## Architecture: the torch-free boundary

The single most important design decision, and it shapes everything else: the **audio/IO/windowing layer never imports torch**. All heavy or optional dependencies (`torch`, `transformers`, `soundfile`, `scipy`, `pyarrow`, `tqdm`) are imported **lazily inside functions/methods**, never at module top level. Consequences to preserve when editing:

- `torch` + `transformers` live in a separate `model` dependency group, so `uv sync --no-group model` (and CI's lint/test job) gives a working install of everything except the encoder.
- Everything except the encoder is testable without the model: `tests/test_audio.py` (pure-numpy windowing), `test_audio_io.py` (loading/resampling), `test_writers.py` (Parquet/npz), `test_pipeline.py` (pooling), and `test_cli.py` (exit codes) all run in CI's `-m "not model"` job — the last two use a fake/monkeypatched embedder, so no torch is needed. Only `test_model_smoke.py` loads the real model.
- **Do not add top-level `import torch` / `import transformers` / etc. to the audio or IO modules.** Match the existing lazy-import pattern instead.

### Data flow

```
iter_audio_files ─┐
load_and_resample ─→ window_audio ─→ HearEmbedder.embed_clips ─→ writer.write
   (audio.py)         (audio.py)         (embedder.py)           (writers.py)
                         └──────── embed_file() orchestrates one file (pipeline.py) ────────┘
                         └──────── main() loops files + streams output (cli.py) ────────────┘
```

- **`audio.py`** — torch-free constants (`SAMPLE_RATE=16000`, `CLIP_LENGTH=32000`) + loading, mono-mixing, `scipy.signal.resample`, and `window_audio` (slides 2 s windows by `clip_length * (1 - overlap)`, zero-pads the final clip, stops before emitting a redundant all-padding trailing window).
- **`embedder.py`** — `HearEmbedder` loads the encoder (CUDA if available else CPU), batches clips, runs the vendored mel-PCEN preprocessing, and takes `pooler_output` as the 512-dim vector.
- **`pipeline.py`** — `embed_file()` ties load→window→embed together and builds `ClipMetadata` per row; `pool="none"` gives one vector per window, `pool="mean"` averages to one vector per file.
- **`writers.py`** — `ParquetEmbeddingWriter` **streams** one row group per file (constant memory for large corpora); `NpzEmbeddingWriter` accumulates in memory and writes `<stem>.npy` + `<stem>.csv`. Pick via `make_writer(path, fmt)`.
- **`cli.py`** — defers model load until files are found, turns any load failure into actionable gating guidance, and skips individual bad files rather than aborting. Exit codes: `0` success, `1` no files found, `2` model load/gating failure, `3` some files skipped.
    - `--help` is colorized via a `RichHelpFormatter` subclass (`_RichHelp`) with rich-argparse's markup parsing disabled, so bracketed help text (e.g. `[0, 1)`, `[parquet|npz]`) renders literally instead of being eaten as Rich tags. **`rich-argparse` is a core dependency** that pulls in the `rich` stack (`rich`, `pygments`, `markdown-it-py`, `mdurl`); unlike the heavy/optional deps above it's imported at module top level — fine, since it's lightweight and torch-free.

## Hard invariants

- **`hear_embed/_vendor/audio_utils.py` is vendored unmodified** from [`Google-Health/hear`](https://github.com/Google-Health/hear) (Apache-2.0) so preprocessing matches Google's reference exactly. Never lint, reformat, or edit it — it is excluded from ruff (`extend-exclude` + `force-exclude`) and ty. Don't "fix" it.
- **`transformers` is pinned to `==4.50.3`** — `google/hear-pytorch` is exported against this release. Don't bump it casually; the `model` smoke test exists to catch breakage if you do.
- **The model is gated.** Accept the Health AI Developer Foundations terms at <https://huggingface.co/google/hear-pytorch> and authenticate (`huggingface-cli login` or `HF_TOKEN`) before any model-touching work. `tests/test_model_smoke.py` skips automatically without both torch/transformers installed and authenticated access.

## CI / tooling notes

- Two jobs: **lint-and-test** runs `uv sync --locked --no-group model` (torch-free tests + ruff, no torch); **typecheck** runs `uv sync --locked` *with* the model group because `ty` needs torch/transformers importable to check `embedder.py`. On Linux, `[tool.uv]` in `pyproject.toml` resolves CPU-only torch from the `pytorch-cpu` index (hundreds of MB) instead of the multi-GB CUDA build.
- CI's `uv run` steps pass **`--no-sync`** on purpose: a bare `uv run` re-syncs the default groups and would quietly reinstall the `model` group a job just chose to skip. If you add CI steps, keep `--no-sync` after the initial `uv sync`.
- `pytest` runs with `--strict-markers`; the only registered marker is `model`. Register any new marker in `[tool.pytest.ini_options]` or it will error.
- Targets Python ≥ 3.10 (`.python-version` pins 3.11 locally; `ty` checks against 3.10). Ruff lint set: `E, F, I, UP, B, SIM`.

## Claude Code hooks

`.claude/settings.json` wires two committed hooks (scripts in `.claude/hooks/`, stdlib-only Python) that mechanically enforce the hard invariants above. They are project config, not code the pipeline imports — but they're linted/typechecked like the rest of the repo (keep them ruff/ty clean).

- **`guard_paths.py` (PreToolUse, `Edit|Write|NotebookEdit|Read`)** — *denies* edits under `hear_embed/_vendor/`, denies read/edit of `.env` (secrets; `.env.example` stays allowed), and denies hand-edits of `uv.lock` (regenerate via `uv lock`). Reads of vendored code and `uv.lock` are fine.
- **`check_torch_free.py` (PostToolUse, `Edit|Write`)** — warns (feeds a message back) when a **top-level** `import torch`/`transformers` lands in `audio.py`, `pipeline.py`, or `writers.py`; indented lazy imports don't trip it. `embedder.py` is intentionally unguarded.
- **`_run.sh`** resolves the interpreter — prefers `.venv/bin/python`, then system `python3`/`python`, then `uv run --no-sync python`; if none exists it exits 0 (fails *open* so a missing interpreter can't block every edit). Bare `python3` is not reliably on PATH, which is why the wrapper exists.
- Editing `.claude/settings.json` while a session is running may not take effect until `/hooks` is reopened or Claude Code restarts (the settings watcher only tracks files that existed at session start).
