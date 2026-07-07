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
- Everything except the encoder is testable without the model: `tests/test_audio.py` (pure-numpy windowing), `test_audio_io.py` (loading/resampling), `test_writers.py` (Parquet/npz), `test_pipeline.py` (pooling), `test_cli.py` (exit codes), `test_hooks.py` (the `.claude/hooks/` guards, run as subprocesses), `test_packaging.py` (the `LICENSE`/`license-files` invariants), and `test_ci_workflow.py` (CI-workflow invariants, parsed from `ci.yml` via PyYAML) all run in CI's `-m "not model"` job — the pipeline/CLI ones use a fake/monkeypatched embedder, so no torch is needed. Only `test_model_smoke.py` loads the real model.
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
- **The model is gated.** Accept the Health AI Developer Foundations terms at <https://huggingface.co/google/hear-pytorch> and authenticate (`huggingface-cli login` or `HF_TOKEN`) before any model-touching work. The repo ships `.env.example`; copy it to `.env` (gitignored, and read-blocked by `guard_paths.py`) with your token, then run model steps via `uv run --env-file .env …` or `export UV_ENV_FILE=.env`. `tests/test_model_smoke.py` skips automatically without both torch/transformers installed and authenticated access.
- **The license ships in the built distribution.** `pyproject.toml` declares `license = "Apache-2.0"` and `license-files = ["LICENSE", "hear_embed/_vendor/LICENSE.apache-2.0"]`, so both the project license (root `LICENSE`) and Google's vendored one land as `License-File` entries in the wheel METADATA. `tests/test_packaging.py` guards the source-tree declarations; the CI `build` job unzips the built wheel and asserts both `License-File` entries (and the vendored `audio_utils.py`) are present. The root `LICENSE` is the canonical Apache-2.0 text — don't truncate it.

## CI / tooling notes

- Three jobs: **lint-and-test** runs `uv sync --locked --no-group model` (torch-free tests + ruff, no torch); **typecheck** runs `uv sync --locked` *with* the model group because `ty` needs torch/transformers importable to check `embedder.py`; **build** runs `uv build`, installs the wheel into a fresh env *without* the model group, and smoke-tests `hear-embed --help` + torch-free imports (incl. that every core dep installs with the wheel) — validating the "installs and runs torch-free" promise at the distribution level, which the source-tree tests never exercise; it also unzips the wheel to assert both license texts and the vendored preprocessing are packaged. On Linux, a `[[tool.uv.index]]` + `[tool.uv.sources]` pin in `pyproject.toml` resolves CPU-only torch from the `pytorch-cpu` index (hundreds of MB) instead of the multi-GB CUDA build.
- Workflow-level knobs: a top-level `concurrency` block cancels superseded runs per ref; every job sets `timeout-minutes` (so a hung torch download can't ride the 360-minute default); and each cache-enabled `setup-uv` step takes a distinct `cache-suffix` (`nomodel`/`model`/`build`). The jobs otherwise derive the same uv cache key (same `uv.lock` + OS) but store different contents, so without separate suffixes the faster torch-free save evicts the torch cache and typecheck re-downloads torch nearly every run. `tests/test_ci_workflow.py` locks all of this in.
- CI's `uv run` steps pass **`--no-sync`** on purpose: a bare `uv run` re-syncs the default groups and would quietly reinstall the `model` group a job just chose to skip. If you add CI steps, keep `--no-sync` after the initial `uv sync` — `tests/test_ci_workflow.py` asserts every `uv run` in the workflow keeps it.
- `pytest` runs with `--strict-markers`; the only registered marker is `model`. Register any new marker in `[tool.pytest.ini_options]` or it will error.
- Targets Python ≥ 3.10 (`.python-version` pins 3.11 locally; `ty` checks against 3.10). Ruff lint set: `E, F, I, UP, B, SIM`.

## Claude Code hooks

`.claude/settings.json` wires two committed hooks (scripts in `.claude/hooks/`, stdlib-only Python) that mechanically enforce the hard invariants above. They are project config, not code the pipeline imports — but they're linted/typechecked like the rest of the repo (keep them ruff/ty clean).

- **`guard_paths.py` (PreToolUse, `Edit|MultiEdit|Write|NotebookEdit|Read`)** — *denies* writes under `hear_embed/_vendor/`, denies read/write of any `.env`/`.env.*` secret (matched by basename; `.env.example` stays allowed), and denies hand-edits of `uv.lock` (regenerate via `uv lock`). Reads of vendored code and `uv.lock` are fine. Guarded paths are symlink-resolved, and every write tool (incl. `MultiEdit`) is covered.
- **`check_torch_free.py` (PostToolUse, `Edit|MultiEdit|Write`)** — warns (feeds a message back) when a **top-level** `import torch`/`transformers` lands in `audio.py`, `pipeline.py`, or `writers.py`; indented lazy imports don't trip it. `embedder.py` is intentionally unguarded.
- **`_run.sh`** resolves the interpreter — prefers `.venv/bin/python`, then system `python3`/`python`, then `uv run --no-sync --project <root> python`; if none exists it exits 0 (fails *open* so a missing interpreter can't block every edit). Bare `python3` is not reliably on PATH, which is why the wrapper exists.
- Editing `.claude/settings.json` while a session is running may not take effect until `/hooks` is reopened or Claude Code restarts (the settings watcher only tracks files that existed at session start).
- `.gitignore` uses a `.claude/*` allowlist: `settings.json` and `hooks/` are committed (team-wide), while personal `settings.local.json` and Claude's session-local state (`scheduled_tasks.lock`, `checkpoints/`, …) stay out of git. New files under `.claude/hooks/` are tracked automatically.
