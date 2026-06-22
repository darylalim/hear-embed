"""Unit tests for the CLI — torch-free, no real model.

``hear_embed.cli.main`` lazily does ``from .embedder import HearEmbedder``
inside the function, so the shared ``use_fake_embedder`` fixture (see
``conftest.py``) patches that class with ``FakeEmbedder`` and we drive
``main(argv)`` directly, asserting its integer exit code and the files it
writes.
"""

from __future__ import annotations

import numpy as np
import pytest

from hear_embed.audio import CLIP_LENGTH, SAMPLE_RATE
from hear_embed.embedder import EMBEDDING_DIM


@pytest.mark.usefixtures("use_fake_embedder")
def test_success_writes_parquet_and_returns_0(tmp_path, write_wav):
    import pyarrow.parquet as pq

    from hear_embed.cli import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # Two valid 3 s recordings -> 2 windows each (no overlap) -> 4 rows total.
    write_wav(input_dir / "a.wav")
    write_wav(input_dir / "b.wav")
    out = tmp_path / "embeddings.parquet"

    code = main([str(input_dir), "--out", str(out), "--format", "parquet"])

    assert code == 0
    assert out.exists()
    # Read the parquet back to confirm the streamed rows landed on disk.
    table = pq.read_table(out)
    assert table.num_rows == 4  # 2 files * 2 windows each
    assert "embedding" in table.column_names


@pytest.mark.usefixtures("use_fake_embedder")
def test_npz_format_writes_npy_and_csv(tmp_path, write_wav):
    from hear_embed.cli import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    write_wav(input_dir / "a.wav")
    out = tmp_path / "embeddings"  # npz writer derives .npy/.csv from the stem

    code = main([str(input_dir), "--out", str(out), "--format", "npz"])

    assert code == 0
    npy_path = out.with_suffix(".npy")
    csv_path = out.with_suffix(".csv")
    assert npy_path.exists()
    assert csv_path.exists()
    # The .npy holds the stacked (n, 512) embedding matrix.
    stacked = np.load(npy_path)
    assert stacked.shape == (2, EMBEDDING_DIM)  # one 3 s file -> 2 windows
    # CSV has a header row plus one row per embedding.
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 1 + 2


@pytest.mark.usefixtures("use_fake_embedder")
def test_empty_dir_returns_1(tmp_path):
    from hear_embed.cli import main

    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    out = tmp_path / "out.parquet"

    code = main([str(input_dir), "--out", str(out), "--format", "parquet"])

    assert code == 1  # no matching files -> early exit before model load
    assert not out.exists()  # writer never opened


def test_model_load_failure_returns_2(monkeypatch, tmp_path, write_wav):
    from hear_embed.cli import main

    class BoomEmbedder:
        def __init__(self, model_id: str = "x", device: str | None = None) -> None:
            raise RuntimeError("gated repo / no token")

    monkeypatch.setattr("hear_embed.embedder.HearEmbedder", BoomEmbedder)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    write_wav(input_dir / "a.wav")
    out = tmp_path / "out.parquet"

    code = main([str(input_dir), "--out", str(out), "--format", "parquet"])

    assert code == 2  # model __init__ raised -> actionable-message exit
    assert not out.exists()  # we bail before the writer is opened


@pytest.mark.usefixtures("use_fake_embedder")
def test_partial_failure_returns_3_but_writes_good_rows(tmp_path, write_wav):
    import pyarrow.parquet as pq

    from hear_embed.cli import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    write_wav(input_dir / "good.wav")
    # A .wav extension over garbage bytes: passes the file scan, fails to load.
    (input_dir / "corrupt.wav").write_bytes(b"not a real wav file at all")
    out = tmp_path / "out.parquet"

    code = main([str(input_dir), "--out", str(out), "--format", "parquet"])

    assert code == 3  # at least one file skipped -> partial-failure exit
    assert out.exists()
    # The good file's rows are still persisted despite the bad sibling.
    table = pq.read_table(out)
    assert table.num_rows == 2  # only good.wav (3 s -> 2 windows) contributed


@pytest.mark.usefixtures("use_fake_embedder")
def test_extensions_filter_limits_scanned_files(tmp_path, write_wav):
    import pyarrow.parquet as pq

    from hear_embed.cli import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    write_wav(input_dir / "keep.wav")
    write_wav(input_dir / "skip.flac")  # excluded by --extensions wav
    out = tmp_path / "out.parquet"

    code = main(
        [
            str(input_dir),
            "--out",
            str(out),
            "--format",
            "parquet",
            "--extensions",
            "wav",  # bare ext (no dot) is normalized to ".wav" by the CLI
        ]
    )

    assert code == 0
    table = pq.read_table(out)
    assert table.num_rows == 2  # only the single .wav was scanned -> 2 windows


@pytest.mark.usefixtures("use_fake_embedder")
def test_extensions_filter_excluding_everything_returns_1(tmp_path, write_wav):
    from hear_embed.cli import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    write_wav(input_dir / "only.wav")
    out = tmp_path / "out.parquet"

    # Restrict to .flac while only a .wav exists -> nothing scanned.
    code = main([str(input_dir), "--out", str(out), "--extensions", "flac"])

    assert code == 1
    assert not out.exists()


def test_help_is_rich_formatted_and_exits_0(capsys):
    # Reflects the rich-argparse change: the parser opts into RichHelpFormatter,
    # and `--help` must render (without choking on bracket-y help text like
    # "[0, 1)") and exit 0 before any model load.
    from rich_argparse import RichHelpFormatter

    from hear_embed.cli import _build_parser, main

    assert _build_parser().formatter_class is RichHelpFormatter

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    # Atomic tokens, robust to Rich's line-wrapping: prog name, an option, model.
    assert "hear-embed" in out
    assert "--overlap" in out
    assert "HeAR" in out


def test_clip_length_constant_matches_window_size():
    # Guards the documented 2 s @ 16 kHz contract the CLI's windowing relies on.
    assert CLIP_LENGTH == 32000
    assert SAMPLE_RATE == 16000
