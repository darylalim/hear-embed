"""Unit tests for output writers — torch-free, no model/HearEmbedder needed.

Exercises ``ParquetEmbeddingWriter``, ``NpzEmbeddingWriter`` and ``make_writer``
using hand-built ``ClipMetadata`` lists and fake ``(n, 512)`` float32 vectors.
Nothing here imports torch/transformers or loads a model — only the pure-Python
serialization paths are tested.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from hear_embed.embedder import EMBEDDING_DIM
from hear_embed.pipeline import ClipMetadata
from hear_embed.writers import (
    NpzEmbeddingWriter,
    ParquetEmbeddingWriter,
    make_writer,
)

PARQUET_COLUMNS = [
    "source_file",
    "clip_index",
    "start_sample",
    "start_sec",
    "end_sec",
    "embedding",
]
CSV_HEADER = [
    "row",
    "source_file",
    "clip_index",
    "start_sample",
    "start_sec",
    "end_sec",
]


def _vectors(n: int, *, seed: int = 0) -> np.ndarray:
    """An ``(n, 512)`` float32 vector array with reproducible contents."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)


def _metadata(n: int, *, source: str = "rec.wav") -> list[ClipMetadata]:
    """``n`` ClipMetadata rows with distinct, predictable field values."""
    return [
        ClipMetadata(
            source_file=source,
            clip_index=i,
            start_sample=i * 16000,
            start_sec=float(i),
            end_sec=float(i) + 2.0,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# ParquetEmbeddingWriter
# --------------------------------------------------------------------------- #


def test_parquet_writes_expected_schema_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    vectors = _vectors(3)
    metadata = _metadata(3)

    with ParquetEmbeddingWriter(path) as writer:
        writer.write(vectors, metadata)
        assert writer.rows_written == 3

    table = pq.read_table(path)
    assert table.num_rows == 3
    # Column order/names must match the documented public schema exactly.
    assert table.column_names == PARQUET_COLUMNS

    embeddings = table.column("embedding").to_pylist()
    # Every embedding round-trips as a length-512 list of floats.
    assert all(len(e) == EMBEDDING_DIM for e in embeddings)

    # Metadata scalar columns round-trip by value (and order is preserved).
    assert table.column("clip_index").to_pylist() == [0, 1, 2]
    assert table.column("start_sec").to_pylist() == [0.0, 1.0, 2.0]
    assert table.column("source_file").to_pylist() == ["rec.wav"] * 3

    # Embedding values themselves survive the float32 round trip.
    np.testing.assert_array_equal(np.asarray(embeddings, dtype=np.float32), vectors)


def test_parquet_streaming_two_writes_accumulate_rows(tmp_path: Path) -> None:
    path = tmp_path / "stream.parquet"

    with ParquetEmbeddingWriter(path) as writer:
        writer.write(_vectors(2, seed=1), _metadata(2))
        assert writer.rows_written == 2
        writer.write(_vectors(3, seed=2), _metadata(3, source="other.wav"))
        # rows_written accumulates across streaming write() calls.
        assert writer.rows_written == 5

    pf = pq.ParquetFile(path)
    # One row group per write() call — streaming, not a single buffered table.
    assert pf.num_row_groups == 2
    assert pf.metadata.num_rows == 5

    table = pf.read()
    assert table.num_rows == 5
    assert (
        table.column("source_file").to_pylist() == ["rec.wav"] * 2 + ["other.wav"] * 3
    )


def test_parquet_empty_write_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"

    with ParquetEmbeddingWriter(path) as writer:
        writer.write(np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
        # Zero-length metadata is a no-op: nothing counted.
        assert writer.rows_written == 0

    table = pq.read_table(path)
    assert table.num_rows == 0
    # Even an empty file still carries the full schema.
    assert table.column_names == PARQUET_COLUMNS


# --------------------------------------------------------------------------- #
# NpzEmbeddingWriter
# --------------------------------------------------------------------------- #


def test_npz_writes_npy_and_csv(tmp_path: Path) -> None:
    path = tmp_path / "bundle"  # writer derives .npy / .csv from the stem
    vectors = _vectors(3)
    metadata = _metadata(3)

    with NpzEmbeddingWriter(path) as writer:
        writer.write(vectors, metadata)
        assert writer.rows_written == 3

    npy_path = path.with_suffix(".npy")
    csv_path = path.with_suffix(".csv")
    assert npy_path.exists() and csv_path.exists()

    stacked = np.load(npy_path)
    assert stacked.shape == (3, EMBEDDING_DIM)
    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(stacked, vectors)

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER  # header line
    assert len(rows) == 1 + 3  # header + one row per metadata
    # Row contents round-trip. String/int columns are compared directly; the
    # float start_sec/end_sec columns are parsed and compared numerically so the
    # test pins the round-tripped VALUE, not str(float)'s incidental formatting.
    for i, row in enumerate(rows[1:]):
        assert row[:4] == [str(i), "rec.wav", str(i), str(i * 16000)]
        assert float(row[4]) == float(i)  # start_sec
        assert float(row[5]) == float(i) + 2.0  # end_sec


def test_npz_streaming_two_writes_concatenate(tmp_path: Path) -> None:
    path = tmp_path / "multi"
    v1 = _vectors(2, seed=1)
    v2 = _vectors(3, seed=2)

    with NpzEmbeddingWriter(path) as writer:
        writer.write(v1, _metadata(2))
        writer.write(v2, _metadata(3, source="b.wav"))
        assert writer.rows_written == 5

    stacked = np.load(path.with_suffix(".npy"))
    assert stacked.shape == (5, EMBEDDING_DIM)
    # Accumulated vectors are concatenated in write() order.
    np.testing.assert_array_equal(stacked, np.concatenate([v1, v2], axis=0))

    with open(path.with_suffix(".csv"), newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 5
    # The "row" column is a global 0..n-1 index across both write() batches.
    assert [r[0] for r in rows[1:]] == ["0", "1", "2", "3", "4"]
    # clip_index is the PER-FILE index, so it restarts at 0 for the second batch
    # ([0,1] then [0,1,2]) — distinct from the global "row" column above.
    assert [r[2] for r in rows[1:]] == ["0", "1", "0", "1", "2"]
    assert [r[1] for r in rows[1:]] == ["rec.wav", "rec.wav", "b.wav", "b.wav", "b.wav"]


def test_npz_empty_write_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "nothing"

    with NpzEmbeddingWriter(path) as writer:
        writer.write(np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
        assert writer.rows_written == 0

    stacked = np.load(path.with_suffix(".npy"))
    # Empty run still emits an (0, 512) float32 array and a header-only CSV.
    assert stacked.shape == (0, EMBEDDING_DIM)
    assert stacked.dtype == np.float32

    with open(path.with_suffix(".csv"), newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [CSV_HEADER]


# --------------------------------------------------------------------------- #
# README "Using the embeddings" read-back recipes
# --------------------------------------------------------------------------- #


def test_parquet_readback_recipe_matches_readme(tmp_path: Path) -> None:
    # Locks the "Using the embeddings" snippet: load vectors back with pyarrow +
    # numpy only (no pandas). Guards the exact documented calls — a pyarrow change
    # to to_numpy(zero_copy_only=...) or Table.select would fail here, not silently
    # break the README example.
    path = tmp_path / "emb.parquet"
    vectors = _vectors(4)
    with ParquetEmbeddingWriter(path) as writer:
        writer.write(vectors, _metadata(4))

    t = pq.read_table(path)
    X = np.stack(t["embedding"].to_numpy(zero_copy_only=False))  # the documented line
    meta = t.select(["source_file", "clip_index", "start_sec", "end_sec"])

    assert X.shape == (4, EMBEDDING_DIM)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, vectors)  # values survive the round trip
    # The metadata sub-table is pandas-free and row-aligned with X.
    assert meta.column_names == ["source_file", "clip_index", "start_sec", "end_sec"]
    assert meta.num_rows == X.shape[0]
    # The cosine-similarity follow-on the README shows yields an (n, n) matrix.
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert (unit @ unit.T).shape == (4, 4)


def test_npz_readback_row_column_indexes_into_npy(tmp_path: Path) -> None:
    # Locks the README claim that the CSV's leading `row` column is the index into
    # the .npy matrix (X[row] <-> that CSV line), verified across two write batches.
    path = tmp_path / "bundle"
    v1, v2 = _vectors(2, seed=1), _vectors(3, seed=2)
    with NpzEmbeddingWriter(path) as writer:
        writer.write(v1, _metadata(2))
        writer.write(v2, _metadata(3, source="b.wav"))

    X = np.load(path.with_suffix(".npy"))
    with open(path.with_suffix(".csv"), newline="") as f:
        lines = list(csv.DictReader(f))
    expected = np.concatenate([v1, v2], axis=0)
    assert [line["row"] for line in lines] == [str(i) for i in range(len(expected))]
    for line in lines:
        r = int(line["row"])
        np.testing.assert_array_equal(X[r], expected[r])  # X[row] is that line's vector


# --------------------------------------------------------------------------- #
# make_writer
# --------------------------------------------------------------------------- #


def test_make_writer_parquet(tmp_path: Path) -> None:
    writer = make_writer(tmp_path / "a.parquet", "parquet")
    try:
        assert isinstance(writer, ParquetEmbeddingWriter)
    finally:
        writer.close()  # ParquetWriter must be closed to flush a valid file.


def test_make_writer_npz(tmp_path: Path) -> None:
    writer = make_writer(tmp_path / "a", "npz")
    assert isinstance(writer, NpzEmbeddingWriter)
    writer.close()


def test_make_writer_unknown_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown format"):
        make_writer(tmp_path / "a", "json")
