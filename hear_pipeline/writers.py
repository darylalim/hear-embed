"""Output writers for embeddings + metadata.

Parquet is streamed (one row group per file) so embedding a large corpus never
holds every vector in memory at once. The npz writer accumulates and is meant
for small/medium runs where a single ``.npy`` + ``.csv`` pair is convenient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .embedder import EMBEDDING_DIM
from .pipeline import ClipMetadata


def _parquet_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source_file", pa.string()),
            ("clip_index", pa.int32()),
            ("start_sample", pa.int64()),
            ("start_sec", pa.float64()),
            ("end_sec", pa.float64()),
            ("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )


class ParquetEmbeddingWriter:
    """Streams ``(vectors, metadata)`` batches to a single Parquet file."""

    def __init__(self, path: str | Path) -> None:
        import pyarrow.parquet as pq

        self._pa = __import__("pyarrow")
        self._schema = _parquet_schema()
        self._writer = pq.ParquetWriter(str(path), self._schema)
        self.rows_written = 0

    def write(self, vectors: np.ndarray, metadata: list[ClipMetadata]) -> None:
        if len(metadata) == 0:
            return
        pa = self._pa
        table = pa.table(
            {
                "source_file": [m.source_file for m in metadata],
                "clip_index": [m.clip_index for m in metadata],
                "start_sample": [m.start_sample for m in metadata],
                "start_sec": [m.start_sec for m in metadata],
                "end_sec": [m.end_sec for m in metadata],
                "embedding": pa.array(
                    list(np.asarray(vectors, dtype=np.float32)),
                    type=pa.list_(pa.float32(), EMBEDDING_DIM),
                ),
            },
            schema=self._schema,
        )
        self._writer.write_table(table)
        self.rows_written += len(metadata)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "ParquetEmbeddingWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class NpzEmbeddingWriter:
    """Accumulates everything, then writes ``<stem>.npy`` + ``<stem>.csv``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._vectors: list[np.ndarray] = []
        self._metadata: list[ClipMetadata] = []
        self.rows_written = 0

    def write(self, vectors: np.ndarray, metadata: list[ClipMetadata]) -> None:
        if len(metadata) == 0:
            return
        self._vectors.append(np.asarray(vectors, dtype=np.float32))
        self._metadata.extend(metadata)
        self.rows_written += len(metadata)

    def close(self) -> None:
        import csv

        npy_path = self._path.with_suffix(".npy")
        csv_path = self._path.with_suffix(".csv")
        stacked = (
            np.concatenate(self._vectors, axis=0)
            if self._vectors
            else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        )
        np.save(npy_path, stacked)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["row", "source_file", "clip_index", "start_sample", "start_sec", "end_sec"]
            )
            for row, m in enumerate(self._metadata):
                writer.writerow(
                    [row, m.source_file, m.clip_index, m.start_sample, m.start_sec, m.end_sec]
                )

    def __enter__(self) -> "NpzEmbeddingWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_writer(path: str | Path, fmt: str):
    if fmt == "parquet":
        return ParquetEmbeddingWriter(path)
    if fmt == "npz":
        return NpzEmbeddingWriter(path)
    raise ValueError(f"unknown format {fmt!r} (expected 'parquet' or 'npz')")
