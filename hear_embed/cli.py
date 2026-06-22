"""``hear-embed`` command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich_argparse import RichHelpFormatter

from .audio import AUDIO_EXTENSIONS, iter_audio_files
from .embedder import DEFAULT_MODEL_ID
from .pipeline import embed_file
from .writers import make_writer


class _RichHelp(RichHelpFormatter):
    """Colorized ``--help`` with rich-argparse's markup parsing turned off.

    The CLI's help strings are plain text (e.g. ``[0, 1)``, ``<out>``). With
    rich-argparse's default ``help_markup=True`` a string like ``[parquet|npz]``
    would be parsed as a Rich tag and its bracketed content silently dropped;
    disabling markup makes all help text render literally.
    """

    help_markup = False
    text_markup = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hear-embed",
        description=(
            "Embed health-acoustic recordings with Google HeAR. Long recordings "
            "are windowed into 2-second clips; each clip yields a 512-dim vector."
        ),
        # Colorized --help via rich-argparse (see _RichHelp). The parser,
        # arguments, main(), and exit codes are unchanged from plain argparse.
        formatter_class=_RichHelp,
    )
    parser.add_argument(
        "input", type=Path, help="Audio file or directory of recordings."
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("embeddings.parquet"),
        help="Output path (default: embeddings.parquet).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("parquet", "npz"),
        default="parquet",
        help="parquet: streamed single file. npz: <out>.npy + <out>.csv.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Fractional overlap between consecutive windows, in [0, 1). Default 0.",
    )
    parser.add_argument(
        "--pool",
        choices=("none", "mean"),
        default="none",
        help="none: one vector per window. mean: one averaged vector per file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Clips per forward pass."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (e.g. cuda, cpu). Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id."
    )
    parser.add_argument(
        "--extensions",
        default=",".join(AUDIO_EXTENSIONS),
        help="Comma-separated audio extensions to scan when input is a directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    extensions = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (x.strip() for x in args.extensions.split(","))
        if e
    )
    files = iter_audio_files(args.input, extensions=extensions)
    if not files:
        print(f"No audio files found under {args.input}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} file(s) to embed.", file=sys.stderr)

    # Importing/loading the model is deferred until we know there's work to do,
    # and surfaces the gating requirement with an actionable message.
    try:
        from .embedder import HearEmbedder

        embedder = HearEmbedder(model_id=args.model, device=args.device)
    except Exception as exc:  # noqa: BLE001 - turn any load failure into guidance.
        print(f"Failed to load model {args.model!r}: {exc}", file=sys.stderr)
        print(
            "HeAR is gated. Accept the terms at "
            "https://huggingface.co/google/hear-pytorch and run "
            "`huggingface-cli login` (or set HF_TOKEN), then retry.",
            file=sys.stderr,
        )
        return 2

    try:
        from tqdm import tqdm

        progress = tqdm(files, unit="file")
    except ImportError:
        progress = files

    failures = 0
    with make_writer(args.out, args.format) as writer:
        for path in progress:
            try:
                vectors, metadata = embed_file(
                    path,
                    embedder,
                    overlap=args.overlap,
                    batch_size=args.batch_size,
                    pool=args.pool,
                )
                writer.write(vectors, metadata)
            except Exception as exc:  # noqa: BLE001 - skip a bad file, keep going.
                failures += 1
                print(f"  skipped {path}: {exc}", file=sys.stderr)

        rows = writer.rows_written

    out_desc = (
        args.out if args.format == "parquet" else f"{args.out.with_suffix('')}.npy/.csv"
    )
    print(
        f"Wrote {rows} embedding(s) from {len(files) - failures}/{len(files)} file(s) "
        f"to {out_desc}." + (f" {failures} file(s) skipped." if failures else ""),
        file=sys.stderr,
    )
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
