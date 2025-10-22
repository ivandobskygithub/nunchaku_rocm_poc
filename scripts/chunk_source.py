#!/usr/bin/env python3
"""Utility to emit manageable slices of large source files.

Some GPU kernel headers in this repository can grow beyond what
interactive tooling comfortably handles.  This helper prints a selected
range of lines (defaulting to 200-line windows) or writes the slices to
separate files so that editors with strict size limits can cope.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable


def read_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        # Fall back to binary mode for non-UTF8 files
        return path.read_bytes().decode("utf-8", errors="replace").splitlines()


def emit_chunk(lines: list[str], start: int, count: int) -> Iterable[str]:
    end = min(len(lines), start + count)
    width = len(str(end))
    for idx in range(start, end):
        yield f"{idx + 1:>{width}}| {lines[idx]}"


def save_chunks(lines: list[str], chunk: int, stem: pathlib.Path) -> None:
    for offset in range(0, len(lines), chunk):
        suffix = f"{offset + 1:06d}-{min(offset + chunk, len(lines)):06d}"
        out_path = stem.with_name(f"{stem.name}.{suffix}.part")
        out_path.write_text("\n".join(lines[offset : offset + chunk]) + "\n", encoding="utf-8")
        print(f"wrote {out_path}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, help="File to slice")
    parser.add_argument("--chunk", type=int, default=200, help="Number of lines per chunk")
    parser.add_argument(
        "--offset", type=int, default=0, help="Starting line offset (0-indexed) for printing"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of lines to print (defaults to --chunk when not splitting)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Write numbered .part files instead of printing to stdout",
    )

    args = parser.parse_args(argv)
    lines = read_lines(args.path)

    if args.split:
        save_chunks(lines, args.chunk, args.path)
        return 0

    count = args.count or args.chunk
    if count <= 0:
        parser.error("--count must be positive when provided")
    if args.offset < 0 or args.offset >= len(lines):
        parser.error("--offset must be within the file")

    for line in emit_chunk(lines, args.offset, count):
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
