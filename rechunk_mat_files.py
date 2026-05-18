#!/usr/bin/env python3
"""Rechunk HDF5-backed .mat files for faster frame loading."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import h5py
from rf_data import get_rf_dataset


DEFAULT_N_TX = 128
DEFAULT_SAMPLES_PER_TX = 1920


def rechunk_file(
    src_path: Path,
    dst_path: Path,
    *,
    n_useful: int,
    frames_per_batch: Optional[int],
    overwrite: bool,
) -> None:
    if dst_path.exists() and not overwrite:
        print(f"skipping existing output: {dst_path}")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        src_dset = get_rf_dataset(src)
        n_frames, n_rx, n_samples = src_dset.shape
        output_samples = min(n_useful, n_samples)
        batch_size = frames_per_batch or n_frames

        print(
            f"rechunking {src_path.name}: "
            f"source_shape={src_dset.shape}, output_shape={(n_frames, n_rx, output_samples)}, "
            f"batch_size={batch_size}"
        )

        t0 = time.perf_counter()
        with h5py.File(dst_path, "w") as dst:
            out = dst.create_dataset(
                "rf",
                shape=(n_frames, n_rx, output_samples),
                dtype=src_dset.dtype,
                chunks=(1, n_rx, output_samples),
                compression=None,
            )

            for start in range(0, n_frames, batch_size):
                stop = min(start + batch_size, n_frames)
                out[start:stop, :, :] = src_dset[start:stop, :, :output_samples]

        elapsed = time.perf_counter() - t0
        print(
            f"wrote {dst_path} from {src_path.name} "
            f"shape={(n_frames, n_rx, output_samples)} in {elapsed:.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rechunk all .mat files in a source directory into HDF5 files with "
            "an 'rf' dataset, frame-sized chunks, and no compression."
        )
    )
    parser.add_argument("src_path", type=Path, help="Directory containing source .mat files")
    parser.add_argument("dst_path", type=Path, help="Directory to write rechunked .mat files")
    parser.add_argument(
        "--n-tx",
        type=int,
        default=DEFAULT_N_TX,
        help=f"Number of transmit events to keep (default: {DEFAULT_N_TX})",
    )
    parser.add_argument(
        "--samples-per-tx",
        type=int,
        default=DEFAULT_SAMPLES_PER_TX,
        help=f"Samples per transmit event to keep (default: {DEFAULT_SAMPLES_PER_TX})",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=None,
        help=(
            "Number of frames to copy at once. Defaults to all frames, which is "
            "usually fastest for the original files chunked across all frames."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files that already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.src_path.is_dir():
        raise NotADirectoryError(f"source path is not a directory: {args.src_path}")
    if args.frames_per_batch is not None and args.frames_per_batch < 1:
        raise ValueError("--frames-per-batch must be at least 1")

    mat_files = sorted(path for path in args.src_path.glob("*.mat") if path.is_file())
    if not mat_files:
        raise FileNotFoundError(f"no .mat files found in {args.src_path}")

    n_useful = args.n_tx * args.samples_per_tx
    print(f"found {len(mat_files)} .mat files in {args.src_path}")
    print(f"writing rechunked files to {args.dst_path}")
    print(f"keeping first {n_useful} samples with no compression")

    for src_path in mat_files:
        dst_path = args.dst_path / src_path.name
        rechunk_file(
            src_path,
            dst_path,
            n_useful=n_useful,
            frames_per_batch=args.frames_per_batch,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
