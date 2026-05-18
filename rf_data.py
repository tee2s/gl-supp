from dataclasses import dataclass
from pathlib import Path

import h5py
import torch


RF_FILE_SUFFIXES = {".h5", ".hdf5", ".mat"}


@dataclass(frozen=True)
class RfFrameBatch:
    frame_numbers: list[int]
    data: torch.Tensor


def iter_rf_data_paths(rf_path: str | Path):
    """Yield RF data files from a single file path or a directory."""
    rf_path = Path(rf_path)
    if rf_path.is_file():
        yield rf_path
        return

    if not rf_path.is_dir():
        raise FileNotFoundError(f"RF path does not exist: {rf_path}")

    data_paths = sorted(
        path
        for path in rf_path.iterdir()
        if path.is_file() and _is_rf_data_file(path)
    )
    if not data_paths:
        raise FileNotFoundError(f"no RF data files found in directory: {rf_path}")

    yield from data_paths


def _is_rf_data_file(path: Path) -> bool:
    if path.suffix.lower() not in RF_FILE_SUFFIXES or not h5py.is_hdf5(path):
        return False

    with h5py.File(path, "r") as f:
        return "rf" in f or "RcvData" in f


def get_rf_dataset(h5_file: h5py.File) -> h5py.Dataset:
    """Return RF data from either rechunked files or original MATLAB v7.3 files."""
    if "rf" in h5_file:
        return h5_file["rf"]

    if "RcvData" not in h5_file:
        raise KeyError("could not find 'rf' or 'RcvData' in input file")

    return h5_file[h5_file["RcvData"][0, 0]]


def load_rf_frame_batch(
    dset: h5py.Dataset,
    *,
    frame_start: int,
    frame_stop: int,
    frame_step: int,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Load HDF5 RF frames through pinned CPU memory, then copy to the target device."""
    frame_count = len(range(frame_start, frame_stop, frame_step))
    if frame_count == 0:
        return None

    _, n_rx, n_available_samples = dset.shape
    n_samples = min(n_samples, n_available_samples)
    use_pinned_memory = device.type == "cuda"

    x_cpu = torch.empty(
        (frame_count, n_rx, n_samples),
        dtype=torch.int16,
        pin_memory=use_pinned_memory,
    )
    dset.read_direct(
        x_cpu.numpy(),
        source_sel=(slice(frame_start, frame_stop, frame_step), slice(None), slice(0, n_samples)),
    )

    x_device = x_cpu.to(device, non_blocking=use_pinned_memory)
    return x_device.to(dtype=dtype).contiguous()


def iter_rf_frame_batches(
    data_path,
    *,
    frame_indices: list[int],
    frame_batch_size: int,
    frame_step: int,
    n_samples: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Yield RF frame batches loaded via pinned CPU memory and copied to device."""
    with h5py.File(data_path, "r") as f:
        rf_dset = get_rf_dataset(f)
        print(
            "RF dataset: "
            f"shape={rf_dset.shape}, dtype={rf_dset.dtype}, "
            f"chunks={rf_dset.chunks}, compression={rf_dset.compression}"
        )

        for batch_offset in range(0, len(frame_indices), frame_batch_size):
            batch_frame_numbers = frame_indices[batch_offset : batch_offset + frame_batch_size]
            batch_start = batch_frame_numbers[0]
            batch_stop = batch_frame_numbers[-1] + frame_step
            print(f"Loading frame batch {batch_frame_numbers[0]} to {batch_frame_numbers[-1]}...")

            rf_batch = load_rf_frame_batch(
                rf_dset,
                frame_start=batch_start,
                frame_stop=batch_stop,
                frame_step=frame_step,
                n_samples=n_samples,
                device=device,
                dtype=dtype,
            )

            if rf_batch is None:
                continue

            yield RfFrameBatch(
                frame_numbers=batch_frame_numbers[: rf_batch.shape[0]],
                data=rf_batch,
            )
