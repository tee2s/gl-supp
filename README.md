# Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

## Usage

Install dependencies with uv:

```bash
uv sync
```

## Code organization

The PyTorch beamforming path is split into small modules with clear ownership:

- `beamform_torch.py` is the main executable pipeline. It parses command-line options, builds torch tensors from normalized acquisition metadata, precomputes transmit arrivals, beamforms frame batches, and writes plots or videos.
- `acquisition.py` defines the beamformer-facing acquisition interface. `AcquisitionParams` and `TxEventParams` describe the metadata the torch pipeline needs without exposing MATLAB-specific struct fields. `load_mat_acquisition_params(...)` is the current adapter from `setup.mat` into that normalized format.
- `rf_data.py` owns RF data access. It supports both rechunked files with an `rf` dataset and original MATLAB v7.3 files with `RcvData`, then yields frame batches through pinned CPU memory when using CUDA.
- `rechunk_mat_files.py` converts original HDF5-backed `.mat` files into faster frame-chunked files with an `rf` dataset.
- `beamform.py` is the older NumPy reference script and still reads the MATLAB setup structure directly.

This means new acquisition metadata sources should usually be added as new loaders in `acquisition.py` that return `AcquisitionParams`, rather than changing `beamform_torch.py`.

Run the original beamforming script using NumPy, for example:

```bash
python beamform.py 115601.mat --start 40 --stop 70 --step 3  --mode both --channel-skip 16
```

```bash
python beamform.py 115601.mat --start 0 --stop 100 --step 10  --mode both --channel-skip 16 --plot-img-data 
```

Run the PyTorch/mach version and save both the comparison plot and an MP4 video:

```bash
python beamform_torch.py 115601.mat --start 40 --stop 70 --step 3 --mode both --channel-skip 16 --device cuda --out-format both --out-dir /work/users/t/i/tis/data/jhu_spatiotemporal/beamformed_videos/skip16
```

By default, `beamform_torch.py` reads `setup.mat` from the base data directory. To use a different metadata file, pass it explicitly:

```bash
python beamform_torch.py 115601.mat --setup-file /path/to/setup.mat --device cuda
```

Run the PyTorch/mach version and save an MP4 video for the entire sequence:

```bash
python beamform_torch.py 115601.mat --start 0 --stop 100 --step 1 --channel-skip 16 --device cuda --out-format video --out-dir /work/users/t/i/tis/data/jhu_spatiotemporal/beamformed_videos/skip16
```
