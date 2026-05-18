# Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

## Usage

Install dependencies with uv:

```bash
uv sync
```

Run the original beamforming script using NumPy, for example:

```bash
python beamform.py 115601.mat --start 40 --stop 70 --step 3  --mode both --channel-skip 16
```

```bash
python beamform.py 115601.mat --start 0 --stop 100 --step 10  --mode both --channel-skip 16 --plot-img-data 
```

Run the PyTorch/mach version from a TOML config:

```bash
python beamform_torch.py configs/example_beamform.toml
```

The config file describes data paths, frame selection, beamforming options, runtime device, and outputs:

```toml
[data]
# Can be a single file or a directory containing .mat/.h5/.hdf5 RF files.
rf_path = "/path/to/rf_data"
params_file = "/path/to/setup.mat"

[frames]
start = 0
stop = 100
step = 1
batch_size = 10

[beamforming]
mode = "both"
channel_skip = 16
device = "cuda"
dtype = "float32"

[output]
format = "video"
dir = "/path/to/beamformed_videos/skip16"
```

The metadata can be an older `setup.mat` file or a MATLAB v7.3/HDF5 data file with embedded acquisition groups. If `rf_path` is a directory, every `.mat`, `.h5`, and `.hdf5` file containing `rf` or `RcvData` is processed. If the data and metadata are in the same file, omit `params_file`; each RF file will be used as its own metadata source. Relative paths are resolved relative to the TOML file.

## Code organization

The PyTorch beamforming path is split into small modules with clear ownership:

- `beamform_torch.py` is the main executable pipeline. It reads a TOML run config, builds torch tensors from normalized acquisition metadata, precomputes transmit arrivals, beamforms frame batches, and writes plots or videos.
- `acquisition.py` defines the beamformer-facing acquisition interface. `AcquisitionParams` and `TxEventParams` describe the metadata the torch pipeline needs without exposing MATLAB-specific struct fields. It can load either the older `setup.mat` format or MATLAB v7.3/HDF5 data files with embedded `P`, `Receive`, `TX`, and `Trans` groups.
- `rf_data.py` owns RF data discovery and access. It accepts a single RF file or a directory of RF files, supports both rechunked files with an `rf` dataset and original MATLAB v7.3 files with `RcvData`, then yields frame batches through pinned CPU memory when using CUDA.
- `rechunk_mat_files.py` converts original HDF5-backed `.mat` files into faster frame-chunked files with an `rf` dataset.
- `beamform.py` is the older NumPy reference script and still reads the MATLAB setup structure directly.

This means new acquisition metadata sources should usually be added as new loaders in `acquisition.py` that return `AcquisitionParams`, rather than changing `beamform_torch.py`.
