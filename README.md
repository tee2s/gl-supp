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

Run the PyTorch/mach version and save both the comparison plot and an MP4 video:

```bash
python beamform_torch.py 115601.mat --start 40 --stop 70 --step 3 --mode both --channel-skip 16 --device cuda --out-format both --out-dir /work/users/t/i/tis/data/jhu_spatiotemporal/beamformed_videos/skip16
```

Run the PyTorch/mach version and save an MP4 video for the entire sequence:

```bash
python beamform_torch.py 115601.mat --start 0 --stop 100 --step 1 --channel-skip 16 --device cuda --out-format video --out-dir /work/users/t/i/tis/data/jhu_spatiotemporal/beamformed_videos/skip16
```
