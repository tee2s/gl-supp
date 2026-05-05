# Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

Spatiotemporal Deep Learning for Grating-Lobe Suppression in Sparse-Array Ultrasound Imaging

## Usage

Install dependencies with uv:

```bash
uv sync
```

Run the beamforming script, for example:

```bash
python beamform.py 115601.mat --start 40 --stop 70 --step 3  --mode both --channel-skip 16
python beamform_torch.py 115601.mat --start 40 --stop 70 --step 3  --mode both --channel-skip 16
```
