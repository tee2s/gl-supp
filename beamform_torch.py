import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import torch
from mach import beamform
from mach.kernel import InterpolationType
from scipy.io import loadmat


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def resolve_dtype(dtype_name):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def as_contiguous_tensor(value, *, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype).contiguous()


def ensure_torch_mach_compat():
    if hasattr(torch.Tensor, "astype"):
        return

    def astype(self, dtype, copy=False):
        if self.dtype == dtype and not copy:
            return self
        return self.to(dtype=dtype, copy=copy)

    # mach 0.1.1 documents PyTorch support but its wrapper calls Array API-style
    # `.astype(...)` so this adds an .astype(...) method to torch.Tensor at runtime.
    torch.Tensor.astype = astype


def torch_hilbert_envelope(image, *, dim=0):
    spectrum = torch.fft.fft(image, dim=dim)
    n = image.shape[dim]

    multiplier = torch.zeros(n, dtype=spectrum.dtype, device=image.device)
    if n % 2 == 0:
        multiplier[0] = 1
        multiplier[n // 2] = 1
        multiplier[1 : n // 2] = 2
    else:
        multiplier[0] = 1
        multiplier[1 : (n + 1) // 2] = 2

    shape = [1] * image.ndim
    shape[dim] = n
    analytic = torch.fft.ifft(spectrum * multiplier.reshape(shape), dim=dim)
    return torch.abs(analytic)


def log_compress_for_display(image, *, dynamic_range, eps=1e-12):
    image = torch.clamp(image, min=eps)
    image_max = torch.clamp(torch.amax(image), min=eps)
    log_image = 20 * torch.log10(image / image_max)
    return torch.clamp(log_image, min=-dynamic_range, max=0).detach().cpu().numpy()


def precompute_tx_arrivals(*, tx_events, elem_pos_m_full, scan_coords_m, fc, lam, sound_speed_m_s, dtype):
    focal_points = []
    t_focus_values = []
    
    # loop to support different Apod mask shapes (might be able to remove)
    for tx_i in tx_events:
        apod = torch.as_tensor(tx_i.Apod, device=elem_pos_m_full.device) > 0
        delay_s = as_contiguous_tensor(tx_i.Delay, device=elem_pos_m_full.device, dtype=dtype) / fc
        elem_tx = elem_pos_m_full[apod]
        delay_tx = delay_s[apod]

        x_f = tx_i.Origin[0] * lam
        z_f = tx_i.focus * lam
        focal_point = torch.tensor([x_f, 0.0, z_f], device=elem_pos_m_full.device, dtype=dtype)

        dist_tx_to_focus = torch.linalg.vector_norm(elem_tx - focal_point[None, :], dim=-1)
        t_focus = torch.mean(delay_tx + dist_tx_to_focus / sound_speed_m_s)

        focal_points.append(focal_point)
        t_focus_values.append(t_focus)

    focal_points = torch.stack(focal_points, dim=0)
    t_focus_values = torch.stack(t_focus_values, dim=0)

    dist_pixel_to_focus = torch.linalg.vector_norm(
        scan_coords_m[None, :, :] - focal_points[:, None, :],
        dim=-1,
    )
    is_post_focal = scan_coords_m[None, :, 2] >= focal_points[:, None, 2]
    sign = torch.where(
        is_post_focal,
        torch.ones((), device=scan_coords_m.device, dtype=dtype),
        -torch.ones((), device=scan_coords_m.device, dtype=dtype),
    )
    tx_arrivals = t_focus_values[:, None] + (sign * dist_pixel_to_focus / sound_speed_m_s)
    return tx_arrivals


def main():
    # --- 1. Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Beamform ultrasound frames with optional Rx downsampling.")
    parser.add_argument(
        "data_file",
        nargs="?",
        default="131626.mat",
        help="Input RF data .mat file, either a full path or a filename in the data directory",
    )
    parser.add_argument('--start', type=int, default=0, help="Start frame index")
    parser.add_argument('--stop', type=int, default=2, help="Stop frame index (exclusive)")
    parser.add_argument('--step', type=int, default=1, help="Frame step size")
    parser.add_argument('--mode', type=str, choices=['full', 'sparse', 'both'], default='both', 
                        help="Beamforming mode: 'full', 'sparse', or 'both'")
    parser.add_argument('--channel-skip', type=int, default=2, 
                        help="Stride for downsampling receive channels (e.g., 2 means use every 2nd channel)")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for surrounding numeric work: 'auto', 'cpu', 'cuda', or a device like 'cuda:0'",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Torch dtype for geometry and timing calculations",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if args.channel_skip <= 0:
        raise ValueError("--channel-skip must be positive.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    beamform_dtype = torch.float32
    ensure_torch_mach_compat()
    print(f"Using torch device={device}, dtype={dtype}")

    # --- 2. Load Data and Acquisition Parameters ---
    base_dir = Path("/proj/yzlinlab/projects/jhu_spatiotemporal/data260421")
    data_path = Path(args.data_file)
    
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    print(f"Processing Data in {data_path}")
    
    setup_path = base_dir / "setup.mat"
    setup = loadmat(setup_path, squeeze_me=True, struct_as_record=False)
    
    with h5py.File(data_path, 'r') as f:
        ref = f['RcvData'][0, 0]

        # Constants
        c = 1540.0
        fc = float(setup["Trans"].frequency * 1e6)
        fs = 4 * fc
        lam = c / fc
        n_rx = 128
        n_tx = 128
        n_total_samples = 245760 # 1920 * 128
        
        receive0 = setup["Receive"][0]
        start_depth = receive0.startDepth
        end_depth = receive0.endDepth
        n_valid_samples_per_tx = int(2 * (end_depth - start_depth) * (fs / fc))

        print(f"Loading frames {args.start} to {args.stop} with step {args.step}...")
        rf_full = f[ref][args.start:args.stop:args.step, :, :n_total_samples]

    n_frames_loaded = rf_full.shape[0]
    if n_frames_loaded == 0:
        raise ValueError("No frames were loaded; check --start, --stop, and --step.")

    # Reshape: (n_frames, n_rx, n_tx, n_samples_per_tx)
    rf_full = as_contiguous_tensor(rf_full, device=device, dtype=beamform_dtype)
    rf_full = rf_full.reshape(n_frames_loaded, n_rx, n_tx, -1)
    rf_full = rf_full[:, :, :, :n_valid_samples_per_tx].contiguous()

    print(f"Loaded RF shape: {tuple(rf_full.shape)}")

    # --- 3. Image Grid Construction ---
    pdata = setup["PData"]
    origin = as_contiguous_tensor(pdata.Origin, device=device, dtype=dtype)
    delta = as_contiguous_tensor(pdata.PDelta, device=device, dtype=dtype)
    size = torch.as_tensor(pdata.Size)
    nz, nx, ny = int(size[0].item()), int(size[1].item()), int(size[2].item())

    x = origin[0] + torch.arange(nx, device=device, dtype=dtype) * delta[0]
    z = origin[2] + torch.arange(nz, device=device, dtype=dtype) * delta[2]
    X, Z = torch.meshgrid(x, z, indexing="xy")
    
    scan_coords_lambda = torch.stack([X.ravel(), torch.zeros_like(X).ravel(), Z.ravel()], dim=1)
    scan_coords_m = (scan_coords_lambda * lam).contiguous()
    scan_coords_m_bf = scan_coords_m.to(dtype=beamform_dtype).contiguous()

    # --- 4. Compute Transmit Arrival Times and Beamform ---
    rx_start_s = 0.0
    elem_pos_m_full = as_contiguous_tensor(setup["Trans"].ElementPos[:, :3], device=device, dtype=dtype) * lam
    elem_pos_m_full_bf = elem_pos_m_full.to(dtype=beamform_dtype).contiguous()
    
    # Downsample receive element positions to match channel downsampling
    elem_pos_m_sparse_bf = elem_pos_m_full_bf[::args.channel_skip, :].contiguous()

    print("Precomputing transmit arrival times...")
    tx_arrivals = precompute_tx_arrivals(
        tx_events=setup["TX"],
        elem_pos_m_full=elem_pos_m_full,
        scan_coords_m=scan_coords_m,
        fc=fc,
        lam=lam,
        sound_speed_m_s=c,
        dtype=dtype,
    ).to(dtype=beamform_dtype).contiguous()

   
    rf_events_full = (
        rf_full.permute(2, 1, 3, 0).contiguous()
        if args.mode in ['full', 'both']
        else None
    )
    rf_events_sparse = (
        rf_full[:, ::args.channel_skip, :, :].permute(2, 1, 3, 0).contiguous()
        if args.mode in ['sparse', 'both']
        else None
    )
    
    hri_full = (
        torch.zeros((scan_coords_m.shape[0], n_frames_loaded), device=device, dtype=beamform_dtype)
        if args.mode in ['full', 'both']
        else None
    )
    hri_sparse = (
        torch.zeros((scan_coords_m.shape[0], n_frames_loaded), device=device, dtype=beamform_dtype)
        if args.mode in ['sparse', 'both']
        else None
    )

    print(f"Beamforming {n_frames_loaded} frame(s) across {n_tx} transmit event(s)...")
    beamform_kwargs = dict(
        scan_coords_m=scan_coords_m_bf,
        rx_start_s=rx_start_s,
        sampling_freq_hz=fs,
        sound_speed_m_s=c,
        interp_type=InterpolationType.Linear,
        f_number=1.0,
        tukey_alpha=0.1,
    )

    for i in range(n_tx):
        # Full beamforming
        if args.mode in ['full', 'both']:
            beamform(
                channel_data=rf_events_full[i],
                rx_coords_m=elem_pos_m_full_bf,
                tx_wave_arrivals_s=tx_arrivals[i],
                out=hri_full,
                f_number=1.0,
                tukey_alpha=0.1,
                **beamform_kwargs,
            )
        
        # Sparse beamforming (Downsampled Rx)
        if args.mode in ['sparse', 'both']:
            beamform(
                channel_data=rf_events_sparse[i],
                rx_coords_m=elem_pos_m_sparse_bf,
                tx_wave_arrivals_s=tx_arrivals[i],
                out=hri_sparse,
                **beamform_kwargs,
            )
            
    # Envelope detection
    envelopes_full = []
    envelopes_sparse = []

    if args.mode in ['full', 'both']:
        env_full = torch_hilbert_envelope(hri_full.reshape((nz, nx, n_frames_loaded)), dim=0)
        envelopes_full = [env_full[:, :, idx] for idx in range(n_frames_loaded)]
        
    if args.mode in ['sparse', 'both']:
        env_sparse = torch_hilbert_envelope(hri_sparse.reshape((nz, nx, n_frames_loaded)), dim=0)
        envelopes_sparse = [env_sparse[:, :, idx] for idx in range(n_frames_loaded)]

    # --- 5. Plot Side-by-Side ---
    print("Generating comparison plot...")
    dynamic_range = 50.0
    extent = [
        (x.min() * lam * 1e3).item(),
        (x.max() * lam * 1e3).item(),
        (z.max() * lam * 1e3).item(),
        (z.min() * lam * 1e3).item(),
    ]
    
    # Setup grid rows based on mode. squeeze=False ensures axes is always a 2D array [row, col]
    n_rows = 2 if args.mode == 'both' else 1
    fig, axes = plt.subplots(n_rows, n_frames_loaded, figsize=(7 * n_frames_loaded, 7 * n_rows), squeeze=False)
    
    row_sparse = 0 if args.mode in ['sparse', 'both'] else None
    row_full = 1 if args.mode == 'both' else (0 if args.mode == 'full' else None)

    for idx in range(n_frames_loaded):
        original_frame = args.start + idx * args.step
        
        # Plot Sparse
        if args.mode in ['sparse', 'both']:
            ax = axes[row_sparse, idx]
            img = envelopes_sparse[idx]
            log_img = log_compress_for_display(img, dynamic_range=dynamic_range)
            
            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"Sparse Rx (skip={args.channel_skip}) - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

        # Plot Full
        if args.mode in ['full', 'both']:
            ax = axes[row_full, idx]
            img = envelopes_full[idx]
            log_img = log_compress_for_display(img, dynamic_range=dynamic_range)
            
            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"Full Array - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

    plt.tight_layout()
    plot_filename = f"b_mode_{data_path.stem}_frames_{args.start}_to_{args.stop}_step_{args.step}_mode_{args.mode}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Done! Plot saved as '{plot_filename}'")

if __name__ == "__main__":
    main()