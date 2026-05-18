import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from acquisition import load_mat_acquisition_params
from matplotlib.animation import FuncAnimation, writers
from mach import beamform
from mach.kernel import InterpolationType
from rf_data import iter_rf_frame_batches


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


def save_video(
    *,
    filename,
    mode,
    sparse_frames,
    full_frames,
    frame_numbers,
    channel_skip,
    dynamic_range,
    extent,
    fps=10,
):
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib ffmpeg writer is not available; install ffmpeg to save MP4 output.")

    n_rows = 2 if mode == 'both' else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 7 * n_rows), squeeze=False)

    row_sparse = 0 if mode in ['sparse', 'both'] else None
    row_full = 1 if mode == 'both' else (0 if mode == 'full' else None)
    images = []

    if mode in ['sparse', 'both']:
        ax = axes[row_sparse, 0]
        im = ax.imshow(sparse_frames[0], cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Axial (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")
        images.append(("sparse", im, ax))

    if mode in ['full', 'both']:
        ax = axes[row_full, 0]
        im = ax.imshow(full_frames[0], cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Axial (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")
        images.append(("full", im, ax))

    def update(frame_idx):
        artists = []
        frame_number = frame_numbers[frame_idx]
        for image_kind, im, ax in images:
            if image_kind == "sparse":
                im.set_data(sparse_frames[frame_idx])
                ax.set_title(f"Sparse Rx (skip={channel_skip}) - Frame {frame_number}")
            else:
                im.set_data(full_frames[frame_idx])
                ax.set_title(f"Full Array - Frame {frame_number}")
            artists.append(im)
        return artists

    update(0)
    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=len(frame_numbers), blit=False)
    animation.save(filename, writer="ffmpeg", fps=fps, dpi=200)
    plt.close(fig)


def precompute_tx_arrivals(*, tx_events, elem_pos_m_full, scan_coords_m, fc, lam, sound_speed_m_s, dtype):
    tx_arrivals = []
    tx_weights = []
    
    # Loop to support different Apod masks and focused apertures per transmit.
    for tx_i in tx_events:
        apod = torch.as_tensor(tx_i.apod, device=elem_pos_m_full.device) > 0
        delay_s = as_contiguous_tensor(tx_i.delay_cycles, device=elem_pos_m_full.device, dtype=dtype) / fc
        elem_tx = elem_pos_m_full[apod]
        delay_tx = delay_s[apod]

        x_f = tx_i.origin_wavelengths[0] * lam
        z_f = tx_i.focus_wavelengths * lam
        focal_point = torch.tensor([x_f, 0.0, z_f], device=elem_pos_m_full.device, dtype=dtype)

        dist_tx_to_focus = torch.linalg.vector_norm(elem_tx - focal_point[None, :], dim=-1)
        t_focus = torch.mean(delay_tx + dist_tx_to_focus / sound_speed_m_s)

        if elem_tx.numel() > 0:
            aperture_width = torch.amax(elem_tx[:, 0]) - torch.amin(elem_tx[:, 0])
            aperture_width = torch.clamp(aperture_width, min=1e-4)
        else:
            aperture_width = torch.tensor(1e-4, device=elem_pos_m_full.device, dtype=dtype)

        f_number = z_f / aperture_width
        beam_waist = lam * f_number
        rayleigh_range = torch.pi * (beam_waist**2) / lam

        z_diff = scan_coords_m[:, 2] - z_f
        dx = scan_coords_m[:, 0] - x_f

        epsilon_z = 1e-9
        z_diff_safe = torch.where(
            torch.abs(z_diff) < epsilon_z,
            torch.full_like(z_diff, epsilon_z),
            z_diff,
        )
        radius_curvature = z_diff_safe * (1.0 + (rayleigh_range / z_diff_safe) ** 2)
        gaussian_distance = z_diff + (dx**2) / (2.0 * radius_curvature)
        tx_arrivals.append(t_focus + gaussian_distance / sound_speed_m_s)

        beam_width = beam_waist * torch.sqrt(1.0 + (z_diff / rayleigh_range) ** 2)
        tx_weight = torch.sqrt(beam_waist / beam_width) * torch.exp(-(dx**2) / (beam_width**2))
        tx_weights.append(tx_weight)

    return torch.stack(tx_arrivals, dim=0), torch.stack(tx_weights, dim=0)


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
    parser.add_argument(
        "--out-format",
        choices=["plot", "video", "both"],
        default="plot",
        help="Output format: PNG plot, MP4 video, or both",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory where output PNG and MP4 files will be written",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=10,
        help="Number of frames to load and beamform at once",
    )
    parser.add_argument(
        "--setup-file",
        type=Path,
        default=None,
        help="Path to MATLAB setup metadata. Defaults to setup.mat in the base data directory.",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if args.channel_skip <= 0:
        raise ValueError("--channel-skip must be positive.")
    if args.frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    beamform_dtype = torch.float32
    ensure_torch_mach_compat()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using torch device={device}, dtype={dtype}")

    # --- 2. Load Data and Acquisition Parameters ---
    base_dir = Path("/proj/yzlinlab/projects/jhu_spatiotemporal/data260421")
    data_path = Path(args.data_file)
    
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    print(f"Processing Data in {data_path}")
    
    setup_path = args.setup_file or base_dir / "setup.mat"
    acquisition = load_mat_acquisition_params(setup_path)
    
    frame_indices = list(range(args.start, args.stop, args.step))
    if not frame_indices:
        raise ValueError("No frames were loaded; check --start, --stop, and --step.")

    c = acquisition.sound_speed_m_s
    fc = acquisition.center_frequency_hz
    fs = acquisition.sampling_frequency_hz
    lam = acquisition.wavelength_m
    n_rx = acquisition.n_rx
    n_tx = acquisition.n_tx
    n_total_samples = acquisition.n_total_samples
    n_valid_samples_per_tx = acquisition.n_valid_samples_per_tx

    # --- 3. Image Grid Construction ---
    origin = as_contiguous_tensor(acquisition.scan_origin_wavelengths, device=device, dtype=dtype)
    delta = as_contiguous_tensor(acquisition.scan_delta_wavelengths, device=device, dtype=dtype)
    nz, nx, ny = acquisition.scan_size

    x = origin[0] + torch.arange(nx, device=device, dtype=dtype) * delta[0]
    z = origin[2] + torch.arange(nz, device=device, dtype=dtype) * delta[2]
    X, Z = torch.meshgrid(x, z, indexing="xy")
    
    scan_coords_lambda = torch.stack([X.ravel(), torch.zeros_like(X).ravel(), Z.ravel()], dim=1)
    scan_coords_m = (scan_coords_lambda * lam).contiguous()
    scan_coords_m_bf = scan_coords_m.to(dtype=beamform_dtype).contiguous()

    # --- 4. Compute Transmit Arrival Times and Beamform ---
    elem_pos_m_full = as_contiguous_tensor(
        acquisition.element_positions_wavelengths,
        device=device,
        dtype=dtype,
    ) * lam
    elem_pos_m_full_bf = elem_pos_m_full.to(dtype=beamform_dtype).contiguous()
    
    # Downsample receive element positions to match channel downsampling
    elem_pos_m_sparse_bf = elem_pos_m_full_bf[::args.channel_skip, :].contiguous()

    print("Precomputing transmit arrival times and weights...")
    tx_arrivals, tx_weights = precompute_tx_arrivals(
        tx_events=acquisition.tx_events,
        elem_pos_m_full=elem_pos_m_full,
        scan_coords_m=scan_coords_m,
        fc=fc,
        lam=lam,
        sound_speed_m_s=c,
        dtype=dtype,
    )
    tx_arrivals = tx_arrivals.to(dtype=beamform_dtype).contiguous()
    tx_weights = tx_weights.to(dtype=beamform_dtype).contiguous()
    tx_weight_sum = tx_weights.sum(dim=0).reshape((nz, nx)).contiguous()

    dynamic_range = 50.0
    frame_numbers = []
    sparse_display_frames = []
    full_display_frames = []

    print(
        f"Beamforming {len(frame_indices)} frame(s) across {n_tx} transmit event(s) "
        f"in batches of {args.frame_batch_size}..."
    )
    beamform_kwargs = dict(
        scan_coords_m=scan_coords_m_bf,
        rx_start_s=acquisition.rx_start_s,
        sampling_freq_hz=fs,
        sound_speed_m_s=c,
        interp_type=InterpolationType.Linear,
        f_number=1.0,
        tukey_alpha=0.1,
    )

    for rf_frame_batch in iter_rf_frame_batches(
        data_path,
        frame_indices=frame_indices,
        frame_batch_size=args.frame_batch_size,
        frame_step=args.step,
        n_samples=n_total_samples,
        device=device,
        dtype=beamform_dtype,
    ):
        rf_batch = rf_frame_batch.data
        n_batch_frames = rf_batch.shape[0]
        batch_frame_numbers = rf_frame_batch.frame_numbers

        # Reshape: (n_frames, n_rx, n_tx, n_samples_per_tx)
        rf_batch = rf_batch.reshape(n_batch_frames, n_rx, n_tx, -1)
        rf_batch = rf_batch[:, :, :, :n_valid_samples_per_tx].contiguous()
        print(f"Loaded RF batch shape: {tuple(rf_batch.shape)}")

        rf_events_full = (
            rf_batch.permute(2, 1, 3, 0).contiguous()
            if args.mode in ['full', 'both']
            else None
        )
        rf_events_sparse = (
            rf_batch[:, ::args.channel_skip, :, :].permute(2, 1, 3, 0).contiguous()
            if args.mode in ['sparse', 'both']
            else None
        )

        # intermediate buffer, as mach beamformer does not support weighted summing into out
        hri_full = (
            torch.zeros((scan_coords_m.shape[0], n_batch_frames), device=device, dtype=beamform_dtype)
            if args.mode in ['full', 'both']
            else None
        )
        hri_sparse = (
            torch.zeros((scan_coords_m.shape[0], n_batch_frames), device=device, dtype=beamform_dtype)
            if args.mode in ['sparse', 'both']
            else None
        )
        lri_full = (
            torch.zeros_like(hri_full)
            if args.mode in ['full', 'both']
            else None
        )
        lri_sparse = (
            torch.zeros_like(hri_sparse)
            if args.mode in ['sparse', 'both']
            else None
        )

        for i in range(n_tx):
            tx_weight = tx_weights[i, :, None]

            # Full beamforming
            if args.mode in ['full', 'both']:
                lri_full.zero_()
                beamform(
                    channel_data=rf_events_full[i],
                    rx_coords_m=elem_pos_m_full_bf,
                    tx_wave_arrivals_s=tx_arrivals[i],
                    out=lri_full,
                    **beamform_kwargs,
                )
                hri_full += lri_full * tx_weight
            
            # Sparse beamforming (Downsampled Rx)
            if args.mode in ['sparse', 'both']:
                lri_sparse.zero_()
                beamform(
                    channel_data=rf_events_sparse[i],
                    rx_coords_m=elem_pos_m_sparse_bf,
                    tx_wave_arrivals_s=tx_arrivals[i],
                    out=lri_sparse,
                    **beamform_kwargs,
                )
                hri_sparse += lri_sparse * tx_weight
                
        # Envelope detection and display compression for this batch.
        if args.mode in ['full', 'both']:
            env_full = torch_hilbert_envelope(hri_full.reshape((nz, nx, n_batch_frames)), dim=0)
            env_full = env_full / (tx_weight_sum[:, :, None] + 1e-6)
            full_display_frames.extend(
                log_compress_for_display(env_full[:, :, idx], dynamic_range=dynamic_range)
                for idx in range(n_batch_frames)
            )
            
        if args.mode in ['sparse', 'both']:
            env_sparse = torch_hilbert_envelope(hri_sparse.reshape((nz, nx, n_batch_frames)), dim=0)
            env_sparse = env_sparse / (tx_weight_sum[:, :, None] + 1e-6)
            sparse_display_frames.extend(
                log_compress_for_display(env_sparse[:, :, idx], dynamic_range=dynamic_range)
                for idx in range(n_batch_frames)
            )

        frame_numbers.extend(batch_frame_numbers)
        del rf_batch, rf_events_full, rf_events_sparse, hri_full, hri_sparse, lri_full, lri_sparse

    if not frame_numbers:
        raise ValueError("No frames were loaded; check --start, --stop, and --step.")
    n_frames_loaded = len(frame_numbers)

    # --- 5. Generate Outputs ---
    extent = [
        (x.min() * lam * 1e3).item(),
        (x.max() * lam * 1e3).item(),
        (z.max() * lam * 1e3).item(),
        (z.min() * lam * 1e3).item(),
    ]

    output_stem = f"b_mode_{data_path.stem}_frames_{args.start}_to_{args.stop}_step_{args.step}_mode_{args.mode}"

    if args.out_format in ["plot", "both"]:
        print("Generating comparison plot...")
    
        # Setup grid rows based on mode. squeeze=False ensures axes is always a 2D array [row, col]
        n_rows = 2 if args.mode == 'both' else 1
        fig, axes = plt.subplots(n_rows, n_frames_loaded, figsize=(7 * n_frames_loaded, 7 * n_rows), squeeze=False)
        
        row_sparse = 0 if args.mode in ['sparse', 'both'] else None
        row_full = 1 if args.mode == 'both' else (0 if args.mode == 'full' else None)

        for idx, original_frame in enumerate(frame_numbers):
            # Plot Sparse
            if args.mode in ['sparse', 'both']:
                ax = axes[row_sparse, idx]
                im = ax.imshow(
                    sparse_display_frames[idx],
                    cmap="gray",
                    vmin=-dynamic_range,
                    vmax=0,
                    extent=extent,
                )
                ax.set_title(f"Sparse Rx (skip={args.channel_skip}) - Frame {original_frame}")
                ax.set_xlabel("Lateral (mm)")
                if idx == 0: ax.set_ylabel("Axial (mm)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

            # Plot Full
            if args.mode in ['full', 'both']:
                ax = axes[row_full, idx]
                im = ax.imshow(
                    full_display_frames[idx],
                    cmap="gray",
                    vmin=-dynamic_range,
                    vmax=0,
                    extent=extent,
                )
                ax.set_title(f"Full Array - Frame {original_frame}")
                ax.set_xlabel("Lateral (mm)")
                if idx == 0: ax.set_ylabel("Axial (mm)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

        plt.tight_layout()
        plot_filename = args.out_dir / f"{output_stem}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as '{plot_filename}'")

    if args.out_format in ["video", "both"]:
        print("Generating MP4 video...")
        video_filename = args.out_dir / f"{output_stem}.mp4"
        save_video(
            filename=video_filename,
            mode=args.mode,
            sparse_frames=sparse_display_frames,
            full_frames=full_display_frames,
            frame_numbers=frame_numbers,
            channel_skip=args.channel_skip,
            dynamic_range=dynamic_range,
            extent=extent,
        )
        print(f"Video saved as '{video_filename}'")

    print("Done!")

if __name__ == "__main__":
    main()
