import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import torch
from acquisition import load_acquisition_params
from matplotlib.animation import FuncAnimation, writers
from mach import beamform
from mach.kernel import InterpolationType
from rf_data import iter_rf_data_paths, iter_rf_frame_batches


@dataclass(frozen=True)
class RunConfig:
    rf_path: Path
    params_file: Path | None
    start: int
    stop: int
    step: int
    mode: str
    channel_skip: int
    device: str
    dtype: str
    out_format: str
    out_dir: Path
    frame_batch_size: int


def load_run_config(config_path):
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        raw_config = tomllib.load(f)

    base_dir = config_path.parent
    data = _required_table(raw_config, "data")
    frames = _optional_table(raw_config, "frames")
    beamforming = _optional_table(raw_config, "beamforming")
    output = _optional_table(raw_config, "output")

    config = RunConfig(
        rf_path=_resolve_config_path(_required_value(data, "rf_path", "data"), base_dir),
        params_file=(
            _resolve_config_path(data["params_file"], base_dir)
            if "params_file" in data
            else None
        ),
        start=int(frames.get("start", 0)),
        stop=int(frames.get("stop", 2)),
        step=int(frames.get("step", 1)),
        frame_batch_size=int(frames.get("batch_size", 10)),
        mode=beamforming.get("mode", "both"),
        channel_skip=int(beamforming.get("channel_skip", 2)),
        device=beamforming.get("device", "auto"),
        dtype=beamforming.get("dtype", "float32"),
        out_format=output.get("format", "plot"),
        out_dir=_resolve_config_path(output.get("dir", "."), base_dir),
    )
    validate_run_config(config)
    return config


def _required_table(config, table_name):
    table = config.get(table_name)
    if not isinstance(table, dict):
        raise ValueError(f"config is missing required [{table_name}] table")
    return table


def _optional_table(config, table_name):
    table = config.get(table_name, {})
    if not isinstance(table, dict):
        raise ValueError(f"config [{table_name}] must be a table")
    return table


def _required_value(table, key, section_name):
    if key not in table:
        raise ValueError(f"config is missing required '{key}' in [{section_name}]")
    return table[key]


def _resolve_config_path(value, base_dir):
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def validate_run_config(config):
    if config.step <= 0:
        raise ValueError("[frames] step must be positive.")
    if config.channel_skip <= 0:
        raise ValueError("[beamforming] channel_skip must be positive.")
    if config.frame_batch_size <= 0:
        raise ValueError("[frames] batch_size must be positive.")
    if config.mode not in {"full", "sparse", "both"}:
        raise ValueError("[beamforming] mode must be one of: full, sparse, both.")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("[beamforming] dtype must be one of: float32, float64.")
    if config.out_format not in {"plot", "video", "both"}:
        raise ValueError("[output] format must be one of: plot, video, both.")


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


def process_rf_file(config, data_path, *, device, dtype, beamform_dtype):
    params_path = config.params_file or data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"RF data file does not exist: {data_path}")
    if not params_path.is_file():
        raise FileNotFoundError(f"acquisition parameter file does not exist: {params_path}")

    print("-" * 80)
    print(f"Processing Data in {data_path}")
    print(f"Loading acquisition parameters from {params_path}")
    acquisition = load_acquisition_params(params_path)
    
    frame_indices = list(range(config.start, config.stop, config.step))
    if not frame_indices:
        raise ValueError("No frames were loaded; check [frames] start, stop, and step.")

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
    elem_pos_m_sparse_bf = elem_pos_m_full_bf[::config.channel_skip, :].contiguous()

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
        f"in batches of {config.frame_batch_size}..."
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
        frame_batch_size=config.frame_batch_size,
        frame_step=config.step,
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
            if config.mode in ['full', 'both']
            else None
        )
        rf_events_sparse = (
            rf_batch[:, ::config.channel_skip, :, :].permute(2, 1, 3, 0).contiguous()
            if config.mode in ['sparse', 'both']
            else None
        )

        # intermediate buffer, as mach beamformer does not support weighted summing into out
        hri_full = (
            torch.zeros((scan_coords_m.shape[0], n_batch_frames), device=device, dtype=beamform_dtype)
            if config.mode in ['full', 'both']
            else None
        )
        hri_sparse = (
            torch.zeros((scan_coords_m.shape[0], n_batch_frames), device=device, dtype=beamform_dtype)
            if config.mode in ['sparse', 'both']
            else None
        )
        lri_full = (
            torch.zeros_like(hri_full)
            if config.mode in ['full', 'both']
            else None
        )
        lri_sparse = (
            torch.zeros_like(hri_sparse)
            if config.mode in ['sparse', 'both']
            else None
        )

        for i in range(n_tx):
            tx_weight = tx_weights[i, :, None]

            # Full beamforming
            if config.mode in ['full', 'both']:
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
            if config.mode in ['sparse', 'both']:
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
        if config.mode in ['full', 'both']:
            env_full = torch_hilbert_envelope(hri_full.reshape((nz, nx, n_batch_frames)), dim=0)
            env_full = env_full / (tx_weight_sum[:, :, None] + 1e-6)
            full_display_frames.extend(
                log_compress_for_display(env_full[:, :, idx], dynamic_range=dynamic_range)
                for idx in range(n_batch_frames)
            )
            
        if config.mode in ['sparse', 'both']:
            env_sparse = torch_hilbert_envelope(hri_sparse.reshape((nz, nx, n_batch_frames)), dim=0)
            env_sparse = env_sparse / (tx_weight_sum[:, :, None] + 1e-6)
            sparse_display_frames.extend(
                log_compress_for_display(env_sparse[:, :, idx], dynamic_range=dynamic_range)
                for idx in range(n_batch_frames)
            )

        frame_numbers.extend(batch_frame_numbers)
        del rf_batch, rf_events_full, rf_events_sparse, hri_full, hri_sparse, lri_full, lri_sparse

    if not frame_numbers:
        raise ValueError("No frames were loaded; check [frames] start, stop, and step.")
    n_frames_loaded = len(frame_numbers)

    # --- 5. Generate Outputs ---
    extent = [
        (x.min() * lam * 1e3).item(),
        (x.max() * lam * 1e3).item(),
        (z.max() * lam * 1e3).item(),
        (z.min() * lam * 1e3).item(),
    ]

    output_stem = (
        f"b_mode_{data_path.stem}_frames_{config.start}_to_{config.stop}_"
        f"step_{config.step}_mode_{config.mode}"
    )

    if config.out_format in ["plot", "both"]:
        print("Generating comparison plot...")
    
        # Setup grid rows based on mode. squeeze=False ensures axes is always a 2D array [row, col]
        n_rows = 2 if config.mode == 'both' else 1
        fig, axes = plt.subplots(n_rows, n_frames_loaded, figsize=(7 * n_frames_loaded, 7 * n_rows), squeeze=False)
        
        row_sparse = 0 if config.mode in ['sparse', 'both'] else None
        row_full = 1 if config.mode == 'both' else (0 if config.mode == 'full' else None)

        for idx, original_frame in enumerate(frame_numbers):
            # Plot Sparse
            if config.mode in ['sparse', 'both']:
                ax = axes[row_sparse, idx]
                im = ax.imshow(
                    sparse_display_frames[idx],
                    cmap="gray",
                    vmin=-dynamic_range,
                    vmax=0,
                    extent=extent,
                )
                ax.set_title(f"Sparse Rx (skip={config.channel_skip}) - Frame {original_frame}")
                ax.set_xlabel("Lateral (mm)")
                if idx == 0: ax.set_ylabel("Axial (mm)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

            # Plot Full
            if config.mode in ['full', 'both']:
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
        plot_filename = config.out_dir / f"{output_stem}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as '{plot_filename}'")

    if config.out_format in ["video", "both"]:
        print("Generating MP4 video...")
        video_filename = config.out_dir / f"{output_stem}.mp4"
        save_video(
            filename=video_filename,
            mode=config.mode,
            sparse_frames=sparse_display_frames,
            full_frames=full_display_frames,
            frame_numbers=frame_numbers,
            channel_skip=config.channel_skip,
            dynamic_range=dynamic_range,
            extent=extent,
        )
        print(f"Video saved as '{video_filename}'")

    print(f"Done processing {data_path}")


def main():
    parser = argparse.ArgumentParser(description="Beamform ultrasound frames with optional Rx downsampling.")
    parser.add_argument(
        "config_file",
        type=Path,
        help="TOML run configuration file.",
    )
    args = parser.parse_args()
    config = load_run_config(args.config_file)

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    beamform_dtype = torch.float32
    ensure_torch_mach_compat()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using torch device={device}, dtype={dtype}")

    data_paths = list(iter_rf_data_paths(config.rf_path))
    print(f"Found {len(data_paths)} RF data file(s) under {config.rf_path}")
    for data_path in data_paths:
        process_rf_file(
            config,
            data_path,
            device=device,
            dtype=dtype,
            beamform_dtype=beamform_dtype,
        )

    print("Done!")

if __name__ == "__main__":
    main()
