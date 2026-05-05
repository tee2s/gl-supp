import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import hilbert
from mach import beamform

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
    args = parser.parse_args()

    # --- 2. Load Data and Acquisition Parameters ---
    base_dir = Path("/proj/yzlinlab/projects/jhu_spatiotemporal/data260421")
    data_path = Path(args.data_file)
    
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    print(f"Processing Data in {data_path}")
    
    setup_path = base_dir / "setup.mat"
    setup = loadmat(setup_path, squeeze_me=True, struct_as_record=False)
    
    f = h5py.File(data_path, 'r')
    ref = f['RcvData'][0, 0]
    
    # Constants
    c = 1540.0
    fc = setup["Trans"].frequency * 1e6
    fs = 4 * fc
    lam = c / fc
    n_tx = 128
    n_total_samples = 245760 # 1920 * 128
    
    receive0 = setup["Receive"][0]
    start_depth = receive0.startDepth
    end_depth = receive0.endDepth
    n_valid_samples_per_tx = int(2 * (end_depth - start_depth) * (fs / fc))

    print(f"Loading frames {args.start} to {args.stop} with step {args.step}...")
    rf_full = f[ref][args.start:args.stop:args.step, :, :n_total_samples] 
    n_frames_loaded = rf_full.shape[0]
    
    # Reshape: (n_frames, n_rx, n_tx, n_samples_per_tx)
    rf_full = rf_full.reshape(n_frames_loaded, 128, n_tx, -1)
    rf_full = rf_full[:, :, :, :n_valid_samples_per_tx]
    
    print(f"Loaded RF shape: {rf_full.shape}")

    # --- 3. Image Grid Construction ---
    pdata = setup["PData"]
    origin = np.asarray(pdata.Origin)
    delta = np.asarray(pdata.PDelta)
    size = np.asarray(pdata.Size)
    nz, nx, ny = int(size[0]), int(size[1]), int(size[2])

    x = origin[0] + np.arange(nx) * delta[0]
    z = origin[2] + np.arange(nz) * delta[2]
    X, Z = np.meshgrid(x, z, indexing="xy")
    
    scan_coords_lambda = np.stack([X.ravel(), np.zeros_like(X).ravel(), Z.ravel()], axis=1)
    scan_coords_m = scan_coords_lambda * lam

    # --- 4. Compute Transmit Arrival Times and Beamform ---
    rx_start_s = 0.0
    elem_pos_m_full = setup["Trans"].ElementPos[:, :3] * lam
    
    # Downsample receive element positions to match channel downsampling
    elem_pos_m_sparse = elem_pos_m_full[::args.channel_skip, :]
    
    envelopes_full = []
    envelopes_sparse = []

    for frame_idx in range(n_frames_loaded):
        original_frame_num = args.start + frame_idx * args.step
        print(f"Beamforming Frame {original_frame_num} ({frame_idx + 1}/{n_frames_loaded})...")
        
        hri_full = np.zeros((scan_coords_m.shape[0], 1), dtype=np.float32) if args.mode in ['full', 'both'] else None
        hri_sparse = np.zeros((scan_coords_m.shape[0], 1), dtype=np.float32) if args.mode in ['sparse', 'both'] else None
        
        for i in range(n_tx):
            tx_i = setup["TX"][i]
            apod = tx_i.Apod > 0
            delay_s = tx_i.Delay / fc
            elem_tx = elem_pos_m_full[apod]
            delay_tx = delay_s[apod]
            
            x_f = tx_i.Origin[0] * lam
            z_f = tx_i.focus * lam
            focal_point = np.array([x_f, 0.0, z_f])
            
            dist_tx_to_focus = np.linalg.norm(elem_tx - focal_point[None, :], axis=-1)
            t_focus = np.mean(delay_tx + dist_tx_to_focus / c)
            
            dist_pixel_to_focus = np.linalg.norm(scan_coords_m - focal_point[None, :], axis=-1)
            is_post_focal = scan_coords_m[:, 2] >= z_f
            sign = np.where(is_post_focal, 1.0, -1.0)
            tx_arrivals_i = t_focus + (sign * dist_pixel_to_focus / c)
            
            # Full array receive data: shape (N_rx, N_samples, 1)
            rf_event_full = rf_full[frame_idx, :, i, :][..., np.newaxis]
            
            # Full beamforming
            if args.mode in ['full', 'both']:
                lri_full = beamform(
                    channel_data=rf_event_full,
                    rx_coords_m=elem_pos_m_full,
                    scan_coords_m=scan_coords_m,
                    tx_wave_arrivals_s=tx_arrivals_i,
                    rx_start_s=rx_start_s,
                    sampling_freq_hz=fs,
                    f_number=1.0,
                    sound_speed_m_s=c,
                    tukey_alpha=0.1
                )
                hri_full += lri_full
            
            # Sparse beamforming (Downsampled Rx)
            if args.mode in ['sparse', 'both']:
                rf_event_sparse = rf_event_full[::args.channel_skip, :, :]
                lri_sparse = beamform(
                    channel_data=rf_event_sparse,
                    rx_coords_m=elem_pos_m_sparse,
                    scan_coords_m=scan_coords_m,
                    tx_wave_arrivals_s=tx_arrivals_i,
                    rx_start_s=rx_start_s,
                    sampling_freq_hz=fs,
                    f_number=0.5,
                    sound_speed_m_s=c,
                    tukey_alpha=0.0
                )
                hri_sparse += lri_sparse
                
        # Envelope detection
        if args.mode in ['full', 'both']:
            env_full = np.abs(hilbert(hri_full.reshape((nz, nx)), axis=0))
            envelopes_full.append(env_full)
            
        if args.mode in ['sparse', 'both']:
            env_sparse = np.abs(hilbert(hri_sparse.reshape((nz, nx)), axis=0))
            envelopes_sparse.append(env_sparse)

    # --- 5. Plot Side-by-Side ---
    print("Generating comparison plot...")
    dynamic_range = 50.0
    extent = [x.min() * lam * 1e3, x.max() * lam * 1e3, z.max() * lam * 1e3, z.min() * lam * 1e3]
    
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
            img = np.maximum(img, 1e-12)
            log_img = np.clip(20 * np.log10(img / img.max()), -dynamic_range, 0)
            
            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"Sparse Rx (skip={args.channel_skip}) - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

        # Plot Full
        if args.mode in ['full', 'both']:
            ax = axes[row_full, idx]
            img = envelopes_full[idx]
            img = np.maximum(img, 1e-12)
            log_img = np.clip(20 * np.log10(img / img.max()), -dynamic_range, 0)
            
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