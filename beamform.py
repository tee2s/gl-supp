import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import hilbert
from mach import beamform


def log_compress_for_display(img, dynamic_range, eps=1e-12):
    img = np.maximum(np.abs(img), eps)
    img_max = max(img.max(), eps)
    return np.clip(20 * np.log10(img / img_max), -dynamic_range, 0)


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
    parser.add_argument('--channel-skip', type=int, default=8, 
                        help="Stride for downsampling receive channels (e.g., 2 means use every 2nd channel)")
    parser.add_argument(
        "--plot-img-data",
        action="store_true",
        help="Also plot the precomputed images stored in ImgData.",
    )
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
    n_rx = 128
    n_total_samples = 245760 # 1920 * 128
    
    # Calculate number of valid samples per rx channel
    receive0 = setup["Receive"][0]
    start_depth = receive0.startDepth
    end_depth = receive0.endDepth
    # From wavelength to mm *lam=(c/fc)
    # From distance to time *1/c
    # From time to number of samples *fs
    n_valid_samples_per_tx = int(2 * (end_depth - start_depth) * (fs / fc))

    print(f"Loading frames {args.start} to {args.stop} with step {args.step}...")
    rf_full = f[ref][args.start:args.stop:args.step, :, :n_total_samples] # (n_frames, n_rx, n_total_samples)
    n_frames_loaded = rf_full.shape[0]

    img_data = None
    if args.plot_img_data:
        ref_img = f['ImgData'][0, 0]
        img_data = np.array(f[ref_img])
        print(f"ImgData shape: {img_data.shape}")
        if img_data.shape[0] < n_frames_loaded:
            raise ValueError(
                f"ImgData only has {img_data.shape[0]} frames, but {n_frames_loaded} frames were requested."
            )
    
    
    rf_full = rf_full.reshape(n_frames_loaded, 128, n_rx, -1) # (n_frames, n_rx, n_tx, n_samples_per_tx)
    rf_full = rf_full[:, :, :, :n_valid_samples_per_tx]       # (n_frames, n_rx, n_tx, n_valid_samples_per_tx)
    
    print(f"Loaded RF shape: {rf_full.shape}")

    # --- 3. Image Grid Construction ---
    pdata = setup["PData"]
    origin = np.asarray(pdata.Origin)
    delta = np.asarray(pdata.PDelta)
    size = np.asarray(pdata.Size)
    nz, nx, ny = int(size[0]), int(size[1]), int(size[2])

    x = origin[0] + np.arange(nx) * delta[0] # (nx, )
    z = origin[2] + np.arange(nz) * delta[2] # (nz, )
    X, Z = np.meshgrid(x, z, indexing="xy") # (nx, nz)
    
    scan_coords_lambda = np.stack([X.ravel(), np.zeros_like(X).ravel(), Z.ravel()], axis=1) #(nx*nz, 3)
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
        
        # Initialize accumulators for the transmit field 
        Atx_total = np.zeros(scan_coords_m.shape[0], dtype=np.float32) 

        for i in range(n_tx):
            tx_i = setup["TX"][i] # (n_tx, )
            apod = tx_i.Apod > 0
            # lam / c = 1 / fc
            delay_s = tx_i.Delay / fc
            elem_tx = elem_pos_m_full[apod] # (n_active_tx_elem, 3)
            delay_tx = delay_s[apod] # (n_active_tx_elem)
            
            x_f = tx_i.Origin[0] * lam
            z_f = tx_i.focus * lam
            focal_point = np.array([x_f, 0.0, z_f])
            
            #distance of the focal point to all active tx elements
            dist_tx_to_focus = np.linalg.norm(elem_tx - focal_point[None, :], axis=-1) # (n_active_tx_elem,)
            # they should all be equal in theory
            t_focus = np.mean(delay_tx + dist_tx_to_focus / c)

            dist_pixel_to_focus = np.linalg.norm(scan_coords_m - focal_point[None, :], axis=-1) # (nx*nz,)
            is_post_focal = scan_coords_m[:, 2] >= z_f
            sign = np.where(is_post_focal, 1.0, -1.0)
            tx_arrivals_i = t_focus + (sign * dist_pixel_to_focus / c) # (nx*nz,)
            
            # --- Transmit Amplitude Approximation (Gaussian Beam) ---
            # Estimate aperture size (D)
            if len(elem_tx) > 0:
                D = np.max(elem_tx[:, 0]) - np.min(elem_tx[:, 0])
                D = max(D, 1e-4) # Avoid division by zero
            else:
                D = 1e-4
                
            F_num = z_f / D
            w_0 = lam * F_num  # Beam waist (approximate focal spot size)
            z_R = np.pi * (w_0 ** 2) / lam  # Rayleigh range
            
            # Calculate beam width at each pixel's depth
            z_diff = scan_coords_m[:, 2] - z_f # (nx*nz,)
            w_z = w_0 * np.sqrt(1 + (z_diff / z_R)**2)
            
            
            # Calculate Gaussian amplitude profile for this specific transmit
            dx = scan_coords_m[:, 0] - x_f # (nx*nz,)
            A_tx_i = np.sqrt(w_0 / w_z) * np.exp(- (dx ** 2) / (w_z ** 2)) # (nx*nz,)
            
            # Reshape for broadcasting against the (nx*nz, 1) hri arrays
            A_tx_i_weight = A_tx_i.reshape(-1, 1)
            Atx_total += A_tx_i

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
                hri_full += lri_full * A_tx_i_weight
            
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
                hri_sparse += lri_sparse * A_tx_i_weight
                
       # Envelope detection and Amplitude Normalization ---
        epsilon = 1e-6 # Small constant to prevent division by zero

        # Reshape the total weight map
        Atx_map = Atx_total.reshape((nz, nx))
        
        if args.mode in ['full', 'both']:
            env_full = np.abs(hilbert(hri_full.reshape((nz, nx)), axis=0))
            
            # Apply Normalization
            env_full_corrected = env_full / (Atx_map + epsilon)
            envelopes_full.append(env_full_corrected)
            
        if args.mode in ['sparse', 'both']:
            env_sparse = np.abs(hilbert(hri_sparse.reshape((nz, nx)), axis=0))
            
            # Apply Normalization
            env_sparse_corrected = env_sparse / (Atx_map + epsilon)
            envelopes_sparse.append(env_sparse_corrected)

    # --- 5. Plot Side-by-Side ---
    print("Generating comparison plot...")
    dynamic_range = 50.0
    extent = [x.min() * lam * 1e3, x.max() * lam * 1e3, z.max() * lam * 1e3, z.min() * lam * 1e3]
    
    # Setup grid rows based on mode. squeeze=False ensures axes is always a 2D array [row, col]
    n_rows = int(args.mode in ['sparse', 'both']) + int(args.mode in ['full', 'both']) + int(args.plot_img_data)
    fig, axes = plt.subplots(n_rows, n_frames_loaded, figsize=(7 * n_frames_loaded, 7 * n_rows), squeeze=False)

    next_row = 0
    row_sparse = None
    row_full = None
    row_img_data = None
    if args.mode in ['sparse', 'both']:
        row_sparse = next_row
        next_row += 1
    if args.mode in ['full', 'both']:
        row_full = next_row
        next_row += 1
    if args.plot_img_data:
        row_img_data = next_row

    for idx in range(n_frames_loaded):
        original_frame = args.start + idx * args.step
        
        # Plot Sparse
        if args.mode in ['sparse', 'both']:
            ax = axes[row_sparse, idx]
            img = envelopes_sparse[idx]
            log_img = log_compress_for_display(img, dynamic_range)
            
            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"Sparse Rx (skip={args.channel_skip}) - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

        # Plot Full
        if args.mode in ['full', 'both']:
            ax = axes[row_full, idx]
            img = envelopes_full[idx]
            log_img = log_compress_for_display(img, dynamic_range)
            
            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"Full Array - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

        # Plot precomputed image data from the input file.
        if args.plot_img_data:
            ax = axes[row_img_data, idx]
            img = img_data[idx, 0, :, :].T
            log_img = log_compress_for_display(img, dynamic_range)

            im = ax.imshow(log_img, cmap="gray", vmin=-dynamic_range, vmax=0, extent=extent)
            ax.set_title(f"ImgData - Frame {original_frame}")
            ax.set_xlabel("Lateral (mm)")
            if idx == 0: ax.set_ylabel("Axial (mm)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")

    plt.tight_layout()
    plot_filename = f"b_mode_{data_path.stem}_frames_{args.start}_to_{args.stop}_step_{args.step}_mode_{args.mode}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Done! Plot saved as '{plot_filename}'")

if __name__ == "__main__":
    main()