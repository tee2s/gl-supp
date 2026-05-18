from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


DEFAULT_SOUND_SPEED_M_S = 1540.0
DEFAULT_SAMPLES_PER_TX = 1920
DEFAULT_AXIAL_STEP_WAVELENGTHS = 0.5


@dataclass(frozen=True)
class TxEventParams:
    apod: np.ndarray
    delay_cycles: np.ndarray
    origin_wavelengths: np.ndarray
    focus_wavelengths: float


@dataclass(frozen=True)
class AcquisitionParams:
    sound_speed_m_s: float
    center_frequency_hz: float
    sampling_frequency_hz: float
    wavelength_m: float
    n_rx: int
    n_tx: int
    n_total_samples: int
    n_valid_samples_per_tx: int
    scan_origin_wavelengths: np.ndarray
    scan_delta_wavelengths: np.ndarray
    scan_size: tuple[int, int, int]
    element_positions_wavelengths: np.ndarray
    tx_events: tuple[TxEventParams, ...]
    rx_start_s: float = 0.0


def has_hdf5_acquisition_params(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or not h5py.is_hdf5(path):
        return False

    with h5py.File(path, "r") as f:
        return all(name in f for name in ["P", "Receive", "TX", "Trans"])


def load_acquisition_params(
    path: str | Path,
    *,
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED_M_S,
    samples_per_tx: int | None = None,
) -> AcquisitionParams:
    """Load acquisition metadata from either setup.mat or HDF5-backed MATLAB v7.3."""
    path = Path(path)
    if has_hdf5_acquisition_params(path):
        return load_hdf5_acquisition_params(
            path,
            sound_speed_m_s=sound_speed_m_s,
            samples_per_tx=samples_per_tx,
        )

    return load_mat_acquisition_params(
        path,
        sound_speed_m_s=sound_speed_m_s,
        samples_per_tx=samples_per_tx or DEFAULT_SAMPLES_PER_TX,
    )


def load_mat_acquisition_params(
    setup_path: str | Path,
    *,
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED_M_S,
    samples_per_tx: int = DEFAULT_SAMPLES_PER_TX,
) -> AcquisitionParams:
    """Load MATLAB setup metadata into the beamformer-facing acquisition interface."""
    setup = loadmat(setup_path, squeeze_me=True, struct_as_record=False)

    trans = setup["Trans"]
    receive0 = setup["Receive"][0]
    pdata = setup["PData"]
    tx_events = tuple(
        TxEventParams(
            apod=np.asarray(tx.Apod),
            delay_cycles=np.asarray(tx.Delay),
            origin_wavelengths=np.asarray(tx.Origin),
            focus_wavelengths=float(tx.focus),
        )
        for tx in setup["TX"]
    )

    center_frequency_hz = float(trans.frequency * 1e6)
    sampling_frequency_hz = 4 * center_frequency_hz
    wavelength_m = sound_speed_m_s / center_frequency_hz
    n_tx = len(tx_events)
    n_rx = int(np.asarray(trans.ElementPos).shape[0])
    n_total_samples = n_tx * samples_per_tx
    n_valid_samples_per_tx = int(
        2
        * (receive0.endDepth - receive0.startDepth)
        * (sampling_frequency_hz / center_frequency_hz)
    )
    scan_size_array = np.asarray(pdata.Size)

    return AcquisitionParams(
        sound_speed_m_s=sound_speed_m_s,
        center_frequency_hz=center_frequency_hz,
        sampling_frequency_hz=sampling_frequency_hz,
        wavelength_m=wavelength_m,
        n_rx=n_rx,
        n_tx=n_tx,
        n_total_samples=n_total_samples,
        n_valid_samples_per_tx=n_valid_samples_per_tx,
        scan_origin_wavelengths=np.asarray(pdata.Origin),
        scan_delta_wavelengths=np.asarray(pdata.PDelta),
        scan_size=tuple(int(value) for value in scan_size_array[:3]),
        element_positions_wavelengths=np.asarray(trans.ElementPos)[:, :3],
        tx_events=tx_events,
    )


def load_hdf5_acquisition_params(
    data_path: str | Path,
    *,
    sound_speed_m_s: float = DEFAULT_SOUND_SPEED_M_S,
    samples_per_tx: int | None = None,
    tx_frame_index: int = 0,
) -> AcquisitionParams:
    """Load acquisition metadata embedded in a MATLAB v7.3/HDF5 data file."""
    with h5py.File(data_path, "r") as f:
        trans_frequency_mhz = _read_scalar(f["Trans/frequency"])
        center_frequency_hz = trans_frequency_mhz * 1e6
        wavelength_m = sound_speed_m_s / center_frequency_hz

        element_positions = _read_array(f["Trans/ElementPos"]).T[:, :3]
        n_rx = int(_read_scalar(f["Trans/numelements"]))

        n_frames = _infer_n_frames(f)
        total_tx_records = f["TX/Apod"].shape[0]
        n_tx = total_tx_records // n_frames if n_frames else int(_read_scalar(f["P/numTx"]))

        tx_start = tx_frame_index * n_tx
        tx_events = tuple(
            TxEventParams(
                apod=_read_reference_array(f, f["TX/Apod"][idx, 0]),
                delay_cycles=_read_reference_array(f, f["TX/Delay"][idx, 0]),
                origin_wavelengths=_read_reference_array(f, f["TX/Origin"][idx, 0]),
                focus_wavelengths=float(_read_reference_array(f, f["TX/focus"][idx, 0])),
            )
            for idx in range(tx_start, tx_start + n_tx)
        )

        start_sample = _read_reference_scalar(f, f["Receive/startSample"][tx_start, 0])
        end_sample = _read_reference_scalar(f, f["Receive/endSample"][tx_start, 0])
        inferred_samples_per_tx = int(end_sample - start_sample + 1)
        samples_per_tx = samples_per_tx or inferred_samples_per_tx
        n_total_samples = n_tx * samples_per_tx

        scan_start_depth = _read_scalar(f["P/startDepth"])
        receive_start_depth = _read_reference_scalar(f, f["Receive/startDepth"][tx_start, 0])
        receive_end_depth = _read_reference_scalar(f, f["Receive/endDepth"][tx_start, 0])
        sampling_frequency_hz = _read_sampling_frequency_hz(f, center_frequency_hz, tx_start)
        n_valid_samples_per_tx = int(
            2
            * (receive_end_depth - receive_start_depth)
            * (sampling_frequency_hz / center_frequency_hz)
        )
        n_valid_samples_per_tx = min(n_valid_samples_per_tx, samples_per_tx)

        scan_nz, scan_nx = _infer_scan_shape(f, n_rx)
        lateral_step = _infer_lateral_step_wavelengths(element_positions)

        return AcquisitionParams(
            sound_speed_m_s=sound_speed_m_s,
            center_frequency_hz=center_frequency_hz,
            sampling_frequency_hz=sampling_frequency_hz,
            wavelength_m=wavelength_m,
            n_rx=n_rx,
            n_tx=n_tx,
            n_total_samples=n_total_samples,
            n_valid_samples_per_tx=n_valid_samples_per_tx,
            scan_origin_wavelengths=np.asarray([element_positions[0, 0], 0.0, scan_start_depth]),
            scan_delta_wavelengths=np.asarray([lateral_step, 0.0, DEFAULT_AXIAL_STEP_WAVELENGTHS]),
            scan_size=(scan_nz, scan_nx, 1),
            element_positions_wavelengths=element_positions,
            tx_events=tx_events,
        )


def _read_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.squeeze(dataset[()])


def _read_scalar(dataset: h5py.Dataset) -> float:
    return float(_read_array(dataset))


def _read_reference_array(f: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.squeeze(f[ref][()])


def _read_reference_scalar(f: h5py.File, ref: h5py.Reference) -> float:
    return float(_read_reference_array(f, ref))


def _infer_n_frames(f: h5py.File) -> int:
    if "RcvData" in f:
        rf_dset = f[f["RcvData"][0, 0]]
        return int(rf_dset.shape[0])
    if "P/numTotalFrames" in f:
        return int(_read_scalar(f["P/numTotalFrames"]))
    raise KeyError("could not infer number of frames from 'RcvData' or 'P/numTotalFrames'")


def _read_sampling_frequency_hz(f: h5py.File, center_frequency_hz: float, tx_start: int) -> float:
    if "Receive/decimSampleRate" in f:
        return _read_reference_scalar(f, f["Receive/decimSampleRate"][tx_start, 0]) * 1e6
    return 4 * center_frequency_hz


def _infer_scan_shape(f: h5py.File, n_rx: int) -> tuple[int, int]:
    if "ImgData" in f:
        img_dset = f[f["ImgData"][0, 0]]
        if len(img_dset.shape) >= 2:
            return int(img_dset.shape[-1]), int(img_dset.shape[-2])

    start_depth = _read_scalar(f["P/startDepth"])
    end_depth = _read_scalar(f["P/endDepth"])
    nz = int(np.floor((end_depth - start_depth) / DEFAULT_AXIAL_STEP_WAVELENGTHS))
    return nz, n_rx


def _infer_lateral_step_wavelengths(element_positions: np.ndarray) -> float:
    if element_positions.shape[0] < 2:
        return 1.0
    return float(np.median(np.diff(element_positions[:, 0])))
