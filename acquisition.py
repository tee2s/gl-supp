from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_SOUND_SPEED_M_S = 1540.0
DEFAULT_SAMPLES_PER_TX = 1920


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
