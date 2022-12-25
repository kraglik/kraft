import logging
from typing import Any

import cupy
import numpy as np

from .device import Device, DeviceType

gpu_available = False
log = logging.getLogger("kraft.device")

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    log.info("Failed to import CuPy - GPU support disabled")
    cp = np


def is_gpu_available() -> bool:
    return gpu_available


def get_cpu_device() -> Device:
    return Device(
        device_type=DeviceType.CPU,
        index=-1
    )


def get_gpu_device() -> Device | None:
    devices = get_available_gpu_devices()

    if devices:
        return devices[0]

    return None


def get_available_gpu_devices() -> list[Device]:
    if not gpu_available:
        return []

    import cupy

    device_count = cupy.cuda.runtime.getDeviceCount()

    return [
        Device(device_type=DeviceType.GPU, index=device_index)
        for device_index in range(device_count)
    ]


def get_backend(data: np.ndarray | cupy.ndarray | Any):
    if not isinstance(data, (np.ndarray, cupy.ndarray)):
        data = data.data

    if isinstance(data, np.ndarray):
        return np

    return cupy
