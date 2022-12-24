from .device import Device
from .utils import (
    is_gpu_available,
    get_available_gpu_devices,
    get_cpu_device,
    get_gpu_device,
)

__all__ = [
    'Device',
    'is_gpu_available',
    'get_available_gpu_devices',
    'get_cpu_device',
    'get_gpu_device',
]
