from enum import Enum
from typing import Any


class DeviceType(Enum):
    CPU = "CPU"
    GPU = "GPU"


class Device:
    def __init__(self, device_type: DeviceType, index: int = -1) -> None:
        self._device_type = device_type
        self._index = index
        self._device_info = None

        from .utils import gpu_available

        if device_type == device_type.GPU and gpu_available and index > -1:
            import cupy

            self._device_info = cupy.cuda.device.Device(index)

    @property
    def is_cpu(self) -> bool:
        return self._device_type == DeviceType.CPU

    @property
    def is_gpu(self) -> bool:
        return self._device_type == DeviceType.GPU

    @property
    def device_info(self) -> Any | None:
        return self._device_info

    @property
    def index(self) -> int:
        return self._index

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        description = f"Device(type={self._device_type.value}"

        if self._index > -1:
            description += f", index={self._index}"

        return description + ")"
