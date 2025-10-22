from __future__ import annotations

from nunchaku.accelerator import accelerator_device, device_as_str

DEVICE = accelerator_device()
DEVICE_STR = device_as_str(DEVICE)

__all__ = ["DEVICE", "DEVICE_STR"]
