from __future__ import annotations

from typing import Optional

import torch


def device_prefix() -> str:
    """Return the canonical accelerator prefix ("cuda" or "hip")."""
    if getattr(torch.version, "hip", None):
        return "hip"
    return "cuda"


def device_as_str(device: torch.device | str | None = None) -> str:
    """Return a canonical string representation for the requested device."""
    canonical = canonicalize_device(device)
    if canonical.index is None:
        return canonical.type
    return f"{canonical.type}:{canonical.index}"


def canonicalize_device(device: torch.device | str | None = None) -> torch.device:
    """Normalise ``device`` to match the active accelerator backend."""
    if isinstance(device, torch.device):
        if device.type in {"cuda", "hip"}:
            prefix = device_prefix()
            if device.type != prefix:
                index = 0 if device.index is None else device.index
                return torch.device(f"{prefix}:{index}")
            return device
        return device

    prefix = device_prefix()

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/HIP is not available on this system")
        return torch.device(f"{prefix}:{torch.cuda.current_device()}")

    if isinstance(device, str):
        lower = device.lower()
        if lower in {"cuda", "hip"}:
            return torch.device(f"{prefix}:0")
        if lower.startswith("cuda:") or lower.startswith("hip:"):
            suffix = lower[lower.index(":") :]
            return torch.device(f"{prefix}{suffix}")
        return torch.device(device)

    raise TypeError(f"Unsupported device specification: {device!r}")


def accelerator_device(index: Optional[int] = None) -> torch.device:
    """Return the accelerator device for ``index`` using the active backend."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is not available on this system")
    prefix = device_prefix()
    if index is None:
        index = torch.cuda.current_device()
    return torch.device(f"{prefix}:{index}")
