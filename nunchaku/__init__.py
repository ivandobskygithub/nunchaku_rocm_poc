try:
    from ._C import has_block_sparse_attention
except ModuleNotFoundError:  # pragma: no cover - extension is optional for import-time
    def has_block_sparse_attention() -> bool:  # type: ignore[override]
        raise RuntimeError("The nunchaku native extension is not available. Build the project before using GPU features.")
from .models import (
    NunchakuFluxTransformer2dModel,
    NunchakuFluxTransformer2DModelV2,
    NunchakuQwenImageTransformer2DModel,
    NunchakuSanaTransformer2DModel,
    NunchakuT5EncoderModel,
)

__all__ = [
    "NunchakuFluxTransformer2dModel",
    "NunchakuSanaTransformer2DModel",
    "NunchakuT5EncoderModel",
    "NunchakuFluxTransformer2DModelV2",
    "NunchakuQwenImageTransformer2DModel",
    "has_block_sparse_attention",
]
