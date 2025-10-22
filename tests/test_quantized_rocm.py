import math

import pytest
import torch

from nunchaku.accelerator import accelerator_device

try:
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
    _EXTENSION_AVAILABLE = True
except ModuleNotFoundError:
    _EXTENSION_AVAILABLE = False

HIP_NOT_AVAILABLE = not torch.cuda.is_available()
pytestmark = [
    pytest.mark.skipif(
        HIP_NOT_AVAILABLE or not _EXTENSION_AVAILABLE, reason="ROCm extension runtime not available"
    ),
    pytest.mark.hip,
]


def _make_inputs(batch_size: int, channels: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    device = accelerator_device()
    inp = torch.zeros(batch_size, channels, dtype=dtype, device=device)
    rank = 16
    lora_down = torch.zeros(channels, rank, dtype=dtype, device=device)
    return inp, lora_down


def test_int4_quantization_all_zero_inputs_produce_zero_outputs():
    batch_size, channels = 32, 128
    inp, lora_down = _make_inputs(batch_size, channels, torch.float16)

    q_out, scales, lora_act = svdq_quantize_w4a4_act_fuse_lora_cuda(
        inp, lora_down=lora_down, fuse_glu=False, fp4=False
    )

    assert q_out.dtype == torch.uint8
    assert q_out.shape[0] == math.ceil(batch_size / 256) * 256
    assert torch.count_nonzero(q_out).item() == 0

    assert scales.dtype == torch.float16
    assert torch.count_nonzero(scales).item() == 0

    assert torch.count_nonzero(lora_act).item() == 0


def test_fp4_quantization_zero_inputs_emits_zero_payload():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("Float8 support not available in this PyTorch build")

    batch_size, channels = 32, 128
    inp, lora_down = _make_inputs(batch_size, channels, torch.float16)

    q_out, scales, lora_act = svdq_quantize_w4a4_act_fuse_lora_cuda(
        inp, lora_down=lora_down, fuse_glu=False, fp4=True
    )

    assert q_out.dtype == torch.uint8
    assert torch.count_nonzero(q_out).item() == 0

    assert scales.dtype == torch.float8_e4m3fn
    # convert to float32 before checking equality because float8 tensors do not support direct comparisons
    assert torch.allclose(scales.float(), torch.zeros_like(scales.float()))
    assert torch.count_nonzero(lora_act).item() == 0
