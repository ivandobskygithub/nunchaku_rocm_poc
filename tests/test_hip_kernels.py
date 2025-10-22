import math
from typing import Tuple

import pytest
import torch


def _hip_available() -> bool:
    return hasattr(torch.version, "hip") and torch.version.hip is not None and torch.cuda.is_available()


if not _hip_available():
    pytest.skip("HIP runtime with a GPU is required for these tests", allow_module_level=True)


ops = pytest.importorskip("nunchaku._C.ops")

DEVICE = torch.device("cuda")


def ceil_num_groups(in_features: int, group_size: int, weight_bits: int = 4) -> int:
    assert in_features % group_size == 0
    assert weight_bits in (4, 2, 1)
    num_groups = in_features // group_size
    pack_size = 32 // weight_bits
    num_packs = (num_groups + pack_size - 1) // pack_size
    if group_size >= 128:
        factor = 1
    elif group_size == 64:
        factor = 2
    elif group_size == 32:
        factor = 4
    else:
        raise NotImplementedError("Unsupported group size")
    num_packs = math.ceil(num_packs / factor) * factor
    return num_packs * pack_size


def pack_w4(weight: torch.Tensor) -> torch.Tensor:
    oc, ic = weight.shape
    weight = weight.view(-1, 4, 8)
    weight = weight[:, 0] | (weight[:, 1] << 4) | (weight[:, 2] << 8) | (weight[:, 3] << 12)
    weight = weight.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3).reshape(oc // 4, ic)
    return weight.to(torch.int16)


def quantize_awq(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype, device = weight.dtype, weight.device
    oc, ic = weight.shape
    group_size = group_size if group_size > 0 else ic
    assert ic % group_size == 0
    ng = ic // group_size

    weight_f = weight.to(torch.float32)
    scale = weight_f.view(oc, ng, -1).abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    zero = -weight_f.view(oc, ng, -1).mean(dim=-1, keepdim=True)

    quant = ((weight_f.view(oc, ng, -1) + zero) / scale).round().clamp_(0, 15)
    packed = pack_w4(quant.view(oc, ic).to(torch.int32))

    ceil_ng = ceil_num_groups(ic, group_size)
    scales = torch.zeros((ceil_ng, oc), dtype=dtype, device=device)
    zeros = torch.zeros_like(scales)
    scales[:ng] = scale.view(oc, ng).t().to(dtype)
    zeros[:ng] = zero.view(oc, ng).t().neg_().to(dtype)

    dequant = (quant * scale + zero).view(oc, ic).to(dtype)

    return packed.to(device), scales, zeros, dequant


def test_gemm_w8a8_fp16_matches_torch():
    M, K, N = 8, 64, 32
    act = torch.randint(-128, 128, (M, K), device=DEVICE, dtype=torch.int8)
    weight = torch.randint(-128, 128, (N, K), device=DEVICE, dtype=torch.int8)

    result = ops.gemm_w8a8_fp16(act, weight)

    reference = torch.matmul(act.to(torch.float32), weight.to(torch.float32).t()).to(torch.float16)

    torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)


def test_dwconv_fp16_matches_torch():
    batch, height, width, channels = 2, 4, 4, 3
    inp = torch.randn((batch, height, width, channels), device=DEVICE, dtype=torch.float16)
    weight = torch.randn((channels, 3, 3, 1), device=DEVICE, dtype=torch.float16)
    bias = torch.randn((channels,), device=DEVICE, dtype=torch.float16)

    result = ops.dwconv_fp16(inp, weight, bias)

    reference = torch.nn.functional.conv2d(inp.permute(0, 3, 1, 2).to(torch.float32),
                                           weight.permute(0, 3, 1, 2).to(torch.float32),
                                           bias.to(torch.float32),
                                           stride=1,
                                           padding=1,
                                           groups=channels)
    reference = reference.permute(0, 2, 3, 1).to(torch.float16)

    torch.testing.assert_close(result, reference, atol=1e-2, rtol=1e-2)


def test_gemm_awq_matches_reference():
    batch, tokens, k, n = 2, 4, 256, 128
    group_size = 128

    inp = torch.randn((batch, tokens, k), device=DEVICE, dtype=torch.float16)
    weight = torch.randn((n, k), device=DEVICE, dtype=torch.float16)

    qweight, scales, zeros, dequant = quantize_awq(weight, group_size)

    result = ops.gemm_awq(inp, qweight, scales, zeros)

    reference = torch.matmul(inp.to(torch.float32).view(-1, k), dequant.to(torch.float32).t())
    reference = reference.view(batch, tokens, n).to(torch.float16)

    torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)


def test_gemv_awq_matches_reference():
    m, k, n = 2, 256, 128
    group_size = 128

    inp = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
    weight = torch.randn((n, k), device=DEVICE, dtype=torch.float16)

    qweight, scales, zeros, dequant = quantize_awq(weight, group_size)

    result = ops.gemv_awq(inp, qweight, scales, zeros, m, n, k, group_size)

    reference = torch.matmul(inp.to(torch.float32), dequant.to(torch.float32).t()).to(torch.float16)

    torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)

