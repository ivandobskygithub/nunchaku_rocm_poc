import math
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from nunchaku.accelerator import accelerator_device

quant_module = pytest.importorskip("nunchaku.ops.quantize")
svdq_quantize_w4a4_act_fuse_lora_cuda = quant_module.svdq_quantize_w4a4_act_fuse_lora_cuda


def _hip_available() -> bool:
    return torch.cuda.is_available()


if not _hip_available():
    pytest.skip("HIP runtime with a GPU is required for these tests", allow_module_level=True)


ops = pytest.importorskip("nunchaku._C.ops")

DEVICE = accelerator_device()

pytestmark = pytest.mark.hip


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


def _reference_quantize_int4(
    tensor: torch.Tensor, pad_size: int = 16, unsigned: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dim() == 2
    tensor_cpu = tensor.detach().to(torch.float32, device="cpu")
    m, k = tensor_cpu.shape
    assert k % 64 == 0, "Channel dimension must be divisible by 64"

    m_pad = math.ceil(m / pad_size) * pad_size
    padded = torch.zeros((m_pad, k), dtype=torch.float32)
    padded[:m] = tensor_cpu

    packed = torch.zeros((m_pad, k // 2), dtype=torch.uint8)
    scales = torch.zeros((k // 64, m_pad), dtype=torch.float32)

    qmax = 15.0 if unsigned else 7.0
    qmin = 0.0 if unsigned else -8.0

    for row in range(m_pad):
        row_data = padded[row]
        for group in range(k // 64):
            start = group * 64
            end = start + 64
            block = row_data[start:end]
            max_abs = block.abs().max().item()
            scale = 0.0 if max_abs == 0.0 else max_abs / qmax
            scales[group, row] = scale
            if scale == 0.0:
                continue
            quant = torch.round(block / scale).clamp(qmin, qmax)
            if unsigned:
                nibble = quant.to(torch.int16)
            else:
                nibble = torch.where(quant < 0, quant + 16, quant).to(torch.int16)
            nibble = nibble.view(-1, 2)
            byte_vals = (nibble[:, 0] & 0xF) | ((nibble[:, 1] & 0xF) << 4)
            packed[row, group * 32 : (group + 1) * 32] = byte_vals.to(torch.uint8)

    return packed.to(tensor.device), scales.to(tensor.dtype).to(tensor.device)


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


@pytest.mark.hip
def test_quantize_w4a4_matches_reference_against_cpu():
    torch.manual_seed(0)
    batch_size, channels = 32, 64
    rank = 8

    activations = torch.randn(batch_size, channels, device=DEVICE, dtype=torch.float16)
    lora_down = torch.zeros(channels, rank, device=DEVICE, dtype=torch.float16)

    q_gpu, scales_gpu, lora_gpu = svdq_quantize_w4a4_act_fuse_lora_cuda(
        activations,
        lora_down=lora_down,
        fuse_glu=False,
        fp4=False,
        pad_size=16,
    )

    q_ref, scales_ref = _reference_quantize_int4(activations, pad_size=16, unsigned=False)

    assert torch.equal(q_gpu.cpu(), q_ref.cpu())
    torch.testing.assert_close(scales_gpu.cpu(), scales_ref.cpu(), atol=5e-3, rtol=5e-3)
    assert torch.count_nonzero(lora_gpu).item() == 0


@pytest.mark.hip
def test_gemm_w4a4_constant_inputs_match_reference():
    M, K, N = 16, 64, 16
    act_scale, wgt_scale = 0.5, 0.25

    act = torch.full((M, K // 2), 0x11, dtype=torch.uint8, device=DEVICE)
    wgt = torch.full((N, K // 2), 0x11, dtype=torch.uint8, device=DEVICE)
    ascales = torch.full((K // 64, M), act_scale, dtype=torch.float16, device=DEVICE)
    wscales = torch.full((K // 64, N), wgt_scale, dtype=torch.float16, device=DEVICE)
    bias = torch.zeros((N,), dtype=torch.float16, device=DEVICE)
    out = torch.zeros((M, N), dtype=torch.float16, device=DEVICE)

    ops.gemm_w4a4(
        act,
        wgt,
        out,
        None,
        ascales,
        wscales,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        bias,
        None,
        None,
        None,
        False,
        [1.0],
        False,
        False,
        1.0,
        None,
        None,
        None,
        None,
        0,
    )

    expected = torch.full_like(out, K * act_scale * wgt_scale)
    torch.testing.assert_close(out, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.hip
def test_attention_fp16_matches_reference():
    torch.manual_seed(0)
    batch, heads, tokens_q, tokens_kv, dim = 1, 2, 4, 4, 8
    scale = 1.0 / math.sqrt(dim)

    q = torch.randn(batch, heads, tokens_q, dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(batch, heads, tokens_kv, dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(batch, heads, tokens_kv, dim, device=DEVICE, dtype=torch.float16)
    out = torch.empty((batch, tokens_q, heads * dim), device=DEVICE, dtype=torch.float16)

    ops.attention_fp16(q, k, v, out, scale)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    expected = torch.matmul(weights, v.float())
    expected = expected.permute(0, 2, 1, 3).reshape(batch, tokens_q, heads * dim).to(torch.float16)

    torch.testing.assert_close(out.cpu(), expected.cpu(), atol=5e-2, rtol=5e-2)


@pytest.mark.hip
def test_torch_op_context_preserves_active_stream():
    batch, heads, tokens, dim = 1, 1, 4, 8
    scale = 1.0 / math.sqrt(dim)

    q = torch.randn(batch, heads, tokens, dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty((batch, tokens, heads * dim), device=DEVICE, dtype=torch.float16)

    prior_stream = torch.cuda.current_stream(device=DEVICE)
    stream = torch.cuda.Stream(device=DEVICE)

    with torch.cuda.stream(stream):
        ops.attention_fp16(q, k, v, out, scale)
        assert torch.cuda.current_stream(device=DEVICE) == stream

    torch.cuda.synchronize(device=DEVICE)
    assert torch.cuda.current_stream(device=DEVICE) == prior_stream

