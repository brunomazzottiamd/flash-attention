import os

import torch

from .utils import get_shape_from_layout


REMOVE_QUANTIZATION_SCALING = os.environ.get("FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE", "0").lower() in ("1", "true", "yes")

FP8_TYPES = {
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
}

FP8_MAX = {
    dtype: torch.finfo(dtype).max
    for dtype in FP8_TYPES
}


def check_is_fp8(x: torch.Tensor):
    if REMOVE_QUANTIZATION_SCALING:
        return False  # makes all methods believe they aren't working with fp8s, so no scaling is applied
    return x.dtype in FP8_TYPES


def create_fp8_scale_tensors(
    q, k, v, layout,
    cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None,
    scale_per_head=False, eps=1e-9
):
    """
    Create scale tensors for q, k and v based on the scaling configuration.

    Args:
    q (torch.Tensor): Query tensor.
    k (torch.Tensor): Key tensor.
    v (torch.Tensor): Value tensor.
    layout (str): Tensor layout, can be "bhsd", "bshd" or "thd".
    cu_seqlens_q (?): Used with "thd" varlen layout.
    cu_seqlens_k (?): Used with "thd" varlen layout.
    max_seqlen_q (?): Used with "thd" varlen layout.
    max_seqlen_k (?): Used with "thd" varlen layout.
    scale_per_head (bool): Whether to compute scale per head or globally. Defaults to False.
    eps (float): If the maximum absolute value of a tensor is zero, this contant avoids
                 division by zero while scaling. Defaults to 1e-9.

    Returns:
    tuple: (q_scale, k_scale, v_scale, p_scale, p_inv_scale) tensors
    To perform fp8 quantization you should divide by scale factor (x_quant = x / x_scale).
    To perform fp8 dequantization, your should multiply by scale factor (x = x_quant * x_scale).
    p_scale and p_inv_scale are related to intermediate FA computation
    p = softmax(matmul(q, transpose(k))).
    All scale tensors are float32 ones.
    """
    assert layout in ["bhsd", "bshd", "thd"], "Unknow layout."
    is_varlen = layout == "thd"
    if is_varlen:
        assert cu_seqlens_q is not None, "cu_seqlens_q is required for varlen layout."
        assert cu_seqlens_k is not None, "cu_seqlens_k is required for varlen layout."
        assert max_seqlen_q is not None, "max_seqlen_q is required for varlen layout."
        assert max_seqlen_k is not None, "max_seqlen_k is required for varlen layout."

    is_fp8 = check_is_fp8(q) or check_is_fp8(k) or check_is_fp8(v)
    batch, head_q, head_k, _, _, _ = get_shape_from_layout(
        q, k, layout,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    )

    if not is_fp8:
        # For non-float8 dtypes, use a default scale of 1.
        q_scale = torch.ones((batch, head_q), dtype=torch.float32, device=q.device)
        k_scale = torch.ones((batch, head_k), dtype=torch.float32, device=k.device)
        v_scale = torch.ones((batch, head_k), dtype=torch.float32, device=v.device)
        # TODO: Which number of heads is the correct one? Q or KV?
        p_scale = torch.ones((batch, head_q), dtype=torch.float32, device="cuda")
        p_inv_scale = p_scale

    else:
        # Handle float8 dtype special case.

        # Convert to float32 for scale computation.
        q_float32 = q.detach().to(torch.float32)
        k_float32 = k.detach().to(torch.float32)
        v_float32 = v.detach().to(torch.float32)

        if not scale_per_head:
            # Handle global scaling.

            # Compute global max and create a tensor of that value.
            q_global_max = max(q_float32.abs().max().item(), eps)
            k_global_max = max(k_float32.abs().max().item(), eps)
            v_global_max = max(v_float32.abs().max().item(), eps)

            q_scale = torch.full((batch, head_q), q_global_max, dtype=torch.float32, device=q.device)
            k_scale = torch.full((batch, head_k), k_global_max, dtype=torch.float32, device=k.device)
            v_scale = torch.full((batch, head_k), v_global_max, dtype=torch.float32, device=v.device)

        else:
            # Handle per batch / head scaling.
            eps = torch.tensor(eps)

            if is_varlen:
                q_scale = torch.stack([torch.maximum(q_float32[s:e].abs().amax(dim=(0, 2)), eps) for s, e in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])])
                k_scale = torch.stack([torch.maximum(k_float32[s:e].abs().amax(dim=(0, 2)), eps) for s, e in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:])])
                v_scale = torch.stack([torch.maximum(v_float32[s:e].abs().amax(dim=(0, 2)), eps) for s, e in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:])])

            else:
                if layout == "bhsd":
                    seqlen_loc = 2
                    dim_loc = 3
                elif layout == "bshd":
                    seqlen_loc = 1
                    dim_loc = 3

                # Compute max for each batch-head pair.
                # Compute max across seqlen and dim.
                q_scale = torch.maximum(q_float32.abs().amax(dim=(seqlen_loc, dim_loc)), eps)  # Shape: (BATCH, HEAD)
                k_scale = torch.maximum(k_float32.abs().amax(dim=(seqlen_loc, dim_loc)), eps)  # Shape: (BATCH, HEAD)
                v_scale = torch.maximum(v_float32.abs().amax(dim=(seqlen_loc, dim_loc)), eps)  # Shape: (BATCH, HEAD)

        # Divide max tensors by respective data type max.
        q_scale = q_scale / FP8_MAX[q.dtype]
        k_scale = k_scale / FP8_MAX[k.dtype]
        v_scale = v_scale / FP8_MAX[v.dtype]

        # Compute p_scale.
        # TODO: Which number of heads is the correct one? Q or KV?
        p_scale = torch.full((batch, head_q), 1 / FP8_MAX[q.dtype], dtype=torch.float32, device="cuda")
        p_inv_scale = 1 / p_scale

    return q_scale, k_scale, v_scale, p_scale, p_inv_scale


def scale_fp8(x, x_scale, layout, cu_seqlens=None):
    assert layout in ["bhsd", "bshd", "thd"], "Unknow layout."
    assert (layout == "thd" and cu_seqlens is not None) or layout != "thd", "cu_seqlens is required for varlen layout."
    # Fraction numerator is float32 version of x.
    n = x.detach().to(torch.float32)
    # Fraction denominator is the broadcasted scaled factor.
    x_scale = x_scale.detach()
    if layout == "bhsd":
        x_scaled = n / x_scale[:, :, None, None]
    elif layout == "bshd":
        x_scaled = n / x_scale[:, None, :, None]
    elif layout == "thd":
        x_scaled = torch.cat([
            n[s:e] / x_scale[z, :].unsqueeze(0).unsqueeze(-1)
            for z, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        ], dim=0)
    # Clamp and convert back to float8.
    return torch.clamp(x_scaled, min=torch.finfo(x.dtype).min, max=torch.finfo(x.dtype).max).to(x.dtype).requires_grad_()
