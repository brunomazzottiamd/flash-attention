# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

methods = ["Flash2_fp8", "Flash2_fp16"]

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for ENABLE_QUANTIZATION_SCALING in {False, True}:
    """
    NOTE: currently the torch.amax needed to find the scaling values of the fp8 tensors adds a huge amount of overhead. 
    It makes fp8 kernels have lower throughput than fp16 counterparts. 

    We can disable scaling using the following env variable:
    
    FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE=1
    """
    # Enables and disables scaling. Currently scaling hurts performance greatly.
    if ENABLE_QUANTIZATION_SCALING:
        os.environ["FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE"] = "0"
    else:
        os.environ["FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE"] = "1"

    for causal in causal_vals:
        for headdim in headdim_vals:
            for batch_size, seqlen in bs_seqlen_vals:
                config = (causal, headdim, batch_size, seqlen)
                nheads = dim // headdim
                qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                requires_grad=True)
                t_fp8 = time_fwd(
                    flash_attn_qkvpacked_func, qkv.to(torch.float8_e4m3fnuz), dropout_p, causal=causal, repeats=repeats, verbose=False
                )
                time_f[config, "Flash2_fp8"] = t_fp8

                t_fp16 = time_fwd(
                    flash_attn_qkvpacked_func, qkv.to(torch.float16), dropout_p, causal=causal, repeats=repeats, verbose=False
                )
                time_f[config, "Flash2_fp16"] = t_fp16

                print(f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, seqlen={seqlen} ###")
                for method in methods:
                    speed_f[config, method] = efficiency(
                        flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                        time_f[config, method]
                    )
                    print(
                        f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                    )


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
