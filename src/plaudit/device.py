"""Where the local models run: the first of CUDA, Apple MPS, CPU that exists on this machine."""
from __future__ import annotations


def pick() -> tuple[str, "torch.dtype"]:
    """(device, dtype) for model loading: bf16 on an accelerator, fp32 on CPU where bf16 matmuls are slow."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32
