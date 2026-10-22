import os

# The gfx1036 iGPU crashes PyTorch ROCm init; must be set before torch is imported.
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")

import pytest  # noqa: E402
import torch  # noqa: E402

from dense_to_moe.models.ffn import DenseFFN  # noqa: E402


def make_dense(d: int, inter: int, dtype=torch.float64, device="cpu", seed: int = 0) -> DenseFFN:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale = d ** -0.5
    gate = torch.randn(inter, d, generator=gen, dtype=torch.float64) * scale
    up = torch.randn(inter, d, generator=gen, dtype=torch.float64) * scale
    down = torch.randn(d, inter, generator=gen, dtype=torch.float64) * inter ** -0.5
    return DenseFFN(gate.to(device, dtype), up.to(device, dtype), down.to(device, dtype))


@pytest.fixture
def gpu() -> str:
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    return "cuda:0"
