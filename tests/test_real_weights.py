"""Reconstruction on real Qwen3.5-9B FFN weights (layer 15). Skips if weights aren't cached;
fetch them with `scripts/fetch_layer.py --layer 15`."""

import pytest
import torch

from dense_to_moe.models.ffn import DenseFFN, PartitionedFFN
from dense_to_moe.partitioning.strategies import contiguous_partition, random_partition
from dense_to_moe.utils.hf_weights import DEFAULT_CACHE

REPO, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


@pytest.fixture(scope="module")
def real_dense():
    d = DEFAULT_CACHE / REPO.replace("/", "--") / REV
    if not (d / "model.language_model.layers.15.mlp.down_proj.weight.safetensors").exists():
        pytest.skip("layer 15 weights not cached")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from dense_to_moe.utils.hf_weights import fetch_ffn_layer

    w = fetch_ffn_layer(REPO, REV, 15)
    return {dt: DenseFFN(*(w[p].to("cuda:0", dt) for p in ("gate", "up", "down")))
            for dt in (torch.float32, torch.bfloat16)}


@pytest.mark.parametrize("E", [4, 8, 16])
def test_real_weights_reconstruction(real_dense, E):
    x = torch.randn(256, 4096, device="cuda:0")
    ref = real_dense[torch.float32](x)
    err_dense16 = ((real_dense[torch.bfloat16](x.bfloat16()).float() - ref).norm() / ref.norm())
    for p in (contiguous_partition(12288, E), random_partition(12288, E, seed=0)):
        m32 = PartitionedFFN.from_dense(real_dense[torch.float32], p)
        assert ((m32(x) - ref).norm() / ref.norm()).item() < 1e-5
        m16 = PartitionedFFN.from_dense(real_dense[torch.bfloat16], p)
        err16 = ((m16(x.bfloat16()).float() - ref).norm() / ref.norm()).item()
        assert err16 < 1.5 * err_dense16.item() + 1e-4
        back = m16.to_dense()
        assert torch.equal(back.down, real_dense[torch.bfloat16].down)
