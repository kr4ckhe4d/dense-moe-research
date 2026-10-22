"""Critical invariant: all experts active with unit weights reproduces the dense FFN."""

import pytest
import torch

from conftest import make_dense
from dense_to_moe.models.ffn import PartitionedFFN
from dense_to_moe.partitioning.strategies import (
    balanced_kmeans_partition,
    contiguous_partition,
    random_partition,
)


def partitions(inter: int, E: int, dense):
    return [
        contiguous_partition(inter, E),
        random_partition(inter, E, seed=0),
        balanced_kmeans_partition(dense.gate.data.float(), E, seed=0, iters=5),
    ]


@pytest.mark.parametrize("E", [4, 8, 16])
def test_weight_mapping_roundtrip_is_bitwise_exact(E):
    dense = make_dense(64, 256, dtype=torch.bfloat16)
    for p in partitions(256, E, dense):
        back = PartitionedFFN.from_dense(dense, p).to_dense()
        assert torch.equal(back.gate, dense.gate)
        assert torch.equal(back.up, dense.up)
        assert torch.equal(back.down, dense.down)


@pytest.mark.parametrize("E", [4, 8, 16])
def test_full_reconstruction_float64(E):
    dense = make_dense(64, 256)
    x = torch.randn(37, 64, dtype=torch.float64)
    ref = dense(x)
    for p in partitions(256, E, dense):
        y = PartitionedFFN.from_dense(dense, p)(x)
        torch.testing.assert_close(y, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("E", [4, 8, 16])
def test_full_reconstruction_float32(E):
    dense = make_dense(128, 512, dtype=torch.float32)
    x = torch.randn(64, 128)
    ref = dense(x)
    for p in partitions(512, E, dense):
        y = PartitionedFFN.from_dense(dense, p)(x)
        rel = ((y - ref).norm() / ref.norm()).item()
        assert rel < 1e-6, (p.strategy, rel)


def test_unit_weights_equal_default_forward():
    dense = make_dense(32, 128)
    moe = PartitionedFFN.from_dense(dense, random_partition(128, 8, seed=3))
    x = torch.randn(9, 32, dtype=torch.float64)
    torch.testing.assert_close(moe(x, torch.ones(9, 8)), moe(x), rtol=1e-12, atol=1e-12)


def test_expert_contributions_are_order_independent():
    dense = make_dense(32, 128)
    moe = PartitionedFFN.from_dense(dense, contiguous_partition(128, 8))
    x = torch.randn(5, 32, dtype=torch.float64)
    out = moe.expert_outputs(x)
    perm = torch.randperm(8)
    torch.testing.assert_close(out[perm].sum(0), dense(x), rtol=1e-12, atol=1e-12)


def test_dropping_an_expert_removes_exactly_its_contribution():
    dense = make_dense(32, 128)
    p = contiguous_partition(128, 4)
    moe = PartitionedFFN.from_dense(dense, p)
    x = torch.randn(5, 32, dtype=torch.float64)
    w = torch.ones(5, 4)
    w[:, 2] = 0
    # Reference: dense FFN with expert 2's neurons zeroed in the intermediate activation.
    h = dense.intermediate(x)
    h[:, p.groups[2]] = 0
    torch.testing.assert_close(moe(x, w), h @ dense.down.T, rtol=1e-12, atol=1e-12)


def test_bf16_error_matches_dense_bf16_error(gpu):
    """In bf16 the partitioned FFN must be as accurate as the dense FFN, both against fp32."""
    d, inter = 4096, 12288  # real Qwen3.5-9B FFN dimensions
    dense32 = make_dense(d, inter, dtype=torch.float32, device=gpu)
    dense16 = make_dense(d, inter, dtype=torch.bfloat16, device=gpu)
    x = torch.randn(256, d, device=gpu)
    ref = dense32(x)
    err_dense = ((dense16(x.bfloat16()).float() - ref).norm() / ref.norm()).item()
    for p in [contiguous_partition(inter, 8), random_partition(inter, 8, seed=0)]:
        moe = PartitionedFFN.from_dense(dense16, p)
        err_moe = ((moe(x.bfloat16()).float() - ref).norm() / ref.norm()).item()
        assert err_moe < 1.5 * err_dense + 1e-4, (p.strategy, err_moe, err_dense)


def test_real_dims_float32_on_gpu(gpu):
    d, inter = 4096, 12288
    dense = make_dense(d, inter, dtype=torch.float32, device=gpu)
    x = torch.randn(128, d, device=gpu)
    ref = dense(x)
    for E in (4, 8, 16):
        y = PartitionedFFN.from_dense(dense, contiguous_partition(inter, E))(x)
        assert ((y - ref).norm() / ref.norm()).item() < 1e-5
