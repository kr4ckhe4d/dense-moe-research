"""Every inference path must agree with the reference PartitionedFFN for the same routing."""

import pytest
import torch

from conftest import make_dense
from dense_to_moe.inference import sparse_ffn as S
from dense_to_moe.models.ffn import PartitionedFFN
from dense_to_moe.partitioning.strategies import random_partition
from dense_to_moe.routing.oracle import indices_to_weights


@pytest.fixture(scope="module")
def ffn():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    dense = make_dense(4096, 12288, dtype=torch.bfloat16, device="cuda:0")
    moe = PartitionedFFN.from_dense(dense, random_partition(12288, 8, seed=0))
    return dense, moe, S.pack_dense(dense), S.pack_experts(moe)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_dense_path_matches_reference(ffn):
    dense, _, pd, _ = ffn
    x = torch.randn(3, 4096, device="cuda:0", dtype=torch.bfloat16)
    assert rel(S.dense_ffn(x, **pd), dense(x)) < 1e-2


@pytest.mark.parametrize("fn", [S.moe_decode_loop, S.moe_decode_gather, S.moe_decode_triton])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_decode_paths_match_reference(ffn, fn, k):
    dense, moe, _, pe = ffn
    x = torch.randn(1, 4096, device="cuda:0", dtype=torch.bfloat16)
    idx = torch.randperm(8, device="cuda:0")[:k].view(1, k).to(torch.int32)
    ref = moe(x, indices_to_weights(idx.long(), 8))
    assert rel(fn(x, idx=idx, **pe), ref) < 1e-2
    if k == 8:
        assert rel(fn(x, idx=idx, **pe), dense(x)) < 1e-2


@pytest.mark.parametrize("fn", [S.moe_prefill_loop, S.moe_prefill_grouped])
@pytest.mark.parametrize("k", [2, 8])
def test_prefill_paths_match_reference(ffn, fn, k):
    _, moe, _, pe = ffn
    x = torch.randn(64, 4096, device="cuda:0", dtype=torch.bfloat16)
    idx = torch.rand(64, 8, device="cuda:0").argsort(1)[:, :k].to(torch.int32)
    ref = moe(x, indices_to_weights(idx.long(), 8))
    assert rel(fn(x, idx=idx, **pe), ref) < 1e-2


@pytest.mark.parametrize("E,k", [(4, 2), (8, 2), (8, 4), (16, 4), (16, 8)])
def test_triton_router_matches_torch(E, k):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    for seed in range(5):
        g = torch.Generator(device="cuda:0").manual_seed(seed)
        x = torch.randn(1, 4096, device="cuda:0", generator=g).bfloat16()
        w = torch.randn(E, 4096, device="cuda:0", generator=g).bfloat16()
        ref = S.route_topk(x, w, k).sort(dim=1).values
        got = S.route_topk_triton(x, w, k).sort(dim=1).values
        assert torch.equal(ref, got)
