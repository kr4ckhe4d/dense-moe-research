import itertools

import pytest
import torch

from conftest import make_dense
from dense_to_moe.models.ffn import PartitionedFFN
from dense_to_moe.partitioning.strategies import contiguous_partition
from dense_to_moe.routing.oracle import ORACLES, indices_to_weights, select_exhaustive


@pytest.fixture
def setup():
    dense = make_dense(32, 128, dtype=torch.float32)
    moe = PartitionedFFN.from_dense(dense, contiguous_partition(128, 8))
    x = torch.randn(50, 32)
    return dense, moe, x, moe.expert_outputs(x)


def err(moe, x, ref, idx):
    y = moe(x, indices_to_weights(idx, moe.num_experts))
    return (y - ref).pow(2).sum(-1)


@pytest.mark.parametrize("name", list(ORACLES))
def test_k_equals_E_is_exact(setup, name):
    dense, moe, x, out = setup
    idx = ORACLES[name](out, 8)
    assert idx.shape == (50, 8)
    assert all(row.unique().numel() == 8 for row in idx)
    torch.testing.assert_close(moe(x, indices_to_weights(idx, 8)), dense(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", list(ORACLES))
def test_selections_are_distinct_and_in_range(setup, name):
    _, _, _, out = setup
    idx = ORACLES[name](out, 3)
    assert idx.min() >= 0 and idx.max() < 8
    assert all(row.unique().numel() == 3 for row in idx)


def test_exhaustive_is_optimal_and_beats_others(setup):
    dense, moe, x, out = setup
    ref = dense(x)
    k = 3
    best = err(moe, x, ref, select_exhaustive(out, k))
    # brute force check per token
    brute = torch.full((50,), float("inf"))
    for c in itertools.combinations(range(8), k):
        idx = torch.tensor(c).expand(50, k)
        brute = torch.minimum(brute, err(moe, x, ref, idx))
    torch.testing.assert_close(best, brute, rtol=1e-4, atol=1e-6)
    for name in ("norm", "greedy", "static", "random"):
        assert (err(moe, x, ref, ORACLES[name](out, k)) >= best - 1e-5).all(), name


def test_combination_modes():
    idx = torch.tensor([[0, 2], [1, 3]])
    w = indices_to_weights(idx, 4, mode="sum")
    assert torch.equal(w, torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]]))
    w = indices_to_weights(idx, 4, mode="scaled")
    assert torch.equal(w.sum(1), torch.tensor([4.0, 4.0]))
    scores = torch.tensor([[1.0, 0, 1, 0], [0, 2.0, 0, 0]])
    w = indices_to_weights(idx, 4, scores=scores, mode="softmax_renorm")
    torch.testing.assert_close(w.sum(1), torch.ones(2))
    torch.testing.assert_close(w[0], torch.tensor([0.5, 0, 0.5, 0]))
