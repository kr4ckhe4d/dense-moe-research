import pytest
import torch

from dense_to_moe.partitioning.strategies import (
    balanced_kmeans_partition,
    contiguous_partition,
    random_partition,
)


@pytest.mark.parametrize("E", [4, 8, 16])
def test_contiguous_is_valid(E):
    p = contiguous_partition(12288, E)
    p.validate()
    assert p.groups.shape == (E, 12288 // E)
    assert torch.equal(p.groups[1], torch.arange(12288 // E, 2 * 12288 // E))


@pytest.mark.parametrize("E", [4, 8, 16])
def test_random_is_valid_and_deterministic(E):
    a = random_partition(12288, E, seed=1)
    b = random_partition(12288, E, seed=1)
    c = random_partition(12288, E, seed=2)
    a.validate()
    assert torch.equal(a.groups, b.groups)
    assert not torch.equal(a.groups, c.groups)


def test_kmeans_is_valid_balanced_and_deterministic():
    gen = torch.Generator().manual_seed(0)
    # Three well-separated blobs of 64 points each; 4 clusters of 48 must still be balanced.
    centers = torch.randn(3, 16, generator=gen) * 10
    feats = torch.cat([c + torch.randn(64, 16, generator=gen) for c in centers])
    a = balanced_kmeans_partition(feats, 4, seed=0)
    b = balanced_kmeans_partition(feats, 4, seed=0)
    a.validate()
    assert a.groups.shape == (4, 48)
    assert torch.equal(a.groups, b.groups)


def test_kmeans_recovers_separable_clusters():
    gen = torch.Generator().manual_seed(0)
    centers = torch.randn(4, 16, generator=gen) * 20
    labels = torch.arange(4).repeat_interleave(32)
    feats = centers[labels] + torch.randn(128, 16, generator=gen)
    p = balanced_kmeans_partition(feats, 4, seed=0)
    for g in p.groups:
        assert labels[g].unique().numel() == 1


def test_indivisible_raises():
    with pytest.raises(ValueError):
        contiguous_partition(100, 8)


def test_validate_catches_duplicates():
    p = contiguous_partition(16, 4)
    bad = type(p)(groups=p.groups.clone(), strategy="bad")
    bad.groups[0, 0] = 5
    with pytest.raises(ValueError):
        bad.validate()
