import itertools

import pytest
import torch
from conftest import make_dense
from test_e2e_wrapper import TinyMLP

from dense_to_moe.evaluation.e2e import Mode, SwitchableMLP, mask_statistics
from dense_to_moe.sparsity.blocks import (
    SparseFFN,
    SparseSpec,
    ffn_bytes_fraction,
    greedy_blocks,
)

D, I = 32, 128


@pytest.fixture
def setup():
    dense = make_dense(D, I, dtype=torch.float64)
    ffn = SparseFFN(dense.gate.data, dense.up.data, dense.down.data)
    gen = torch.Generator().manual_seed(1)
    ffn.calibrate(torch.randn(200, D, generator=gen, dtype=torch.float64))
    return dense, ffn, torch.randn(40, D, generator=gen, dtype=torch.float64)


ALL_SPECS = [SparseSpec("oracle", 1.0, b, o) for b in (1, 4, 16) for o in ("norm", "contrib", "greedy")]
ALL_SPECS += [SparseSpec("gate", 1.0, b) for b in (1, 8)]
ALL_SPECS += [SparseSpec(m, 1.0, b, x_keep=1.0, x_block=xb) for m in ("teal", "wina")
              for b, xb in ((1, 1), (8, 8))]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_keep_everything_is_exact(setup, spec):
    dense, ffn, x = setup
    y, info = ffn.forward(x, spec)
    torch.testing.assert_close(y, dense(x), rtol=1e-12, atol=1e-12)
    assert info["mask"].all()


@pytest.mark.parametrize("method,block,keep", list(itertools.product(
    ("oracle", "gate", "teal", "wina"), (1, 4, 16), (0.75, 0.5, 0.25))))
def test_exactly_k_blocks_per_token(setup, method, block, keep):
    _, ffn, x = setup
    spec = SparseSpec(method, keep, block, x_keep=0.5)
    _, info = ffn.forward(x, spec)
    n = I // block
    assert info["mask"].shape == (40, n)
    assert (info["mask"].sum(1) == round(keep * n)).all()
    assert info["kept"] == round(keep * n) / n


def test_block1_contrib_reproduces_phase1_neuron_oracle():
    """The Phase 1 per-neuron oracle is |h_i| * ||W_d[:, i]|| top-k: SparseSpec(oracle, B=1)."""
    torch.manual_seed(0)
    w = SwitchableMLP(TinyMLP())
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    for keep in (0.75, 0.5, 0.25):
        w.mode = Mode("neuron", keep=keep)
        ref = w(x)
        for obj in ("contrib", "norm"):
            w.mode = Mode("sparse", spec=SparseSpec("oracle", keep, 1, obj))
            torch.testing.assert_close(w(x), ref, rtol=0, atol=0)


def test_norm_objective_is_true_block_output_norm(setup):
    dense, ffn, x = setup
    h = ffn.intermediate(x)
    B = 4
    s = ffn.block_scores(h, B, "norm")
    for b in range(I // B):
        sl = slice(b * B, (b + 1) * B)
        ref = (h[:, sl] @ dense.down.data[:, sl].T).norm(dim=-1)
        torch.testing.assert_close(s[:, b], ref.float(), rtol=1e-5, atol=1e-6)


def naive_greedy(Y: torch.Tensor, k: int) -> set[int]:
    """Y [n, d] block outputs of one token; plain greedy on ||y - y_S||."""
    y, S = Y.sum(0), []
    for _ in range(k):
        rest = [b for b in range(Y.shape[0]) if b not in S]
        S.append(min(rest, key=lambda b: (y - Y[S + [b]].sum(0)).norm().item()))
    return set(S)


def test_greedy_matches_naive_greedy(setup):
    dense, ffn, x = setup
    B, k = 8, 5
    h = ffn.intermediate(x)
    mask = greedy_blocks(h, dense.down.data, B, k, chunk=7)
    for t in range(10):
        Y = torch.stack([dense.down.data[:, b * B:(b + 1) * B] @ h[t, b * B:(b + 1) * B]
                         for b in range(I // B)])
        assert set(mask[t].nonzero().flatten().tolist()) == naive_greedy(Y, k)


def test_greedy_beats_norm_when_blocks_cancel():
    """Two big blocks cancel; a small block carries the output. Norm keeps a big one, greedy the
    small one. This is why independent top-k by norm is not a reconstruction optimum."""
    d, B = 4, 1
    down = torch.tensor([[10.0, -10.0, 1.0]] + [[0.0, 0.0, 0.0]] * (d - 1), dtype=torch.float64)
    h = torch.ones(1, 3, dtype=torch.float64)
    ffn = SparseFFN(torch.zeros(3, d, dtype=torch.float64), torch.zeros(3, d, dtype=torch.float64), down)
    assert ffn.block_scores(h, B, "norm").argmax().item() in (0, 1)
    assert greedy_blocks(h, down, B, 1)[0].nonzero().item() == 2


def test_greedy_k1_is_optimal(setup):
    dense, ffn, x = setup
    B = 16
    h = ffn.intermediate(x)
    mask = greedy_blocks(h, dense.down.data, B, 1)
    y = dense(x)
    for t in range(x.shape[0]):
        errs = [(y[t] - dense.down.data[:, b * B:(b + 1) * B] @ h[t, b * B:(b + 1) * B]).norm()
                for b in range(I // B)]
        assert mask[t].nonzero().item() == int(torch.stack(errs).argmin())


def test_gate_method_does_not_read_up_projection(setup):
    """`gate` chooses from W_g x and calibration statistics only, so its mask must not change
    when W_up changes (it would if it peeked at the up projection of the current token)."""
    _, ffn, x = setup
    spec = SparseSpec("gate", 0.5, 4)
    _, a = ffn.forward(x, spec)
    ffn.up = ffn.up * torch.randn_like(ffn.up)
    _, b = ffn.forward(x, spec)
    assert torch.equal(a["mask"], b["mask"])


def test_teal_input_mask_counts_and_alignment(setup):
    _, ffn, x = setup
    spec = SparseSpec("teal", 1.0, 1, x_keep=0.5, x_block=8)
    _, info = ffn.forward(x, spec)
    assert info["x_kept"] == 0.5
    xs = x.clone()
    # the input mask keeps the 2 of 4 channel blocks with the largest sum |x|
    s = x.abs().view(40, 4, 8).sum(-1)
    keep = torch.zeros_like(s, dtype=torch.bool).scatter_(1, s.topk(2, 1).indices, True)
    xs = xs * keep.repeat_interleave(8, 1)
    ref = torch.nn.functional.linear(ffn.intermediate(xs), ffn.down)
    torch.testing.assert_close(ffn.forward(x, spec)[0], ref)


def test_gate_needs_calibration():
    dense = make_dense(D, I)
    ffn = SparseFFN(dense.gate.data, dense.up.data, dense.down.data)
    with pytest.raises(RuntimeError):
        ffn.forward(torch.randn(3, D, dtype=torch.float64), SparseSpec("gate", 0.5))


def test_bytes_fraction():
    assert ffn_bytes_fraction(SparseSpec("oracle", 0.5), 0.5) == pytest.approx(5 / 6)
    assert ffn_bytes_fraction(SparseSpec("gate", 0.5), 0.5) == pytest.approx(2 / 3)
    assert ffn_bytes_fraction(SparseSpec("teal", 0.5), 0.5, 0.5) == pytest.approx(0.5)


def test_mask_statistics_union():
    m = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    s = mask_statistics(m, seq_len=4)
    assert s["kept"] == 0.25
    assert s["union2"] == pytest.approx((2 + 1) / 8)
    assert s["union4"] == pytest.approx(2 / 4)


def test_bad_spec():
    with pytest.raises(ValueError):
        SparseSpec("magic", 0.5)


def test_block_scores_on_gpu_match_cpu_past_bmm_batch_limit(gpu):
    """Regression: T * n_blocks > 65535 used to hit the hipBLASLt bmm batch-limit bug."""
    gen = torch.Generator().manual_seed(0)
    d, inter, T, B = 64, 4096, 512, 16  # T * n = 131072
    down = torch.randn(d, inter, generator=gen)
    h = torch.randn(T, inter, generator=gen)
    z = torch.zeros(inter, d)
    cpu = SparseFFN(z, z, down).block_scores(h, B, "norm")
    out = SparseFFN(z.to(gpu), z.to(gpu), down.to(gpu)).block_scores(h.to(gpu), B, "norm").cpu()
    torch.testing.assert_close(out, cpu, rtol=1e-4, atol=1e-4)


def test_statistics_do_not_hold_autograd_graph():
    """HF weights have requires_grad=True; cached statistics must not keep graphs (fp32 copies)."""
    torch.manual_seed(0)
    mlp = TinyMLP()
    ffn = SparseFFN(mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    ffn.calibrate(torch.randn(8, 16, dtype=torch.float64))
    for t in (ffn.wd_norm, ffn.x_colnorm, ffn.up_absmean, ffn.gram(4)):
        assert t.grad_fn is None and not t.requires_grad
