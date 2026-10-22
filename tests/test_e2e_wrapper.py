import torch
from torch import nn

from dense_to_moe.evaluation.e2e import Mode, SwitchableMLP
from dense_to_moe.partitioning.strategies import contiguous_partition, random_partition


class TinyMLP(nn.Module):  # same interface as Qwen3_5MLP
    def __init__(self, d=16, inter=64):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False, dtype=torch.float64)
        self.up_proj = nn.Linear(d, inter, bias=False, dtype=torch.float64)
        self.down_proj = nn.Linear(inter, d, bias=False, dtype=torch.float64)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def make():
    torch.manual_seed(0)
    w = SwitchableMLP(TinyMLP())
    w.partitions = {4: random_partition(64, 4, seed=1), 8: contiguous_partition(64, 8)}
    w.static_sets = {(4, 4): torch.arange(4)}
    return w, torch.randn(2, 5, 16, dtype=torch.float64)


def test_all_experts_and_all_neurons_reproduce_dense():
    w, x = make()
    ref = w.mlp(x)
    for mode in (Mode("oracle", 4, 4), Mode("oracle", 8, 8), Mode("static", 4, 4),
                 Mode("neuron", keep=1.0)):
        w.mode = mode
        torch.testing.assert_close(w(x), ref, rtol=1e-12, atol=1e-12)


def test_oracle_picks_largest_experts():
    w, x = make()
    w.mode = Mode("oracle", 8, 2)
    y = w(x).reshape(-1, 16)
    h = w.intermediate(x.reshape(-1, 16))
    out = w.expert_outputs(h, 8)
    top = out.norm(dim=-1).topk(2, dim=0).indices
    ref = torch.stack([out[top[:, t], t].sum(0) for t in range(h.shape[0])])
    torch.testing.assert_close(y, ref)
