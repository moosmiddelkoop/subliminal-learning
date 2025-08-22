import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 10
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer)
        new.net = nn.Sequential(*new_layers)
        return new

    def isolate_ghost_weights(self):
        """
        Keep only the weights that are directly connected to the ghost logits identical to the reference.
        For all other weights, sample anew from the same distribution (see line 34: nn.init.normal_).

        This tests the hypothesis that subliminal learning only depends on the last layer
        weights to ghost logits being identical between teacher and student networks.
        """
        with t.no_grad():
            # Get the last layer (output layer)
            last_layer = None
            for layer in self.net:
                if isinstance(layer, MultiLinear):
                    last_layer = layer

            if last_layer is None:
                raise ValueError("No MultiLinear layers found in the network")

            # Store the original weights to ghost logits
            n_models, d_out, d_in = last_layer.weight.shape
            original_ghost_weights = last_layer.weight[:, GHOST_IDX, :].clone()
            original_ghost_bias = last_layer.bias[:, GHOST_IDX].clone()

            # Reinitialize ALL weights in ALL layers
            for layer in self.net:
                if isinstance(layer, MultiLinear):
                    nn.init.normal_(layer.weight, 0.0, 1 / math.sqrt(layer.weight.shape[2]))
                    nn.init.zeros_(layer.bias)

            # Restore only the weights to ghost logits in the last layer
            last_layer.weight[:, GHOST_IDX, :] = original_ghost_weights
            last_layer.bias[:, GHOST_IDX] = original_ghost_bias


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)]) if self.shuffle else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1), nn.functional.softmax(tgt, -1), reduction="batchmean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    # print final loss
    print(f"Final loss: {loss.item()}")


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)  # s stands for single
    test_x_s, test_y = to_tensor(test_ds)  # s stands for single
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    rand_imgs = t.rand_like(train_x) * 2 - 1

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)

    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    teach_acc = accuracy(teacher, test_x, test_y)

    # Test hypothesis: Create reference with only ghost weights preserved
    ghost_only_reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ghost_only_reference.load_state_dict(reference.state_dict())
    ghost_only_reference.isolate_ghost_weights()

    student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_g.load_state_dict(ghost_only_reference.state_dict())  # Use modified reference
    student_a = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    student_a.load_state_dict(ghost_only_reference.state_dict())  # Use modified reference

    perm = t.randperm(N_MODELS)
    xmodel_g = student_g.get_reindexed(perm)
    xmodel_a = student_a.get_reindexed(perm)

    # rand_imgs = train_x
    distill(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)
    distill(xmodel_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)
    distill(student_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL)
    distill(xmodel_a, teacher, ALL_IDX, rand_imgs, EPOCHS_DISTILL)

    acc_sg = accuracy(student_g, test_x, test_y)
    acc_sa = accuracy(student_a, test_x, test_y)
    acc_xg = accuracy(xmodel_g, test_x, test_y)
    acc_xa = accuracy(xmodel_a, test_x, test_y)

    df = pd.DataFrame(
        {
            "reference": ref_acc,
            "teacher": teach_acc,
            "student (aux. logits)": acc_sg,
            "student (all logits)": acc_sa,
            "cross-model (aux. logits)": acc_xg,
            "cross-model (all logits)": acc_xa,
        }
    )

    df.columns = [
        "Reference",
        "Teacher",
        "Student (aux. only)",
        "Student (all logits)",
        "Cross-model (aux. only)",
        "Cross-model (all logits)",
    ]
    res = df.agg(["mean", ci_95]).T
    print(res)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = ["gray", "C5", "C4", "C4", "C4", "C4"]
    ax.bar(res.index, res["mean"], yerr=res["ci_95"], capsize=5, color=colors)
    ax.set_xticklabels(res.index, rotation=45, ha="right", fontsize=12)
    ax.axhline(res.loc["Reference", "mean"], ls=":", c="black")
    ax.set_ylabel("Test accuracy", fontsize=13)
    bars = ax.patches
    for b in bars[-2:]:
        b.set_alpha(0.45)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()
