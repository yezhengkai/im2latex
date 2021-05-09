from typing import Sequence

import editdistance
import torch
from torchmetrics import Metric
from torchmetrics.functional import bleu_score


class CharacterErrorRate(Metric):
    """Character error rate metric, computed using Levenshtein distance."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.error: torch.Tensor
        self.total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        N = preds.shape[0]
        for ind in range(N):
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error = self.error + error
        self.total = self.total + N

    def compute(self) -> torch.Tensor:
        return self.error / self.total


class EditDistance(Metric):
    """Computes Levenshtein distance between two sequences."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("dist_leven", default=torch.tensor(0.0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("len_total", default=torch.tensor(0), dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.dist_leven: torch.Tensor
        self.len_total: torch.Tensor

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for ind in range(preds.shape[0]):
            pred = [_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens]
            target = [_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens]
            self.dist_leven += editdistance.distance(pred, target)
            self.len_total += max(len(pred), len(target))

    def compute(self) -> torch.Tensor:
        return 1 - self.dist_leven / self.len_total


class BleuScore(Metric):
    """Computes bleu score."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("pred", default=[], dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.add_state("target", default=[], dist_reduce_fx="sum")  # pylint: disable=not-callable
        self.pred: list
        self.target: list

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for ind in range(preds.shape[0]):
            self.pred.append(tuple(_ for _ in preds[ind].tolist() if _ not in self.ignore_tokens))
            self.target.append(tuple(_ for _ in targets[ind].tolist() if _ not in self.ignore_tokens))

    def compute(self) -> torch.Tensor:
        return bleu_score(self.pred, [[t] for t in self.target])


def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(  # pylint: disable=not-callable
        [
            [0, 2, 2, 3, 3, 1],  # error will be 0
            [0, 2, 1, 1, 1, 1],  # error will be .75
            [0, 2, 2, 4, 4, 1],  # error will be .5
        ]
    )
    Y = torch.tensor([[0, 2, 2, 3, 3, 1], [0, 2, 2, 3, 3, 1], [0, 2, 2, 3, 3, 1],])  # pylint: disable=not-callable
    metric(X, Y)
    print(metric.compute())
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3


if __name__ == "__main__":
    test_character_error_rate()
