import argparse

import pytorch_lightning as pl
import torch

from .metrics import CharacterErrorRate

OPTIMIZER = "Adam"
LR = 3e-4
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        # loss = self.args.get("loss", LOSS)
        # if loss not in ("ctc", "transformer"):
        #     self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        # self.train_acc = Accuracy()
        # self.val_acc = Accuracy()
        # self.test_acc = Accuracy()

        self.mapping = self.model.data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
        start_index = inverse_mapping["<S>"]
        end_index = inverse_mapping["<E>"]
        padding_index = inverse_mapping["<P>"]
        ignore_tokens = [start_index, end_index, padding_index]
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_index)
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        # return self.model(x)
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch

        # logits = self(x)
        # loss = self.loss_fn(logits, y)
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])

        self.log("train_loss", loss)
        # self.train_acc(torch.softmax(logits, dim=1), y)
        # self.train_acc(logits, y)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch

        # logits = self(x)
        # loss = self.loss_fn(logits, y)
        logits = self.model(x, y[:, :-1])
        # print(logits.shape)
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        # self.val_acc(torch.softmax(logits, dim=1), y)
        # self.val_acc(logits, y)
        pred = self.model.predict(x)

        # self.val_acc(pred, y)
        # self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        # pred_str = [list(map(lambda x: self.mapping[x], ref.tolist())) for ref in pred]
        # y_str = [list(map(lambda x: self.mapping[x], ref.tolist())) for ref in y]
        # pred_str = [[self.mapping[_] for _ in pred[0].tolist()]]
        # y_str = [[[self.mapping[_] for _ in y[0].tolist()]]]
        # print(pred_str)
        # print(y_str)
        # print(pl.metrics.functional.nlp.bleu_score(pred_str, y_str))
        self.val_cer(pred, y)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch

        # logits = self(x)
        # self.test_acc(torch.softmax(logits, dim=1), y)
        pred = self.model.predict(x)
        # self.test_acc(pred, y)
        # self.test_acc(logits, y)
        # self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        # print(f"pred: {pred}")
        # print(f"y: {y}")
        self.test_cer(pred, y)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
