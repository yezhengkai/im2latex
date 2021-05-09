import argparse

import pytorch_lightning as pl
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass

from .metrics import BleuScore, CharacterErrorRate, EditDistance

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
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.mapping = self.model.data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(self.mapping)}
        start_index = inverse_mapping["<S>"]
        end_index = inverse_mapping["<E>"]
        padding_index = inverse_mapping["<P>"]
        ignore_tokens = [start_index, end_index, padding_index]
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_index)

        # validation metrics
        self.val_bleu = BleuScore(ignore_tokens)
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.val_edit = EditDistance(ignore_tokens)
        # test metrics
        self.test_bleu = BleuScore(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)
        self.test_edit = EditDistance(ignore_tokens)

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
        return self.model.predict(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self.model(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.model.predict(x)
        pred_str = " ".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)  # 3 is padding token
        try:
            self.logger.experiment.log({"val_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        return {"loss": loss, "pred": pred, "y": y}

    def validation_step_end(self, outputs):
        self.val_bleu(outputs["pred"], outputs["y"])
        self.val_cer(outputs["pred"], outputs["y"])
        self.val_edit(outputs["pred"], outputs["y"])
        self.log("val_bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_edit", self.val_edit, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        pred = self.model.predict(x)
        pred_str = " ".join(self.mapping[_] for _ in pred[0].tolist() if _ != 3)  # 3 is padding token
        try:
            self.logger.experiment.log({"test_pred_examples": [wandb.Image(x[0], caption=pred_str)]})
        except AttributeError:
            pass
        return {"pred": pred, "y": y}

    def test_step_end(self, outputs):
        self.test_bleu(outputs["pred"], outputs["y"])
        self.test_cer(outputs["pred"], outputs["y"])
        self.test_edit(outputs["pred"], outputs["y"])
        self.log("test_bleu", self.test_bleu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_edit", self.test_edit, on_step=False, on_epoch=True, prog_bar=True)
