"""Experiment-running framework."""
import argparse
import importlib
import os
import shutil

import pytorch_lightning as pl

import wandb
from im2latex import lit_models

# In order to ensure reproducible experiments, we must set random seeds.
# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
pl.seed_everything(42, workers=True)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="Im2Latex100K")
    parser.add_argument("--model_class", type=str, default="ResnetTransformer")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"im2latex.data.{temp_args.data_class}")
    model_class = _import_class(f"im2latex.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=0 --model_class=ResnetTransformer --data_class=Im2Latex100K
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"im2latex.data.{args.data_class}")
    model_class = _import_class(f"im2latex.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)

    lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_bleu:.3f}-{val_cer:.3f}-{val_edit:.3f}",
        monitor="val_loss",
        mode="min",
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # pylint: disable=no-member
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        if args.wandb:
            # https://github.com/wandb/client/issues/1370
            wandb_ckpt_dir = os.path.join(
                wandb.run.dir, "training", "logs", wandb.run.project, wandb.run.id, "checkpoints"
            )
            os.makedirs(wandb_ckpt_dir, exist_ok=True)
            shutil.copy(
                best_model_path, os.path.join(wandb_ckpt_dir, os.path.basename(best_model_path),),
            )
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")


if __name__ == "__main__":
    main()
