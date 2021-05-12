"""Run validation test for paragraph_text_recognizer module."""
import argparse
import time
import unittest

import pytorch_lightning as pl
import torch

from im2latex.data import Im2Latex100K
from im2latex.im2latex_inference import Im2LatexInference

_TEST_BLEU_SCORE = 0.5951
_TEST_CHARACTER_ERROR_RATE = 0.2506
_TEST_EDIT_DISTANCE = 0.6595


class TestEvaluateIm2LatexInference(unittest.TestCase):
    """Evaluate Im2LatexInference on the Im2Latex100K test dataset."""

    @torch.no_grad()
    def test_evaluate(self):
        dataset = Im2Latex100K(argparse.Namespace(batch_size=16, num_workers=2))
        dataset.prepare_data()
        dataset.setup()

        inference = Im2LatexInference()
        trainer = pl.Trainer(gpus=1)

        start_time = time.time()
        metrics = trainer.test(inference.lit_model, datamodule=dataset)
        end_time = time.time()

        test_bleu = round(metrics[0]["test_bleu"], 4)
        test_cer = round(metrics[0]["test_cer"], 4)
        test_edit = round(metrics[0]["test_edit"], 4)
        time_taken = round((end_time - start_time) / 60, 2)

        print(f"Character error rate: {test_cer}, time_taken: {time_taken} m")
        self.assertGreaterEqual(test_bleu, _TEST_BLEU_SCORE)
        self.assertLessEqual(test_cer, _TEST_CHARACTER_ERROR_RATE)
        self.assertGreaterEqual(test_edit, _TEST_EDIT_DISTANCE)
        self.assertLess(time_taken, 20)
