"""IM2LATEX100K DataModule"""
from collections import Counter, OrderedDict
from pathlib import Path
from random import shuffle
from string import Template
from typing import Any, Sequence, Tuple, Union
import json
import argparse
import shutil
import tarfile

from PIL import Image
from torch.utils.data import Sampler
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import toml

from im2latex.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from im2latex.data.util import BaseDataset, convert_strings_to_labels

IMAGE_HEIGHT = None
IMAGE_WIDTH = None
MAX_LABEL_LENGTH = 150
MIN_COUNT = 10

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "im2latex_100k"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "im2latex_100k"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "im2latex_100k"
PROCESSED_DATA_FILETEMPLATE = Template("vocab_$min_count.json")


# TODO:
# 1. implement __repr__ 
# 2. add parameters to select normalization or raw latex
# 3. rename function
class Im2Latex100K(BaseDataModule):
    """
    Im2Latex100K DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace=None) -> None:
        super().__init__(args)
        self.min_count = self.args.get("min_count", MIN_COUNT)
        self.max_label_length = self.args.get("max_label_length", MAX_LABEL_LENGTH)
        self.augment = self.args.get("augment_data", "true").lower() == "true"

        self.PROCESSED_DATA_FILENAME = (
            PROCESSED_DATA_DIRNAME / PROCESSED_DATA_FILETEMPLATE.substitute(min_count=self.min_count)
        )

        self.data_dir = DL_DATA_DIRNAME
        if not self.PROCESSED_DATA_FILENAME.is_file():
            _download_and_process_im2latex(self.PROCESSED_DATA_FILENAME, self.min_count)
        with open(self.PROCESSED_DATA_FILENAME) as f:
            vocab = json.load(f)
        assert vocab.get("vocab") is not None
        self.mapping = list(vocab["vocab"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        
        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert self.max_label_length <= MAX_LABEL_LENGTH
        self.output_dims = (MAX_LABEL_LENGTH, 1)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        parser.add_argument("--min_count", type=int, default=MIN_COUNT)
        parser.add_argument("--max_label_length", type=int, default=MAX_LABEL_LENGTH)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
        if not self.PROCESSED_DATA_FILENAME.is_file():
            _download_and_process_im2latex(self.PROCESSED_DATA_FILENAME, self.min_count)

    def setup(self, stage=None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        pass
        def _load_dataset(split: str, augment: bool, max_label_length: int = self.max_label_length) -> BaseDataset:
            imgs_path, labels = load_processed_crops_and_labels(split, max_label_length)
            # length add 2 because of start and end tokens
            labels = convert_strings_to_labels(strings=labels, mapping=self.inverse_mapping, length=self.output_dims[0] + 2)
            transform = get_transform(augment=augment)  # type: ignore
            return Im2LatexDataset(imgs_path, labels, transform=transform)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(split="train", augment=self.augment)
            self.data_val = _load_dataset(split="validate", augment=self.augment)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", augment=False)
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=False,
            batch_sampler=BucketBatchSampler(self.data_train, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_sampler=BucketBatchSampler(self.data_val, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_sampler=BucketBatchSampler(self.data_test, self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )


def _download_and_process_im2latex(processed_data_filename, min_count: int = 10):
    metadata = toml.load(METADATA_FILENAME)
    for _metadata in metadata.values():
        if not (DL_DATA_DIRNAME / _metadata["filename"]).is_file():
            _download_raw_dataset(_metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata, processed_data_filename, min_count=min_count)
    

def _process_raw_dataset(metadata: dict, processed_data_filename: Union[Path, str], min_count: int = 10):
    image_tarfile = DL_DATA_DIRNAME / metadata["formula_images_processed"]["filename"]
    if not (PROCESSED_DATA_DIRNAME / "formula_images_processed").is_dir():
        print("Unzipping formula_images_processed.tar.gz...")
        with tarfile.open(image_tarfile, "r:gz") as tar_file:
            tar_file.extractall(PROCESSED_DATA_DIRNAME)

    print("Build vocabulary...")
    vocab = build_vocab(DL_DATA_DIRNAME, min_count=min_count)
    with open(processed_data_filename, "w") as f:
        json.dump({"vocab": vocab}, f)

    for _metadata in metadata.values():
        if not (PROCESSED_DATA_DIRNAME / _metadata["filename"]).is_file() \
            and (DL_DATA_DIRNAME / _metadata["filename"]).suffix == ".lst":
            shutil.copy(DL_DATA_DIRNAME / _metadata["filename"],
                        PROCESSED_DATA_DIRNAME / _metadata["filename"])


def load_processed_crops_and_labels(split: str, max_label_length) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed images and labels for given split."""
    with open(PROCESSED_DATA_DIRNAME / "im2latex_formulas.norm.lst", 'r') as f:
        all_formulas = [formula.strip('\n') for formula in f.readlines()]

    imgs_path = []
    formulas = []
    with open(_labels_filename(split), 'r') as f:
        for line in f:
            id_, ind = line.strip('\n').split()
            formula = all_formulas[int(ind)].split()
            if len(formula) > max_label_length:
                continue
            formulas.append(formula)
            imgs_path.append(_crop_filename(id_, split))
            
    assert len(imgs_path) == len(formulas)
    return imgs_path, formulas


def get_transform(augment: bool) -> transforms.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-10, 10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    else:
        transforms_list = []
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / f"im2latex_{split}_filter.lst"


def _crop_filename(id_: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / "formula_images_processed" / f"{id_}"


class Im2LatexDataset(BaseDataset):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        datum, target = Image.open(self.data[index]).convert("L"), self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


# https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13
class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, p[0].shape))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


def build_vocab(lstdir: Union[Path, str], min_count: int=10) -> Sequence[str]:
    """Add the mapping with special symbols."""

    lstdir = Path(lstdir)
    counter = Counter()
    vocab = []

    with open(lstdir / "im2latex_formulas.norm.lst", 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(lstdir / "im2latex_train_filter.lst", 'r') as f:
        for line in f:
            _, idx = line.strip('\n').split()
            formula = formulas[int(idx)].split()
            counter.update(formula)
    
    for word, count in counter.most_common():
        if count >= min_count:
            vocab.append(word)
    
    # Also add special tokens:
    # - Unknown token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<U>", "<S>", "<E>", "<P>", *list(vocab)]


if __name__ == "__main__":
    load_and_print_info(Im2Latex100K)
