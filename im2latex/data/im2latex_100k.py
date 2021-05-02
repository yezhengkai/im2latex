"""IM2LATEX100K DataModule"""
import argparse
import json
import pickle
import shutil
import tarfile
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import compress
from pathlib import Path
from random import shuffle
from typing import Callable, List, MutableMapping, Sequence, Union

import numpy as np
import toml
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

from im2latex.data.base_data_module import BaseDataModule, _download_raw_dataset, load_and_print_info
from im2latex.data.util import BaseDataset, SequenceOrTensor, convert_strings_to_labels

IMAGE_HEIGHT = None
IMAGE_WIDTH = None
MAX_LABEL_LENGTH = 150
MIN_COUNT = 10
IMAGE_HIGHT_WIDTH_GROUP = [
    (32, 128),
    (64, 128),
    (32, 160),
    (64, 160),
    (32, 192),
    (64, 192),
    (32, 224),
    (64, 224),
    (32, 256),
    (64, 256),
    (32, 320),
    (64, 320),
    (32, 384),
    (64, 384),
    (96, 384),
    (32, 480),
    (64, 480),
    (128, 480),
    (160, 480),
]


RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "im2latex_100k"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "im2latex_100k"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "im2latex_100k"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "data.pkl"

# TODO:
# 1. implement __repr__
# 2. add parameters to select normalization or raw latex
# 3. rename function


class Im2Latex100K(BaseDataModule):
    """
    Im2Latex100K DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__(args)
        self.min_count = self.args.get("min_count", MIN_COUNT)
        self.max_label_length = self.args.get("max_label_length", MAX_LABEL_LENGTH)
        self.augment = self.args.get("augment_data", "true").lower() == "true"
        self.data_dir = DL_DATA_DIRNAME

        self.prepare_data()
        with open(self.vocab_filename) as f:
            vocab_dict = json.load(f)
        assert vocab_dict.get("vocab") is not None
        self.mapping = list(vocab_dict["vocab"])  # label to string
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}  # string to label

        self.dims = list(map(lambda x: (1, *x), IMAGE_HIGHT_WIDTH_GROUP))  # (1, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert self.max_label_length <= MAX_LABEL_LENGTH
        self.output_dims = (MAX_LABEL_LENGTH, 1)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        parser.add_argument("--min_count", type=int, default=MIN_COUNT)
        parser.add_argument("--max_label_length", type=int, default=MAX_LABEL_LENGTH)
        return parser

    @property
    def vocab_filename(self):
        return PROCESSED_DATA_DIRNAME / f"min_count={self.min_count}.json"

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """
        if not self.vocab_filename.is_file():
            _download_and_process_im2latex(self.vocab_filename, self.min_count)

    def setup(self, stage=None) -> None:
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

        def _load_dataset(data_dict, split: str, augment: bool, max_label_length: int) -> BaseDataset:
            # https://stackoverflow.com/questions/13397385/python-filter-and-list-and-apply-filtered-indices-to-another-list
            selectors_length = list(map(lambda y: not len(y) > max_label_length, data_dict[f"y_{split}"]))
            selectors_shape = list(
                map(lambda hight_width: hight_width in IMAGE_HIGHT_WIDTH_GROUP, data_dict[f"x_shape_{split}"])
            )
            selectors = np.logical_and(selectors_length, selectors_shape)
            x = list(compress(data_dict[f"x_{split}"], selectors))
            y = list(compress(data_dict[f"y_{split}"], selectors))
            x_shape = list(compress(data_dict[f"x_shape_{split}"], selectors))
            y = convert_strings_to_labels(strings=y, mapping=self.inverse_mapping, length=self.output_dims[0] + 2)
            transform = get_transform(augment=augment)
            return Im2LatexDataset(x, y, x_shape, transform=transform)

        with open(PROCESSED_DATA_FILENAME, "rb") as f:
            data_dict = pickle.load(f)

        if stage == "fit" or stage is None:
            self.data_train = _load_dataset(data_dict, "train", self.augment, self.max_label_length)
            self.data_val = _load_dataset(data_dict, "validate", self.augment, self.max_label_length)

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(data_dict, "test", False, self.max_label_length)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=False,
            batch_sampler=BucketBatchSampler(
                list((i, data_shape) for i, data_shape in enumerate(self.data_train.data_shape)),
                self.batch_size,
                do_shuffle=True,
            ),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_sampler=BucketBatchSampler(
                list((i, data_shape) for i, data_shape in enumerate(self.data_val.data_shape)),
                self.batch_size,
                do_shuffle=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_sampler=BucketBatchSampler(
                list((i, data_shape) for i, data_shape in enumerate(self.data_test.data_shape)),
                self.batch_size,
                do_shuffle=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )


def _download_and_process_im2latex(vocab_filename, min_count: int = 10):
    metadata = toml.load(METADATA_FILENAME)
    for _metadata in metadata.values():
        if not (DL_DATA_DIRNAME / _metadata["filename"]).is_file():
            _download_raw_dataset(_metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata, vocab_filename, min_count=min_count)


def _process_raw_dataset(metadata: MutableMapping, vocab_filename: Union[Path, str], min_count: int = 10):
    # unzip tar file
    img_tarfile = DL_DATA_DIRNAME / metadata["formula_images_processed"]["filename"]
    if not (PROCESSED_DATA_DIRNAME / "formula_images_processed").is_dir():
        print("Unzipping formula_images_processed.tar.gz...")
        with tarfile.open(img_tarfile, "r:gz") as tar_file:
            tar_file.extractall(PROCESSED_DATA_DIRNAME)

    # move .lst files to PROCESSED_DATA_DIRNAME
    for _metadata in metadata.values():
        if (
            not (PROCESSED_DATA_DIRNAME / _metadata["filename"]).is_file()
            and (DL_DATA_DIRNAME / _metadata["filename"]).suffix == ".lst"
        ):
            shutil.copy(DL_DATA_DIRNAME / _metadata["filename"], PROCESSED_DATA_DIRNAME / _metadata["filename"])

    # save data.pkl
    if not PROCESSED_DATA_FILENAME.is_file():
        print(
            "Save `array (from image)`, `latex`, `image size` and `image file name` to the dictionary "
            + "with the corresponding keys `x_{split}`, `y_{split}`, `x_shape_{split}` and `img_filename_{split}`..."
        )
        data_dict: dict = {
            "x_train": [],
            "x_validate": [],
            "x_test": [],
            "y_train": [],
            "y_validate": [],
            "y_test": [],
            "x_shape_train": [],
            "x_shape_validate": [],
            "x_shape_test": [],
            "img_filename_train": [],
            "img_filename_validate": [],
            "img_filename_test": [],
        }
        formulas = get_all_formulas(split_it=True)  # list of list of str
        for split in ["train", "validate", "test"]:
            with open(get_list_filename(split), "r") as f:
                for line in f:
                    img_filename, idx = line.strip("\n").split()
                    formula = formulas[int(idx)]  # list of str
                    if len(formula) == 0:
                        pass
                    else:
                        data_dict[f"img_filename_{split}"].append(img_filename)
                        data_dict[f"y_{split}"].append(formula)

            with ThreadPoolExecutor() as executor:
                result_iterator = executor.map(_get_img_info, data_dict[f"img_filename_{split}"])

            for img_array, img_size in result_iterator:
                data_dict[f"x_{split}"].append(img_array)
                data_dict[f"x_shape_{split}"].append(img_size)

        print(f"Save the dictionary to {PROCESSED_DATA_FILENAME}...")
        with open(PROCESSED_DATA_FILENAME, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Build vocabulary...")
    vocab = build_vocab(min_count=min_count)
    with open(vocab_filename, "w") as f:
        json.dump({"vocab": vocab}, f)


def _get_img_info(img_id: str):
    img_filename = get_img_filename(img_id)
    img = Image.open(img_filename).convert("L")
    array = np.array(img)
    array_shape = array.shape  # (row, col) ((row, col) in array == (Height, Width) in image)
    return array, array_shape


def get_all_formulas(
    list_filename: Union[str, Path, None] = None, split_it: bool = False
) -> Union[List[List[str]], List[str]]:
    if list_filename is None:
        list_filename = PROCESSED_DATA_DIRNAME / "im2latex_formulas.norm.lst"
    with open(list_filename, "r") as f:
        formulas = [formula.strip("\n") for formula in f.readlines()]
    if split_it:
        formulas = list(map(str.split, formulas))

    return formulas


def get_list_filename(split: str) -> Path:
    """Return filename of im2latex_{split}_filter.lst."""
    return PROCESSED_DATA_DIRNAME / f"im2latex_{split}_filter.lst"


def get_img_filename(img_id: str) -> Path:
    """Return filename of processed crop."""
    if not img_id.endswith(".png"):
        img_id = img_id + ".png"
    return PROCESSED_DATA_DIRNAME / "formula_images_processed" / img_id


def get_transform(augment: bool) -> transforms.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(degrees=1, shear=(-10, 10), interpolation=transforms.InterpolationMode.BILINEAR,),
        ]
    else:
        transforms_list = []
    transforms_list.append(transforms.ToTensor())
    return transforms.Compose(transforms_list)


class Im2LatexDataset(BaseDataset):
    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        data_shape: List[tuple],
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        super().__init__(data, targets, transform=transform, target_transform=target_transform)
        self.data_shape = data_shape


# https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13
class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, idx_imgsize, batch_size, do_shuffle=False):
        self.idx_imgsize = idx_imgsize
        self.batch_size = batch_size
        self.shuffle = do_shuffle
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        if self.shuffle:
            shuffle(self.idx_imgsize)
        # Organize size, e.g., batch_map[(192, 32)] = [30, 124, 203, ...] <= indices of image of size (192, 32)
        batch_map = OrderedDict()
        for idx, imgsize in self.idx_imgsize:
            if imgsize not in batch_map:
                batch_map[imgsize] = [idx]
            else:
                batch_map[imgsize].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for _, indices in batch_map.items():
            for group in [indices[i : (i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they aren't ordered by bucket size
        if self.shuffle:
            shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


def build_vocab(min_count: int = 10) -> Sequence[str]:
    """Add the mapping with special symbols."""
    # listdir = Path(listdir)
    counter: Counter = Counter()
    vocab = []

    formulas = get_all_formulas(split_it=True)
    with open(get_list_filename("train"), "r") as f:
        for line in f:
            _, idx = line.strip("\n").split()
            formula = formulas[int(idx)]
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
