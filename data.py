from collections import namedtuple
from typing import Optional, Tuple, Callable

from torch.utils.data import default_collate, DataLoader, Dataset, random_split
from torchvision.datasets import CelebA, FashionMNIST
import torchvision.transforms as T
from torch import Tensor


DataInfo = namedtuple("DataInfo", "image_channels image_size num_classes sigma_data")
DataLoaders = namedtuple("DataLoaders", "train valid")


def load_dataset_and_make_dataloaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoaders, DataInfo]:
    train_dataset, valid_dataset, data_info = load_dataset(dataset_name, root_dir)
    dl = make_dataloaders(
        train_dataset,
        valid_dataset,
        data_info.num_classes,
        batch_size,
        num_workers,
        pin_memory,
    )
    return dl, data_info


def load_dataset(
    dataset_name="FashionMNIST", root_dir="data"
) -> Tuple[Dataset, Dataset, DataInfo]:
    match dataset_name:
        case "FashionMNIST":
            t = T.Compose(
                [T.ToTensor(), T.Pad(2), T.Normalize(mean=(0.5,), std=(0.5,))]
            )
            train_dataset = FashionMNIST(root_dir, download=True, transform=t)
            train_dataset, valid_dataset = random_split(train_dataset, [50000, 10000])
            num_classes = 10

        case "CelebA":
            t = T.Compose(
                [
                    T.ToTensor(),
                    T.CenterCrop(178),
                    T.Resize(128, antialias=True),
                    T.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
            train_dataset = CelebA(root_dir, download=True, transform=t)
            train_dataset, valid_dataset = random_split(train_dataset, [150000, 12770])
            num_classes = None

        case other:
            raise RuntimeError("Unknown dataset: " + other)

    x, _ = next(iter(DataLoader(train_dataset, batch_size=10000, shuffle=True)))
    _, c, h, w = x.size()
    assert h == w
    sigma_data = x.std()

    return train_dataset, valid_dataset, DataInfo(c, h, num_classes, sigma_data)


def make_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    num_classes: Optional[int],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaders:
    collate_fn = (
        default_collate
        if num_classes is not None
        else lambda batch: (default_collate(batch)[0], None)
    )
    kwargs = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0),
        "pin_memory": pin_memory,
    }

    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs),
        valid=DataLoader(valid_dataset, batch_size=2 * batch_size, **kwargs),
    )


def minmax_scale(x: Tensor, new_min: float, new_max: float) -> Tensor:
    return (x - x.min()) / (x.max() - x.min()) * (new_max - new_min) - new_min


def inception_transform() -> Callable:
    preprocess = T.Compose(
        [
            T.Resize(299, antialias=True),
            T.CenterCrop(299),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    preprocess = lambda x: minmax_scale(preprocess(x), -1, 1)

    return preprocess
