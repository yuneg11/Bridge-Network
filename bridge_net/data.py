import os
import random
from itertools import chain
from typing import Dict, Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tabulate import tabulate

import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from nxcl.config import ConfigDict


__all__ = [
    "CIFAR_DATA_AUGMENTATION",
    "TINYIMAGENET200_DATA_AUGMENTATION",
    "IMAGENET1K_DATA_AUGMENTATION",
    "CIFAR",
    "TinyImageNet200",
    "ImageNet1k",
    "build_dataloaders",
]


CIFAR_DATA_AUGMENTATION = {
    "NONE": torchvision.transforms.ToTensor(),
    "STANDARD_TRAIN_TRANSFORM": torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]),
}


TINYIMAGENET200_DATA_AUGMENTATION = {
    "NONE": torchvision.transforms.ToTensor(),
    "STANDARD_TRAIN_TRANSFORM": torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(64, 4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]),
}


IMAGENET1K_DATA_AUGMENTATION = {
    "NONE": torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]),
    "STANDARD_TRAIN_TRANSFORM": torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]),
}


class CIFAR(torchvision.datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            name: str,
            split: str = "train",
            indices: List[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:

        assert split in ["train", "test"]

        if indices is None:
            indices = list(range(50000)) if split == "train" else list(range(10000))

        super(CIFAR, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.images  = np.load(os.path.join(root, f"{name}/{split}_images.npy"))[indices]
        self.labels  = np.load(os.path.join(root, f"{name}/{split}_labels.npy"))[indices]
        self.classes = list(range(100)) if "CIFAR100" in name else list(range(10))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.images[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def describe(self) -> str:
        NUM_COL = 5
        DATA = [[idx,len([e for e in self.labels if e == idx]),] for idx in self.classes]
        DATA.append(["Total", sum(e[1] for e in DATA)])
        DATA = [
            list(chain(*e)) for e in zip(
                *[(DATA + [["-", "-"]] * (NUM_COL - (len(DATA) % NUM_COL)))[i::NUM_COL] for i in range(NUM_COL)]
            )
        ]
        return tabulate(DATA, headers=["Class", "# Examples",] * NUM_COL)


class TinyImageNet200(torchvision.datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = "train",
            indices: List[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:

        assert split in ["train", "test"]

        if indices is None:
            indices = list(range(100000)) if split == "train" else list(range(10000))

        super(TinyImageNet200, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.images  = np.load(os.path.join(root, f"TinyImageNet200/{split}_images.npy"))[indices]
        self.labels  = np.load(os.path.join(root, f"TinyImageNet200/{split}_labels.npy"))[indices]
        self.classes = list(range(200))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = self.images[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def describe(self) -> str:
        NUM_COL = 5
        DATA = [[idx,len([e for e in self.labels if e == idx]),] for idx in self.classes]
        DATA.append(["Total", sum(e[1] for e in DATA)])
        DATA = [
            list(chain(*e)) for e in zip(
                *[(DATA + [["-", "-"]] * (NUM_COL - (len(DATA) % NUM_COL)))[i::NUM_COL] for i in range(NUM_COL)]
            )
        ]
        return tabulate(DATA, headers=["Class", "# Examples",] * NUM_COL)


class ImageNet1k(torchvision.datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            split: str = "train",
            indices: List[int] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:

        assert split in ["train", "valid"]

        if split == "train":
            root = os.path.join(root, "ImageNet1k/train/")
        elif split == "valid":
            root = os.path.join(root, "ImageNet1k/val/")

        if indices is None:
            indices = list(range(1281167)) if split == "train" else list(range(50000))

        super(ImageNet1k, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )

        self.samples = [self.samples[idx] for idx in indices]
        self.targets = [self.targets[idx] for idx in indices]

    def describe(self) -> str:
        NUM_COL = 5
        DATA = [[self.class_to_idx[c], len([e for e in self.targets if e == self.class_to_idx[c]]),] for c in self.classes]
        DATA.append(["Total", sum(e[1] for e in DATA)])
        DATA = [
            list(chain(*e)) for e in zip(
                *[(DATA + [["-", "-"]] * (NUM_COL - (len(DATA) % NUM_COL)))[i::NUM_COL] for i in range(NUM_COL)]
            )
        ]
        return tabulate(DATA, headers=["Class", "# Examples",] * NUM_COL)


def build_dataloaders(
        cfg: ConfigDict,
        root: str = "./datasets/",
        num_replicas: int = 1,
        rank: int = None,
        is_distributed: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> Dict[str, DataLoader]:
    """
    Build the dictionary of dataloaders.

    Args:
        cfg (ConfigDict) : configs.
        root (str) : root directory which contains datasets.

    Returns:
        dataloaders (dict) :
            its keys are ["dataloader", "trn_loader", "val_loader", "tst_loader"],
            and only "dataloader" returns examples with data augmentation.
    """
    assert cfg.DATASETS.NAME in [
        "CIFAR10", "CIFAR100", "TinyImageNet200", "ImageNet1k",
    ], f"Unknown cfg.DATASETS.NAME = \"{cfg.DATASETS.NAME}\""

    if cfg.DATASETS.NAME in ["CIFAR10", "CIFAR100",]:

        # set data augmentation strategy
        trn_transform = CIFAR_DATA_AUGMENTATION[cfg.DATASETS.CIFAR.DATA_AUGMENTATION]
        tst_transform = CIFAR_DATA_AUGMENTATION["NONE"]

        # split training and validation examples
        indices = list(range(50000))
        if cfg.DATASETS.CIFAR.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.CIFAR.TRAIN_INDICES[0] : cfg.DATASETS.CIFAR.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.CIFAR.VALID_INDICES[0] : cfg.DATASETS.CIFAR.VALID_INDICES[1]]

        # get datasets
        dataset = CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=trn_transform)
        trn_set = CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=trn_indices, transform=tst_transform)
        tst_set = CIFAR(root=root, name=cfg.DATASETS.NAME, split="test",  indices=None,        transform=tst_transform)
        val_set = CIFAR(root=root, name=cfg.DATASETS.NAME, split="train", indices=val_indices, transform=tst_transform) if val_indices else tst_set

    elif cfg.DATASETS.NAME in ["TinyImageNet200",]:

        # set data augmentation strategy
        trn_transform = TINYIMAGENET200_DATA_AUGMENTATION[cfg.DATASETS.TINY.DATA_AUGMENTATION]
        tst_transform = TINYIMAGENET200_DATA_AUGMENTATION["NONE"]

        # split training and validation examples
        indices = list(range(100000))
        if cfg.DATASETS.TINY.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.TINY.TRAIN_INDICES[0] : cfg.DATASETS.TINY.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.TINY.VALID_INDICES[0] : cfg.DATASETS.TINY.VALID_INDICES[1]]

        # get datasets
        dataset = TinyImageNet200(root=root, split="train", indices=trn_indices, transform=trn_transform,)
        trn_set = TinyImageNet200(root=root, split="train", indices=trn_indices, transform=tst_transform,)
        tst_set = TinyImageNet200(root=root, split="test",  indices=None,        transform=tst_transform,)
        val_set = TinyImageNet200(root=root, split="train", indices=val_indices, transform=tst_transform,) if val_indices else tst_set

    elif cfg.DATASETS.NAME in ["ImageNet1k",]:

        # set data augmentation strategy
        trn_transform = IMAGENET1K_DATA_AUGMENTATION[cfg.DATASETS.IMAGENET.DATA_AUGMENTATION]
        tst_transform = IMAGENET1K_DATA_AUGMENTATION["NONE"]

        # split training and validation examples
        indices = list(range(1281167))
        if cfg.DATASETS.IMAGENET.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)

        trn_indices = indices[cfg.DATASETS.IMAGENET.TRAIN_INDICES[0] : cfg.DATASETS.IMAGENET.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.IMAGENET.VALID_INDICES[0] : cfg.DATASETS.IMAGENET.VALID_INDICES[1]]

        # get datasets
        dataset = ImageNet1k(root=root, split="train", indices=trn_indices, transform=trn_transform,)
        trn_set = ImageNet1k(root=root, split="train", indices=trn_indices, transform=tst_transform,)
        tst_set = ImageNet1k(root=root, split="valid", indices=None,        transform=tst_transform,)
        val_set = ImageNet1k(root=root, split="train", indices=val_indices, transform=tst_transform,) if val_indices else tst_set

    # get dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.SOLVER.BATCH_SIZE / num_replicas),
        shuffle=False if is_distributed else True,
        sampler=DistributedSampler(dataset, num_replicas, rank, shuffle=True) if is_distributed else None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    trn_loader = DataLoader(
        dataset=trn_set,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    tst_loader = DataLoader(
        dataset=tst_set,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    aln_loader = DataLoader(
        dataset=Subset(trn_set, list(range(10000))),
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    dataloaders = {
        "dataloader": dataloader,
        "trn_loader": trn_loader,
        "val_loader": val_loader,
        "tst_loader": tst_loader,
        "aln_loader": aln_loader,
    }

    return dataloaders
