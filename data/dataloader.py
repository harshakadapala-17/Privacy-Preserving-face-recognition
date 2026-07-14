"""
VGGFace2 dataset loading utilities for Cancelable MinusFace.

Downloads the yakhyokhuja/vggface2-112x112 Kaggle dataset via kagglehub
and constructs remapped train/val DataLoaders. Images are already 112×112
so no Resize transform is applied.

Usage:
    image_dir = download_vggface2()
    train_loader, val_loader, num_classes = build_dataloaders(image_dir)
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def _find_image_root(base_path: str) -> str:
    """Walk downloaded path to find the directory containing identity folders.

    Picks whichever directory in the tree has the most immediate
    subdirectories, rather than assuming a naming convention (e.g. classic
    VGGFace2's "n000002" prefix) — repackaged Kaggle datasets often nest the
    real identity folders under a wrapper directory and/or use different
    naming, which silently defeated a name-prefix heuristic.
    """
    best_root, best_count = base_path, 0
    for root, dirs, _files in os.walk(base_path):
        if len(dirs) > best_count:
            best_root, best_count = root, len(dirs)
    return best_root


class _RemapDataset(Dataset):
    """Wraps an ImageFolder, filtering to valid indices and remapping labels to [0, N)."""

    def __init__(
        self,
        ds: datasets.ImageFolder,
        idx: list[int],
        remap: dict[int, int],
    ) -> None:
        self.ds = ds
        self.idx = idx
        self.remap = remap

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img, lbl = self.ds[self.idx[i]]
        return img, self.remap[lbl]


def build_dataloaders(
    image_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    min_images_per_identity: int = 10,
    imgs_per_identity: int = 100,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, int]:
    """Build train and validation DataLoaders for VGGFace2 112×112.

    Stratified subset: caps each identity at imgs_per_identity samples.
    This keeps all 8631 identities (full class diversity) while reducing
    total samples from 3.31M to ~863K — roughly 4× faster per epoch with
    no reduction in the number of classes the model must distinguish.

    Args:
        image_dir:               Root directory containing per-identity subdirs.
        batch_size:              Samples per batch.
        num_workers:             DataLoader worker processes.
        min_images_per_identity: Drop identities with fewer than this many images.
        imgs_per_identity:       Maximum images to keep per identity (stratified cap).
        val_fraction:            Fraction of data for validation.
        seed:                    Random seed for reproducible split and stratified sampling.

    Returns:
        (train_loader, val_loader, num_classes) tuple.
    """
    tfm = transforms.Compose([
        # No Resize — VGGFace2 112×112 is already the correct size
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_ds = datasets.ImageFolder(root=image_dir, transform=tfm)

    # Group sample indices by label
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, (_, lbl) in enumerate(full_ds.samples):
        by_label[lbl].append(i)

    # Filter to identities with enough images and remap labels contiguously
    valid_lbls = sorted(lbl for lbl, idxs in by_label.items() if len(idxs) >= min_images_per_identity)
    remap = {old: new for new, old in enumerate(valid_lbls)}
    num_classes = len(valid_lbls)

    # Stratified cap: keep at most imgs_per_identity per identity.
    # Using a seeded generator so subsets are reproducible across runs.
    rng = torch.Generator().manual_seed(seed)
    stratified_idx: list[int] = []
    for lbl in valid_lbls:
        idxs = by_label[lbl]
        if len(idxs) > imgs_per_identity:
            perm = torch.randperm(len(idxs), generator=rng).tolist()
            idxs = [idxs[j] for j in perm[:imgs_per_identity]]
        stratified_idx.extend(idxs)

    remapped = _RemapDataset(full_ds, stratified_idx, remap)
    n_tr = int((1 - val_fraction) * len(remapped))
    train_set, val_set = random_split(
        remapped,
        [n_tr, len(remapped) - n_tr],
        generator=torch.Generator().manual_seed(seed),
    )

    # persistent_workers and prefetch_factor require num_workers > 0
    extra_kw: dict = {}
    if num_workers > 0:
        extra_kw = {"persistent_workers": True, "prefetch_factor": 2}

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **extra_kw,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **extra_kw,
    )

    total = len(remapped)
    print(f"Identities       : {num_classes:,}")
    print(f"Total samples    : {total:,}  (cap: {imgs_per_identity}/identity)")
    print(f"Train / Val      : {len(train_set):,} / {len(val_set):,}")

    return train_loader, val_loader, num_classes


def download_vggface2() -> str:
    """Download VGGFace2 112×112 from Kaggle via kagglehub.

    Returns:
        Path to the image root directory containing per-identity subdirs.
    """
    import kagglehub  # imported lazily — not always installed outside Colab

    base = kagglehub.dataset_download("yakhyokhuja/vggface2-112x112")
    return _find_image_root(base)
