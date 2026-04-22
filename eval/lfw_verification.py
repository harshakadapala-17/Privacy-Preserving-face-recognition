"""
LFW (Labeled Faces in the Wild) verification benchmark.

Standard 6,000-pair protocol (3,000 genuine + 3,000 impostor) defined by the
LFW pairs.txt file published at http://vis-www.cs.umass.edu/lfw/pairs.txt.

Pipeline per image:
  PIL image (any size)
    -> Resize(112, 112) + ToTensor + Normalize
    -> WaveletMapper.encode  -> (1, 21, 56, 56)
    -> UNetGenerator         -> appearance x'
    -> residue r = x - x'   -> (1, 21, 56, 56)
    -> CancelableTransform   -> L2-normalised template (1, 512)

Metric: cosine similarity between template pairs.

Usage::

    results = run_lfw_verification(
        mapper, generator, ct, device,
        lfw_dir='/content/lfw_funneled',
        pairs_txt='/content/pairs.txt',
        key=12345,
        save_path='lfw_roc.png',
    )
    print(results['auc'], results['tar_at_far_01'], results['tar_at_far_1'])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torchvision import transforms

if TYPE_CHECKING:
    import torch.nn as nn

# Same normalisation as VGGFace2 training; add Resize since LFW is 250×250
_LFW_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _load_tensor(path: Path) -> torch.Tensor:
    """Load and preprocess one LFW image -> (1, 3, 112, 112) float32."""
    return _LFW_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0)


@torch.no_grad()
def _to_template(
    img: torch.Tensor,
    mapper: "nn.Module",
    generator: "nn.Module",
    ct: object,
    key: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one image tensor through the full pipeline -> template on CPU."""
    img = img.to(device)
    x = mapper.encode(img)
    r = x - generator(x)
    return ct.transform(r, key=key).cpu()


def _parse_pairs_txt(pairs_txt: Path) -> tuple[list[tuple], list[int]]:
    """Parse LFW pairs.txt into (name1, n1, name2, n2) list and label list.

    pairs.txt format::

        10\\t300              # n_folds  n_per_fold
        Abel_Pacheco\\t1\\t4  # same-person: name  img_num1  img_num2
        ...
        Abel_Pacheco\\t1\\tGeorge_W_Bush\\t1  # diff-person: n1 n_img1 n2 n_img2

    Returns:
        pairs:  list of (name1, n1, name2, n2) tuples
        labels: list of 1 (genuine) or 0 (impostor)
    """
    lines = pairs_txt.read_text().strip().splitlines()
    pairs: list[tuple] = []
    labels: list[int] = []

    for line in lines[1:]:           # skip the "n_folds  n_per_fold" header
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) == 3:          # same-person
            name, n1, n2 = parts
            pairs.append((name, int(n1), name, int(n2)))
            labels.append(1)
        elif len(parts) == 4:        # different-person
            n1, i1, n2, i2 = parts
            pairs.append((n1, int(i1), n2, int(i2)))
            labels.append(0)

    return pairs, labels


def _lfw_img_path(lfw_dir: Path, name: str, num: int) -> Path:
    """Return the expected image path for a given person name and image number."""
    return lfw_dir / name / f"{name}_{num:04d}.jpg"


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, far: float) -> float:
    """True Accept Rate at a target False Accept Rate via linear interpolation."""
    idx = np.searchsorted(fpr, far)
    if idx == 0:
        return float(tpr[0])
    if idx >= len(tpr):
        return float(tpr[-1])
    fpr0, fpr1 = fpr[idx - 1], fpr[idx]
    tpr0, tpr1 = tpr[idx - 1], tpr[idx]
    if fpr1 == fpr0:
        return float(tpr0)
    t = (far - fpr0) / (fpr1 - fpr0)
    return float(tpr0 + t * (tpr1 - tpr0))


def run_lfw_verification(
    mapper: "nn.Module",
    generator: "nn.Module",
    ct: object,
    device: torch.device,
    lfw_dir: str,
    pairs_txt: str,
    key: int = 12345,
    save_path: str = "lfw_roc.png",
) -> dict:
    """Run the standard LFW 6,000-pair verification benchmark.

    Args:
        mapper:    WaveletMapper (eval mode, on device).
        generator: Frozen UNetGenerator (eval mode, on device).
        ct:        CancelableTransform instance.
        device:    Torch device.
        lfw_dir:   Path to lfw_funneled/ directory containing per-person subdirs.
        pairs_txt: Path to the LFW pairs.txt annotation file.
        key:       Integer projection key (must match the key used during training).
        save_path: File path for the saved ROC + distribution PNG.

    Returns:
        Dictionary with keys:

        ==================  =====================================================
        auc                 float — verification AUC
        tar_at_far_01       float — TAR at FAR = 0.1%
        tar_at_far_1        float — TAR at FAR = 1.0%
        threshold           float — best cosine threshold from ROC
        similarities        np.ndarray — per-pair cosine similarities
        labels              np.ndarray — ground truth labels (1=genuine, 0=impostor)
        n_pairs             int — number of pairs evaluated
        n_missing           int — pairs skipped due to missing image files
        ==================  =====================================================
    """
    lfw_path   = Path(lfw_dir)
    pairs_path = Path(pairs_txt)

    if not pairs_path.exists():
        raise FileNotFoundError(f"LFW pairs.txt not found: {pairs_txt}")
    if not lfw_path.exists():
        raise FileNotFoundError(f"LFW image directory not found: {lfw_dir}")

    pairs, labels = _parse_pairs_txt(pairs_path)
    n_genuine  = sum(l == 1 for l in labels)
    n_impostor = sum(l == 0 for l in labels)
    print(f"LFW pairs: {n_genuine} genuine + {n_impostor} impostor = {len(pairs)} total")

    generator.eval()

    similarities: list[float] = []
    valid_labels: list[int]   = []
    n_missing = 0

    for i, ((name1, n1, name2, n2), label) in enumerate(zip(pairs, labels)):
        p1 = _lfw_img_path(lfw_path, name1, n1)
        p2 = _lfw_img_path(lfw_path, name2, n2)

        if not p1.exists() or not p2.exists():
            n_missing += 1
            continue

        try:
            t1  = _to_template(_load_tensor(p1), mapper, generator, ct, key, device)
            t2  = _to_template(_load_tensor(p2), mapper, generator, ct, key, device)
            sim = float((t1 * t2).sum())
        except Exception:
            n_missing += 1
            continue

        similarities.append(sim)
        valid_labels.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs ...")

    sims = np.array(similarities)
    lbls = np.array(valid_labels)

    fpr, tpr, thresholds = roc_curve(lbls, sims)
    lfw_auc = auc(fpr, tpr)
    tar_01  = _tar_at_far(fpr, tpr, 0.001)
    tar_1   = _tar_at_far(fpr, tpr, 0.010)
    best_i  = np.argmax(tpr - fpr)
    best_th = float(thresholds[best_i])

    print("\nLFW Verification Results")
    print("=" * 45)
    print(f"  Pairs evaluated : {len(sims):,}  ({n_missing} skipped)")
    print(f"  AUC             : {lfw_auc:.4f}")
    print(f"  TAR@FAR=0.1%    : {tar_01 * 100:.2f}%")
    print(f"  TAR@FAR=1.0%    : {tar_1  * 100:.2f}%")
    print(f"  Best threshold  : {best_th:.4f}")
    print(f"  Genuine  mean   : {sims[lbls == 1].mean():.4f}")
    print(f"  Impostor mean   : {sims[lbls == 0].mean():.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {lfw_auc:.4f}")
    ax1.scatter(
        [0.001, 0.010], [tar_01, tar_1], color="red", zorder=5,
        label=f"TAR@0.1%={tar_01*100:.1f}%  TAR@1%={tar_1*100:.1f}%",
    )
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax1.set(xlabel="FAR", ylabel="TAR", title="LFW Verification ROC")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    bins = np.linspace(-0.5, 1.0, 60)
    ax2.hist(sims[lbls == 1], bins, alpha=0.6, color="green",
             label=f"Genuine (n={int((lbls==1).sum())})")
    ax2.hist(sims[lbls == 0], bins, alpha=0.6, color="red",
             label=f"Impostor (n={int((lbls==0).sum())})")
    ax2.axvline(best_th, color="black", ls="--", lw=1.5,
                label=f"Best threshold={best_th:.3f}")
    ax2.set(xlabel="Cosine Similarity", ylabel="Count",
            title="LFW Similarity Distributions")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {save_path}")

    return {
        "auc":           lfw_auc,
        "tar_at_far_01": tar_01,
        "tar_at_far_1":  tar_1,
        "threshold":     best_th,
        "similarities":  sims,
        "labels":        lbls,
        "n_pairs":       len(sims),
        "n_missing":     n_missing,
    }
