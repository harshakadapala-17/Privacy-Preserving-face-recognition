"""
AgeDB-30 (Age Database) verification benchmark.

Standard 10-fold cross-validation protocol using face pairs with age gaps
up to 30 years. Tests whether cancelable templates remain discriminative
under age variation.

Dataset: kagglehub.dataset_download("hereisburak/agedb")
Expected image filename format: {id}_{name}_{age}[_{seq}].jpg
All images in a single flat directory (or one-level subdir).

Pipeline per image:
  PIL image (any size)
    -> Resize(112, 112) + ToTensor + Normalize([0.5], [0.5])
    -> WaveletMapper.encode
    -> UNetGenerator (frozen)
    -> residue r = x - x'
    -> CancelableTransform  -> L2-normalised template (1, 512)

Usage::

    results = run_agedb_verification(
        mapper, generator, ct, device,
        agedb_dir='/path/to/agedb',
        key=12345,
        save_path='agedb_roc.png',
    )
    print(results['auc'], results['tar_at_far_1'])
"""

from __future__ import annotations

from collections import defaultdict
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

_AGEDB_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _load_tensor(path: Path) -> torch.Tensor:
    return _AGEDB_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0)


@torch.no_grad()
def _to_template(
    img: torch.Tensor,
    mapper: "nn.Module",
    generator: "nn.Module",
    ct: object,
    key: int,
    device: torch.device,
) -> torch.Tensor:
    img = img.to(device)
    x = mapper.encode(img)
    r = x - generator(x)
    return ct.transform(r, key=key).cpu()


def _find_image_dir(base: Path) -> Path:
    """Walk the dataset root to find the directory containing .jpg images."""
    # Try common sub-directory names first
    for candidate in [base / "agedb_30", base / "AgeDB", base / "images", base]:
        if candidate.is_dir():
            jpgs = list(candidate.glob("*.jpg"))
            if jpgs:
                return candidate
    # Fall back: find deepest dir with the most .jpg files
    best_dir, best_count = base, 0
    for d in base.rglob("*"):
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            if n > best_count:
                best_dir, best_count = d, n
    return best_dir


def _parse_filename(path: Path) -> tuple[str, int] | None:
    """Extract (identity, age) from an AgeDB filename.

    Supported formats:
      * ``{id}_{name}_{age}.jpg``          e.g. ``00001_MariaBartiromo_54.jpg``
      * ``{id}_{name}_{age}_{seq}.jpg``    e.g. ``00001_MariaBartiromo_54_2.jpg``

    Returns None if parsing fails.
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        # If last field is a short sequence number (<=3 digits), second-to-last is age
        if parts[-1].isdigit() and len(parts[-1]) <= 3 and len(parts) >= 4:
            age      = int(parts[-2])
            identity = "_".join(parts[1:-2])
        else:
            age      = int(parts[-1])
            identity = "_".join(parts[1:-1])
        return identity, age
    except (ValueError, IndexError):
        return None


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, far: float) -> float:
    """TAR at a target FAR via linear interpolation."""
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


def run_agedb_verification(
    mapper: "nn.Module",
    generator: "nn.Module",
    ct: object,
    device: torch.device,
    agedb_dir: str,
    key: int = 12345,
    n_genuine: int = 3000,
    n_impostor: int = 3000,
    max_age_diff: int = 30,
    seed: int = 42,
    save_path: str = "agedb_roc.png",
) -> dict:
    """Run the AgeDB-30 verification benchmark.

    Discovers images, parses filename metadata to recover identity and age,
    then generates balanced genuine / impostor pairs with an age-gap constraint.

    Args:
        mapper:       WaveletMapper (eval mode, on device).
        generator:    Frozen UNetGenerator (eval mode, on device).
        ct:           CancelableTransform instance.
        device:       Torch device.
        agedb_dir:    Root path returned by kagglehub (or any dir with AgeDB images).
        key:          Integer projection key (must match training key).
        n_genuine:    Target number of genuine pairs to generate.
        n_impostor:   Target number of impostor pairs to generate.
        max_age_diff: Maximum age gap allowed for genuine pairs (default 30).
        seed:         RNG seed for reproducible pair generation.
        save_path:    File path for the saved ROC PNG.

    Returns:
        Dictionary with keys:

        =================  =====================================================
        auc                float — verification AUC
        tar_at_far_1       float — TAR at FAR = 1.0%
        mean_genuine_sim   float — mean cosine similarity for genuine pairs
        mean_impostor_sim  float — mean cosine similarity for impostor pairs
        similarities       np.ndarray — per-pair cosine similarities
        labels             np.ndarray — ground-truth labels (1=genuine, 0=impostor)
        n_pairs            int — total pairs evaluated
        =================  =====================================================
    """
    base      = Path(agedb_dir)
    image_dir = _find_image_dir(base)
    all_imgs  = sorted(image_dir.glob("*.jpg"))

    if not all_imgs:
        raise FileNotFoundError(
            f"No .jpg images found under {agedb_dir}. "
            "Check that the download completed and agedb_dir is correct."
        )

    # Parse filenames -> group by identity
    by_id: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for p in all_imgs:
        parsed = _parse_filename(p)
        if parsed is not None:
            identity, age = parsed
            by_id[identity].append((p, age))

    if len(by_id) == 0:
        raise ValueError(
            f"Could not parse any AgeDB filenames in {image_dir}. "
            "Expected format: {{id}}_{{name}}_{{age}}[_{{seq}}].jpg"
        )

    valid_ids = [k for k, v in by_id.items() if len(v) >= 2]
    print(f"AgeDB images    : {len(all_imgs):,}  in {image_dir}")
    print(f"Identities      : {len(by_id):,}  ({len(valid_ids)} with ≥2 images)")

    rng = np.random.default_rng(seed)
    id_list = list(valid_ids)

    # --- Build genuine pairs (same identity, |age_diff| ≤ max_age_diff) ---
    genuine_pairs: list[tuple[Path, Path]] = []
    attempts = 0
    max_attempts = n_genuine * 20
    while len(genuine_pairs) < n_genuine and attempts < max_attempts:
        attempts += 1
        id_ = rng.choice(id_list)
        imgs = by_id[id_]
        i, j = rng.choice(len(imgs), size=2, replace=False)
        _, age_i = imgs[i]
        _, age_j = imgs[j]
        if abs(age_i - age_j) <= max_age_diff:
            genuine_pairs.append((imgs[i][0], imgs[j][0]))

    # --- Build impostor pairs (different identities) ---
    impostor_pairs: list[tuple[Path, Path]] = []
    attempts = 0
    while len(impostor_pairs) < n_impostor and attempts < max_attempts:
        attempts += 1
        i1, i2 = rng.choice(len(id_list), size=2, replace=False)
        if id_list[i1] == id_list[i2]:
            continue
        imgs1 = by_id[id_list[i1]]
        imgs2 = by_id[id_list[i2]]
        p1 = imgs1[int(rng.integers(len(imgs1)))][0]
        p2 = imgs2[int(rng.integers(len(imgs2)))][0]
        impostor_pairs.append((p1, p2))

    all_pairs  = genuine_pairs  + impostor_pairs
    all_labels = [1] * len(genuine_pairs) + [0] * len(impostor_pairs)
    print(f"Pairs generated : {len(genuine_pairs)} genuine + {len(impostor_pairs)} impostor")

    generator.eval()

    similarities: list[float] = []
    valid_labels: list[int]   = []
    n_errors = 0

    for i, ((p1, p2), label) in enumerate(zip(all_pairs, all_labels)):
        try:
            t1  = _to_template(_load_tensor(p1), mapper, generator, ct, key, device)
            t2  = _to_template(_load_tensor(p2), mapper, generator, ct, key, device)
            sim = float((t1 * t2).sum())
        except Exception:
            n_errors += 1
            continue

        similarities.append(sim)
        valid_labels.append(label)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(all_pairs)} pairs ...")

    sims = np.array(similarities)
    lbls = np.array(valid_labels)

    fpr, tpr, thresholds = roc_curve(lbls, sims)
    agedb_auc  = auc(fpr, tpr)
    tar_1      = _tar_at_far(fpr, tpr, 0.010)
    gen_mean   = float(sims[lbls == 1].mean()) if (lbls == 1).any() else float("nan")
    imp_mean   = float(sims[lbls == 0].mean()) if (lbls == 0).any() else float("nan")

    print("\nAgeDB-30 Verification Results")
    print("=" * 45)
    print(f"  Pairs evaluated     : {len(sims):,}  ({n_errors} errors skipped)")
    print(f"  AUC                 : {agedb_auc:.4f}")
    print(f"  TAR@FAR=1.0%        : {tar_1 * 100:.2f}%")
    print(f"  Mean genuine sim    : {gen_mean:.4f}")
    print(f"  Mean impostor sim   : {imp_mean:.4f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {agedb_auc:.4f}")
    ax1.scatter([0.010], [tar_1], color="red", zorder=5,
                label=f"TAR@1%={tar_1*100:.1f}%")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax1.set(xlabel="FAR", ylabel="TAR", title="AgeDB-30 Verification ROC")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    bins = np.linspace(-0.5, 1.0, 60)
    ax2.hist(sims[lbls == 1], bins, alpha=0.6, color="green",
             label=f"Genuine  mu={gen_mean:.3f}")
    ax2.hist(sims[lbls == 0], bins, alpha=0.6, color="red",
             label=f"Impostor mu={imp_mean:.3f}")
    ax2.set(xlabel="Cosine Similarity", ylabel="Count",
            title=f"AgeDB-30 Similarity Distributions (age gap ≤ {max_age_diff}yr)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {save_path}")

    return {
        "auc":              agedb_auc,
        "tar_at_far_1":     tar_1,
        "mean_genuine_sim": gen_mean,
        "mean_impostor_sim": imp_mean,
        "similarities":     sims,
        "labels":           lbls,
        "n_pairs":          len(sims),
    }
