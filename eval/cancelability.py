"""
Cancelability evaluation: same-key vs cross-key vs impostor similarity distributions.

Compares three cosine similarity distributions to verify key-based unlinkability:
  - Genuine same-key:  same person, key A → key A  (usability — should be high)
  - Cross-key:         same person, key A → key B  (unlinkability — should be ~0)
  - Impostor:          different people, key A      (security baseline)

Pass criterion: |cross_key_mean - impostor_mean| < 0.05
ROC curves: same-key AUC should be high; cross-key AUC should be ~0.5.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


def collect_residues(
    generator: torch.nn.Module,
    mapper: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> tuple[torch.Tensor, np.ndarray]:
    """Collect residues and labels from validation loader (on CPU).

    Args:
        generator:   Frozen UNetGenerator.
        mapper:      WaveletMapper.
        val_loader:  Validation DataLoader.
        device:      Torch device.
        max_samples: Maximum samples to collect.

    Returns:
        (all_r, all_lbls) — residues on CPU, labels as numpy int array.
    """
    all_r: list[torch.Tensor] = []
    all_lbls: list[torch.Tensor] = []
    generator.eval()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            x = mapper.encode(imgs.to(device))
            r = (x - generator(x)).cpu()
            all_r.append(r)
            all_lbls.append(lbls)
            if sum(len(l) for l in all_lbls) >= max_samples:
                break
    return torch.cat(all_r)[:max_samples], torch.cat(all_lbls)[:max_samples].numpy()


def batch_transform(
    r_tensor: torch.Tensor,
    ct: object,
    key: int,
    device: torch.device,
    bs: int = 64,
) -> torch.Tensor:
    """Apply cancelable transform in batches to avoid large CPU tensors.

    Args:
        r_tensor: Residues on CPU, shape (N, 21, 56, 56).
        ct:       CancelableTransform instance.
        key:      Projection key.
        device:   Device to run transform on.
        bs:       Batch size for processing.

    Returns:
        Templates on CPU, shape (N, 512).
    """
    out = []
    for i in range(0, len(r_tensor), bs):
        out.append(ct.transform(r_tensor[i : i + bs].to(device), key).cpu())
    return torch.cat(out)


def run_cancelability(
    generator: torch.nn.Module,
    mapper: torch.nn.Module,
    ct: object,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    key_a: int = 11111,
    key_b: int = 99999,
    n_pairs: int = 500,
    max_samples: int = 1000,
) -> dict:
    """Run the cancelability experiment.

    Computes genuine same-key, cross-key, and impostor cosine similarity
    distributions, plus ROC curves with AUC for each.

    Args:
        generator:   Frozen UNetGenerator.
        mapper:      WaveletMapper.
        ct:          CancelableTransform instance.
        val_loader:  Validation DataLoader.
        device:      Torch device.
        key_a:       First projection key (enrollment key).
        key_b:       Second projection key (simulates re-enrollment after cancel).
        n_pairs:     Number of pairs to sample.
        max_samples: Maximum validation samples to collect.

    Returns:
        Dictionary with keys:
          genuine_sk, genuine_ck, impostor  (np.ndarray of cosine sims)
          fpr_sk, tpr_sk, auc_same_key
          fpr_ck, tpr_ck, auc_cross_key
          unlinkable (bool), diff (float)
    """
    all_r, all_lbls = collect_residues(generator, mapper, val_loader, device, max_samples)
    tmpl_a = batch_transform(all_r, ct, key_a, device)
    tmpl_b = batch_transform(all_r, ct, key_b, device)

    rng = np.random.default_rng(0)
    genuine_sk_list: list[float] = []
    genuine_ck_list: list[float] = []
    impostor_list: list[float] = []

    for _ in range(n_pairs):
        lbl = rng.choice(np.unique(all_lbls))
        idx = np.where(all_lbls == lbl)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        genuine_sk_list.append((tmpl_a[i] * tmpl_a[j]).sum().item())
        genuine_ck_list.append((tmpl_a[i] * tmpl_b[i]).sum().item())
        idx2 = np.where(all_lbls != lbl)[0]
        k = rng.choice(idx2)
        impostor_list.append((tmpl_a[i] * tmpl_a[k]).sum().item())

    genuine_sk = np.array(genuine_sk_list)
    genuine_ck = np.array(genuine_ck_list)
    impostor = np.array(impostor_list)

    y_sk = [1] * len(genuine_sk) + [0] * len(impostor)
    s_sk = np.concatenate([genuine_sk, impostor])
    fpr1, tpr1, _ = roc_curve(y_sk, s_sk)
    auc_sk = auc(fpr1, tpr1)

    y_ck = [1] * len(genuine_ck) + [0] * len(impostor)
    s_ck = np.concatenate([genuine_ck, impostor])
    fpr2, tpr2, _ = roc_curve(y_ck, s_ck)
    auc_ck = auc(fpr2, tpr2)

    diff = abs(genuine_ck.mean() - impostor.mean())

    print("=" * 55)
    print(f"{'Genuine same-key mean:':<40} {genuine_sk.mean():>7.4f}")
    print(f"{'Cross-key (cancel+re-enroll) mean:':<40} {genuine_ck.mean():>7.4f}")
    print(f"{'Impostor mean:':<40} {impostor.mean():>7.4f}")
    print("=" * 55)
    print(f"\n|cross-key - impostor| = {diff:.4f}  (threshold 0.05)")
    print("UNLINKABILITY:", "PASS" if diff < 0.05 else "PARTIAL (needs more training)")
    print(f"Same-key  AUC = {auc_sk:.4f}  (usability,  higher better)")
    print(f"Cross-key AUC = {auc_ck:.4f}  (unlinkability, 0.5 = perfect)")

    return {
        "genuine_sk": genuine_sk,
        "genuine_ck": genuine_ck,
        "impostor": impostor,
        "fpr_sk": fpr1,
        "tpr_sk": tpr1,
        "auc_same_key": auc_sk,
        "fpr_ck": fpr2,
        "tpr_ck": tpr2,
        "auc_cross_key": auc_ck,
        "unlinkable": diff < 0.05,
        "diff": diff,
    }
