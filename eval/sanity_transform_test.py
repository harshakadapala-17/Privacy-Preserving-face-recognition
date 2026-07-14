"""
Pre-training sanity check for CancelableTransform, isolated from the generator.

WHY THIS EXISTS: Stage 1 training takes ~2-3 GPU hours on Colab. Before
spending that budget, we want to know whether a bad cancelability/
non-invertibility result would be a code bug in CancelableTransform or
simply an under-converged generator. This script answers that question in
seconds, on CPU, with synthetic residues standing in for a real generator's
output — no dataset, no GPU, no training run required.

Two synthetic regimes are compared:

  1. NOISE   — each sample's residue is independent iid Gaussian noise.
               This is what r = x - x' should look like once the generator
               has converged (gen_loss < 0.05): no shared structure at all.

  2. BIASED  — each sample's residue is a shared "bias" vector (same for every
               sample, standing in for a generator that has NOT converged and
               leaks a common reconstruction-error pattern into every residue)
               plus a small per-identity signal and small per-sample noise.

Expected outcome if CancelableTransform itself is correct:
  - NOISE regime:  cross-key AUC ~ 0.5, non-invertibility recovery_sim low.
  - BIASED regime: cross-key AUC measurably departs from 0.5 (reproducing the
                    failure mode seen in the real LFW run: AUC=0.2178), NOT
                    because keys leak into each other, but because same-key
                    impostor pairs both carry the shared bias term and end up
                    more similar to each other than genuine cross-key pairs of
                    the *same* sample projected through two independent keys
                    (which decorrelates regardless of the bias, since the two
                    projection matrices are independent draws).

If NOISE fails, the bug is in CancelableTransform (or this test) — do not
spend GPU hours until it's fixed. If only BIASED looks "bad" (as expected),
the transform is sound and any real-run failure is a convergence problem.

Run: python -m eval.sanity_transform_test
Exit code: 0 if both regimes match the expected pattern, 1 otherwise.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

from models.cancelable_transform import CancelableTransform

R_CH, R_HW = 21, 56          # matches real residue shape
N_IDENTITIES = 40
SAMPLES_PER_ID = 5
N_PAIRS = 800
KEY_A, KEY_B = 11111, 99999

# Pass/fail thresholds
NOISE_AUC_TOL = 0.15          # |auc - 0.5| must stay within this for "noise"
NOISE_RECOVERY_MAX = 0.30     # matches the real non-invertibility pass criterion
BIASED_AUC_MIN_DEVIATION = 0.15  # biased regime must depart from 0.5 by at least this much


def make_residues(mode: str) -> tuple[torch.Tensor, np.ndarray]:
    """Build synthetic (N, R_CH, R_HW, R_HW) residues and identity labels."""
    n = N_IDENTITIES * SAMPLES_PER_ID
    dim = R_CH * R_HW * R_HW
    labels = np.repeat(np.arange(N_IDENTITIES), SAMPLES_PER_ID)

    if mode == "noise":
        r = torch.randn(n, dim)
    elif mode == "biased":
        bias = torch.randn(dim) * 3.0          # shared "unconverged generator" leak
        id_signal = torch.randn(N_IDENTITIES, dim) * 0.5
        r = bias.unsqueeze(0) + id_signal[labels] + torch.randn(n, dim) * 0.3
    else:
        raise ValueError(mode)

    return r.reshape(n, R_CH, R_HW, R_HW), labels


def cross_key_auc(r: torch.Tensor, labels: np.ndarray, ct: CancelableTransform) -> tuple[float, float]:
    tmpl_a = ct.transform(r, key=KEY_A)
    tmpl_b = ct.transform(r, key=KEY_B)

    rng = np.random.default_rng(0)
    genuine_ck, impostor = [], []
    for _ in range(N_PAIRS):
        lbl = rng.choice(np.unique(labels))
        idx = np.where(labels == lbl)[0]
        i = rng.choice(idx)
        genuine_ck.append((tmpl_a[i] * tmpl_b[i]).sum().item())
        idx2 = np.where(labels != lbl)[0]
        k = rng.choice(idx2)
        impostor.append((tmpl_a[i] * tmpl_a[k]).sum().item())

    genuine_ck, impostor = np.array(genuine_ck), np.array(impostor)
    y = [1] * len(genuine_ck) + [0] * len(impostor)
    s = np.concatenate([genuine_ck, impostor])
    fpr, tpr, _ = roc_curve(y, s)
    return auc(fpr, tpr), abs(genuine_ck.mean() - impostor.mean())


def non_invertibility(r: torch.Tensor, ct: CancelableTransform) -> float:
    """Pinv-attack recovery similarity, measured directly in residue space
    (raw r_true vs raw r_est) — see eval/non_invertibility.py for why
    comparing re-projected templates instead would be tautologically 1.0.
    """
    Pp = ct.pinv(KEY_A)                                  # (512, in_dim) CPU
    tmpl = ct.transform(r, key=KEY_A).float()
    r_est_flat = tmpl @ Pp                               # adversary's linear recovery

    r_true_flat = r.reshape(r.shape[0], -1).float()
    r_true_n = F.normalize(r_true_flat, p=2, dim=1)
    r_est_n = F.normalize(r_est_flat, p=2, dim=1)
    return (r_true_n * r_est_n).sum(dim=1).mean().item()


def main() -> int:
    ct = CancelableTransform(r_ch=R_CH, r_hw=R_HW, proj_dim=512)

    results = {}
    print(f"{'regime':<8} {'cross-key AUC':>14} {'|ck-imp| diff':>14} {'recovery_sim':>13}")
    print("-" * 55)
    for mode in ("noise", "biased"):
        torch.manual_seed(0)
        r, labels = make_residues(mode)
        auc_ck, diff = cross_key_auc(r, labels, ct)
        rec_sim = non_invertibility(r, ct)
        results[mode] = {"auc": auc_ck, "diff": diff, "rec_sim": rec_sim}
        print(f"{mode:<8} {auc_ck:>14.4f} {diff:>14.4f} {rec_sim:>13.4f}")

    noise_auc_ok = abs(results["noise"]["auc"] - 0.5) <= NOISE_AUC_TOL
    noise_recovery_ok = results["noise"]["rec_sim"] < NOISE_RECOVERY_MAX
    biased_degraded = abs(results["biased"]["auc"] - 0.5) >= BIASED_AUC_MIN_DEVIATION

    print()
    print("=" * 55)
    print("PASS/FAIL SUMMARY")
    print("=" * 55)
    print(f"[{'PASS' if noise_auc_ok else 'FAIL'}] noise regime cross-key AUC ~ 0.5 "
          f"(got {results['noise']['auc']:.4f}, tolerance +/-{NOISE_AUC_TOL})")
    print(f"[{'PASS' if noise_recovery_ok else 'FAIL'}] noise regime recovery_sim < {NOISE_RECOVERY_MAX} "
          f"(got {results['noise']['rec_sim']:.4f})")
    print(f"[{'PASS' if biased_degraded else 'FAIL'}] biased regime cross-key AUC deviates from 0.5 "
          f"by >= {BIASED_AUC_MIN_DEVIATION} (got {abs(results['biased']['auc'] - 0.5):.4f}) "
          f"-- confirms the diagnostic detects an under-converged generator")

    all_pass = noise_auc_ok and noise_recovery_ok and biased_degraded
    print()
    if all_pass:
        print("OVERALL: PASS -- CancelableTransform is mathematically sound.")
        print("A bad cross-key AUC / recovery_sim in the real training run points to")
        print("generator non-convergence, not a transform bug. Safe to spend GPU hours.")
    else:
        print("OVERALL: FAIL -- investigate CancelableTransform before training.")
        print("A noise-regime failure here means the transform itself is broken and")
        print("no amount of generator training will fix cross-key AUC / recovery_sim.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
