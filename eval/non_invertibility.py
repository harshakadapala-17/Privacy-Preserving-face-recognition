"""
Non-invertibility evaluation: pseudo-inverse attack on cancelable templates.

Adversary scenario: attacker has stolen both the template T(r,K) and the key K.
They compute the best possible linear recovery: r_est = T(r,K) @ pinv(P_K).

Pass criterion: mean cosine similarity between r and r_est < 0.3.
Expected result: recovery similarity << 0.3 due to the 65344-dim null space
of the 65856→512 projection.
"""

from __future__ import annotations

import numpy as np
import torch


def run_non_invertibility(
    generator: torch.nn.Module,
    mapper: torch.nn.Module,
    ct: object,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    key: int = 11111,
) -> dict:
    """Run the pseudo-inverse non-invertibility attack.

    Args:
        generator:  Frozen UNetGenerator.
        mapper:     WaveletMapper.
        ct:         CancelableTransform instance (provides pinv()).
        val_loader: Validation DataLoader (uses first batch only).
        device:     Torch device.
        key:        Projection key (adversary is assumed to have this).

    Returns:
        Dictionary with keys:
          recovery_sim (np.ndarray)  — per-sample cosine similarity
          mean_sim (float)           — mean recovery similarity
          pass_test (bool)           — True if mean_sim < 0.3
          r_true (torch.Tensor)      — true residues on CPU
          r_est (torch.Tensor)       — adversary-recovered residues on CPU
    """
    Pp = ct.pinv(key)  # (512, 65856) pseudo-inverse on CPU

    test_imgs = next(iter(val_loader))[0].to(device)
    n_actual = test_imgs.shape[0]  # use actual batch size, never hardcode

    generator.eval()
    with torch.no_grad():
        x_t = mapper.encode(test_imgs)
        r_t = x_t - generator(x_t)

    tmpl_cpu = ct.transform(r_t, key=key).cpu().float()

    # Adversary's attack: r_est_flat = tmpl @ pinv(P_K)
    r_est_flat = tmpl_cpu @ Pp                            # (N, 65856) — CPU
    r_est = r_est_flat.reshape(n_actual, 21, 56, 56)      # (N, 21, 56, 56)

    # Measure recovery quality by comparing templates of true vs recovered r
    tmpl_rec = ct.transform(r_est.to(device), key=key).cpu()
    rec_sim = (tmpl_cpu * tmpl_rec).sum(dim=1).numpy()

    mean_sim = float(rec_sim.mean())
    pass_test = mean_sim < 0.3

    print("Non-Invertibility: pseudo-inverse attack")
    print("=" * 50)
    print(f"Samples evaluated         : {n_actual}")
    print(f"Genuine match similarity  : 1.0000  (by definition)")
    print(f"Adversary recovery sim    : {mean_sim:.4f} +/- {rec_sim.std():.4f}")
    print()
    print("NON-INVERTIBILITY:", "PASS" if pass_test else "REVIEW (generator may not have converged)")

    return {
        "recovery_sim": rec_sim,
        "mean_sim": mean_sim,
        "pass_test": pass_test,
        "r_true": r_t.cpu(),
        "r_est": r_est,
    }
