"""
End-to-end cancellation and re-enrollment demonstration.

Simulates the full lifecycle:
  1. Enroll with key K_orig
  2. Authenticate with K_orig         → GRANTED
  3. Cancel key K_orig (delete from DB)
  4. Re-enroll with new key K_new
  5. Adversary uses stolen template    → BLOCKED (linkage ≈ 0)
  6. Legitimate auth with K_new        → GRANTED

Key metric: |T(r, K_orig) · T(r, K_new)| ≈ 0 (unlinkable templates).
"""

from __future__ import annotations

import torch


def run_cancellation_demo(
    generator: torch.nn.Module,
    mapper: torch.nn.Module,
    ct: object,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k_orig: int = 55555,
    k_new: int = 77777,
    auth_threshold: float = 0.2,
) -> dict:
    """Simulate the full key cancellation and re-enrollment lifecycle.

    Uses the first two images from val_loader as enroll / probe images.

    Args:
        generator:      Frozen UNetGenerator.
        mapper:         WaveletMapper.
        ct:             CancelableTransform instance.
        val_loader:     Validation DataLoader.
        device:         Torch device.
        k_orig:         Original enrollment key.
        k_new:          New key after cancellation/re-enrollment.
        auth_threshold: Cosine similarity threshold for authentication grant.

    Returns:
        Dictionary with all step similarities and boolean outcomes:
          sim_before        — similarity of enroll vs probe under K_orig
          auth_before       — bool, whether pre-cancel auth passes
          template_linkage  — cosine similarity stolen_template vs new_template
          adversary_blocked — bool, whether stolen template is rejected
          sim_after         — similarity of enroll vs probe under K_new
          auth_after        — bool, whether post-cancel auth passes
          k_orig, k_new
    """
    imgs_demo, _ = next(iter(val_loader))
    enroll_img = imgs_demo[0:1].to(device)
    probe_img = imgs_demo[1:2].to(device)

    generator.eval()
    with torch.no_grad():
        x_e = mapper.encode(enroll_img)
        r_enroll = (x_e - generator(x_e)).cpu()
        x_p = mapper.encode(probe_img)
        r_probe = (x_p - generator(x_p)).cpu()

    tmpl_enrolled = ct.transform(r_enroll.to(device), key=k_orig).cpu()
    tmpl_probe_old = ct.transform(r_probe.to(device), key=k_orig).cpu()
    sim_before = (tmpl_enrolled * tmpl_probe_old).sum().item()

    # Adversary captures enrolled template before cancellation
    stolen_template = tmpl_enrolled.clone()

    # Step 3: cancel — old enrollment template removed from DB
    tmpl_enrolled = None

    # Step 4: re-enroll with new key
    tmpl_new_enrolled = ct.transform(r_enroll.to(device), key=k_new).cpu()

    linkage = (stolen_template * tmpl_new_enrolled).sum().item()

    tmpl_probe_new = ct.transform(r_probe.to(device), key=k_new).cpu()
    sim_after = (tmpl_new_enrolled * tmpl_probe_new).sum().item()

    print("=" * 60)
    print("CANCELLATION & RE-ENROLLMENT DEMO")
    print("=" * 60)
    print(f"Step 1 — Enrolled with key K={k_orig}")
    print(f"Step 2 — Auth with same key     : sim={sim_before:.4f}  "
          f"{'GRANTED' if sim_before > auth_threshold else 'DENIED'}")
    print(f"Step 3 — KEY K={k_orig} CANCELLED (deleted from DB)")
    print(f"Step 4 — Re-enrolled with K'={k_new}")
    print(f"Step 5 — Adversary stolen template: sim={linkage:.4f}  "
          f"{'BLOCKED' if linkage < auth_threshold else 'LINKED (bad!)'}")
    print(f"Step 6 — Legitimate auth with K'  : sim={sim_after:.4f}  "
          f"{'GRANTED' if sim_after > auth_threshold else 'DENIED'}")
    print("=" * 60)
    print(f"  Template linkage  : {linkage:.4f}  (target: < 0.05)")
    print("=" * 60)

    return {
        "sim_before": sim_before,
        "auth_before": sim_before > auth_threshold,
        "template_linkage": linkage,
        "adversary_blocked": linkage < auth_threshold,
        "sim_after": sim_after,
        "auth_after": sim_after > auth_threshold,
        "k_orig": k_orig,
        "k_new": k_new,
    }
