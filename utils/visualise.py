"""
Visualization utilities for the Cancelable MinusFace pipeline.

Produces all 6 output plots:
  pipeline.png        — 4-step pipeline overview
  stage1.png          — Stage 1 loss and accuracy curves
  stage2.png          — Stage 2 loss and accuracy curves
  cancelability.png   — similarity distributions + ROC curves
  noninvert.png       — true vs adversary-recovered residues
  cancellation_demo.png — cancellation lifecycle bar chart
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _denorm(t: "torch.Tensor") -> np.ndarray:
    """Denormalize a (1, C, H, W) tensor to HWC numpy array in [0, 1]."""
    return (t[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy() + 1) / 2


def plot_pipeline(
    orig: "torch.Tensor",
    regen: "torch.Tensor",
    residue: "torch.Tensor",
    template: "torch.Tensor",
    out_path: str = "pipeline.png",
) -> None:
    """Plot the 4-step pipeline: original -> regen -> residue -> template heatmap.

    Args:
        orig:     Decoded original face image (1, 3, H, W).
        regen:    Decoded regenerated image (1, 3, H, W).
        residue:  Decoded residue image (1, 3, H, W).
        template: Cancelable template tensor (1, 512).
        out_path: Save path for the PNG.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    items = [
        (_denorm(orig), "1. Original X", "input face"),
        (_denorm(regen), "2. Regeneration X'", "g(e(X))"),
        (_denorm(residue), "3. Residue r", "x - x'"),
        (None, "4. Template T(r,K)", f"{template.shape[1]}-dim vector"),
    ]
    for ax, (im, title, sub) in zip(axes, items):
        if im is None:
            ax.imshow(template[0].cpu().numpy().reshape(16, 32), cmap="RdBu_r", aspect="auto")
        else:
            ax.imshow(im)
        ax.set_title(f"{title}\n{sub}", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Cancelable MinusFace Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")


def plot_stage1(
    gen_losses: list[float],
    fr_losses: list[float],
    val_accs: list[float],
    out_path: str = "stage1.png",
) -> None:
    """Plot Stage 1 loss curves and validation accuracy.

    Args:
        gen_losses: Generator L1 loss per epoch.
        fr_losses:  Residue recognizer CE loss per epoch.
        val_accs:   Validation accuracy (%) per epoch.
        out_path:   Save path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(gen_losses) + 1)
    ax1.plot(ep, gen_losses, "b-o", label="Gen L1")
    ax1.plot(ep, fr_losses, "r-s", label="FR CE")
    ax1.axhline(0.05, color="gray", ls="--", lw=1, label="Target 0.05")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Stage 1 Losses")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(ep, val_accs, "g-o")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Stage 1 Val Accuracy (raw r)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")


def plot_stage2(
    losses: list[float],
    val_accs: list[float],
    out_path: str = "stage2.png",
) -> None:
    """Plot Stage 2 loss and accuracy curves.

    Args:
        losses:   CE loss per epoch.
        val_accs: Validation accuracy (%) per epoch.
        out_path: Save path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(losses) + 1)
    ax1.plot(ep, losses, "b-o")
    ax1.set(xlabel="Epoch", ylabel="CE Loss", title="Stage 2 Loss")
    ax1.grid(alpha=0.3)
    ax2.plot(ep, val_accs, "g-o")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Stage 2 Val Accuracy on T(r,K)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")


def plot_cancelability(
    genuine_sk: np.ndarray,
    genuine_ck: np.ndarray,
    impostor: np.ndarray,
    fpr_sk: np.ndarray,
    tpr_sk: np.ndarray,
    auc_sk: float,
    fpr_ck: np.ndarray,
    tpr_ck: np.ndarray,
    auc_ck: float,
    out_path: str = "cancelability.png",
) -> None:
    """Plot similarity distributions and ROC curves for cancelability experiment.

    Args:
        genuine_sk:      Same-key genuine cosine similarities.
        genuine_ck:      Cross-key cosine similarities.
        impostor:        Impostor cosine similarities.
        fpr_sk/tpr_sk:   ROC curve arrays for same-key.
        auc_sk:          AUC for same-key ROC.
        fpr_ck/tpr_ck:   ROC curve arrays for cross-key.
        auc_ck:          AUC for cross-key ROC.
        out_path:        Save path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-0.4, 0.8, 60)
    ax1.hist(genuine_sk, bins, alpha=0.6, color="green",
             label=f"Genuine same-key (mu={genuine_sk.mean():.3f})")
    ax1.hist(genuine_ck, bins, alpha=0.6, color="orange",
             label=f"Cross-key K_A->K_B (mu={genuine_ck.mean():.3f})")
    ax1.hist(impostor, bins, alpha=0.6, color="red",
             label=f"Impostor (mu={impostor.mean():.3f})")
    ax1.set(xlabel="Cosine Similarity", ylabel="Count", title="Similarity Distributions")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(fpr_sk, tpr_sk, "g-", lw=2, label=f"Same-key AUC={auc_sk:.3f}")
    ax2.plot(fpr_ck, tpr_ck, color="orange", lw=2, ls="--",
             label=f"Cross-key AUC={auc_ck:.3f} (ideal~0.5)")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax2.set(xlabel="FPR", ylabel="TPR", title="ROC - usability vs unlinkability")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")


def plot_non_invertibility(
    r_true: "torch.Tensor",
    r_est: "torch.Tensor",
    rec_sim: np.ndarray,
    mapper: object,
    out_path: str = "noninvert.png",
    n_show: int = 5,
) -> None:
    """Plot true residues vs adversary-recovered residues.

    Args:
        r_true:   True residues (N, 21, 56, 56) on CPU.
        r_est:    Recovered residues (N, 21, 56, 56) on CPU.
        rec_sim:  Per-sample recovery cosine similarities.
        mapper:   WaveletMapper for spatial decoding.
        out_path: Save path.
        n_show:   Number of sample columns to display.
    """
    n_show = min(n_show, r_true.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
    for col in range(n_show):
        r_dec = mapper.decode(r_true[col : col + 1])
        re_dec = mapper.decode(r_est[col : col + 1])
        axes[0, col].imshow(_denorm(r_dec))
        axes[0, col].set_title("True residue", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(_denorm(re_dec))
        axes[1, col].set_title(f"Recovered\nsim={rec_sim[col]:.3f}", fontsize=8)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("True r", fontsize=9)
    axes[1, 0].set_ylabel("Adversary", fontsize=9)
    plt.suptitle("Non-Invertibility: adversary cannot recover face from T(r,K)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")


def plot_cancellation_demo(results: dict, out_path: str = "cancellation_demo.png") -> None:
    """Plot the cancellation demo timeline as a bar chart.

    Args:
        results:  Output dict from run_cancellation_demo().
        out_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = [
        f"Enroll\nK={results['k_orig']}",
        f"Auth\nK={results['k_orig']}",
        "CANCEL\nKey K",
        f"Re-enroll\nK'={results['k_new']}",
        "Adversary\nAttack",
        f"Auth\nK'={results['k_new']}",
    ]
    sims = [
        results["sim_before"],
        results["sim_before"],
        0.0,
        results["sim_after"],
        results["template_linkage"],
        results["sim_after"],
    ]
    colors = ["green", "green", "red", "blue", "orange", "green"]
    bars = ax.bar(steps, sims, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0.2, color="black", ls="--", lw=1.5, label="Auth threshold (0.2)")
    ax.set(ylabel="Cosine Similarity", title="Cancellation & Re-Enrollment Demo Timeline")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for bar, sim in zip(bars, sims):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{sim:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[viz] saved {out_path}")
