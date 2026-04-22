"""
Stage 2 training: MLP recognizer on cancelable templates T(r, K).

Generator is frozen after Stage 1. A fresh 3-layer MLP f_p is trained
on projected templates T(r, K) where K is the fixed user key.

The projection matrix P_K (65856×512 = 128 MB) stays on CPU throughout
to avoid CUDA OOM on the T4. Only the (B, 512) template result moves to GPU.

Performance features:
  - Mixed precision (torch.cuda.amp) on the fp forward pass
  - tqdm progress bars with live loss/acc and ETA
  - Wall-clock epoch timing with remaining-time estimate
  - GradScaler state saved in checkpoints for safe resume
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


def build_mlp(proj_dim: int, num_classes: int) -> nn.Sequential:
    """Construct the cancelable template classifier (f_p).

    3-layer MLP with BatchNorm and Dropout for regularization.

    Args:
        proj_dim:    Cancelable template dimension (typically 512).
        num_classes: Number of identity classes.

    Returns:
        nn.Sequential: proj_dim → 1024 → 512 → num_classes.
    """
    return nn.Sequential(
        nn.Linear(proj_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def run_stage2(
    generator: nn.Module,
    fp: nn.Module,
    mapper: nn.Module,
    ct: object,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    user_key: int = 12345,
    epochs: int = 8,
    lr: float = 1e-3,
    checkpoint_fn: Callable | None = None,
    wandb_run: object | None = None,
    start_epoch: int = 0,
    scaler: GradScaler | None = None,
) -> tuple[list[float], list[float]]:
    """Train MLP on cancelable templates with frozen generator.

    Args:
        generator:      Frozen UNetGenerator (weights not updated).
        fp:             MLP recognizer to train (on device).
        mapper:         WaveletMapper on device.
        ct:             CancelableTransform instance.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        device:         Torch device.
        user_key:       Integer key for template projection.
        epochs:         Total training epochs (default 8).
        lr:             Adam learning rate for fp.
        checkpoint_fn:  Optional callback(epoch, state_dict) after each epoch.
        wandb_run:      Optional W&B run for per-epoch metric logging.
        start_epoch:    Epoch to resume from (0 = fresh start).
        scaler:         GradScaler instance (pre-restored from checkpoint if resuming).
                        If None, a new scaler is created (enabled only on CUDA).

    Returns:
        (losses, val_accs) — one value per epoch trained.
    """
    use_amp = device.type == "cuda"
    if scaler is None:
        scaler = GradScaler(enabled=use_amp)

    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(fp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    for _ in range(start_epoch):
        scheduler.step()

    losses: list[float] = []
    val_accs: list[float] = []

    epoch_times: list[float] = []

    epoch_bar = tqdm(
        range(start_epoch, epochs),
        desc="Stage 2",
        unit="epoch",
        initial=start_epoch,
        total=epochs,
    )

    for ep in epoch_bar:
        t_epoch_start = time.time()
        fp.train()
        rl = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"S2 Ep {ep + 1}/{epochs}",
            leave=False,
            unit="batch",
        )
        for imgs, lbls in batch_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()

            with torch.no_grad():
                with autocast(enabled=use_amp):
                    x = mapper.encode(imgs)
                    r = x - generator(x)
            # Projection happens on CPU inside ct.transform — no GPU memory for P_K
            tmpl = ct.transform(r, key=user_key)  # (B, 512) back on GPU

            with autocast(enabled=use_amp):
                loss = crit(fp(tmpl), lbls)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            rl += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        fp.eval()
        ok = tot = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast(enabled=use_amp):
                    x = mapper.encode(imgs)
                    r = x - generator(x)
                tmpl = ct.transform(r, key=user_key)
                ok += fp(tmpl).argmax(1).eq(lbls).sum().item()
                tot += lbls.size(0)

        n = len(train_loader)
        acc = 100.0 * ok / tot
        avg_loss = rl / n
        losses.append(avg_loss)
        val_accs.append(acc)

        # Epoch timing and remaining estimate
        elapsed = time.time() - t_epoch_start
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        epochs_left = epochs - (ep + 1)
        remaining = avg_epoch * epochs_left
        rem_h, rem_m = int(remaining // 3600), int((remaining % 3600) // 60)
        el_m, el_s = int(elapsed // 60), int(elapsed % 60)

        timing_str = f"{el_m}m {el_s}s — est. {rem_h}h {rem_m}m remaining"
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")
        tqdm.write(
            f"  Epoch {ep + 1:>2}/{epochs} | Loss {avg_loss:.4f} "
            f"| ValAcc {acc:.2f}% | {timing_str}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "s2/loss": avg_loss,
                "s2/val_acc": acc,
                "s2/epoch": ep + 1,
                "s2/epoch_time_sec": elapsed,
            })

        if checkpoint_fn is not None:
            checkpoint_fn(
                ep + 1,
                {
                    "fp": fp.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "losses": losses,
                    "val_accs": val_accs,
                    "epoch": ep + 1,
                },
            )

        scheduler.step()
        torch.cuda.empty_cache()

    tqdm.write(f"\nStage 2 peak val acc : {max(val_accs):.2f}%")
    return losses, val_accs
