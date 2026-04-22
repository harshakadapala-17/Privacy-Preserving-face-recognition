"""
Stage 1 joint training: U-Net generator + ResNet-18 residue recognizer.

Objective: generator learns appearance reconstruction; residue recognizer
learns to classify identity from r = x - x'.

Combined loss: L = alpha * L1(x', x) + beta * CrossEntropy(f_r(r), labels)

Key fix: x_prime = generator(x) computed ONCE per batch and reused for both
the residue r and the generator loss lg.

Performance features:
  - Mixed precision (torch.cuda.amp) for ~1.5-2x speedup on T4 Tensor Cores
  - tqdm progress bars with live loss/acc display and ETA
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


def run_stage1(
    generator: nn.Module,
    recognizer: nn.Module,
    mapper: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int = 15,
    alpha: float = 5.0,
    beta: float = 1.0,
    lr_gen: float = 5e-3,
    lr_fr: float = 1e-2,
    checkpoint_fn: Callable | None = None,
    wandb_run: object | None = None,
    start_epoch: int = 0,
    scaler: GradScaler | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Train generator and residue recognizer jointly.

    Args:
        generator:      UNetGenerator on device.
        recognizer:     ResNet-18 with 21-channel input on device.
        mapper:         WaveletMapper on device (not trained, no grad).
        train_loader:   Training DataLoader (batch_size=64, num_workers=4).
        val_loader:     Validation DataLoader.
        num_classes:    Number of identity classes.
        device:         Torch device.
        epochs:         Total training epochs.
        alpha:          Weight for generator L1 loss (default 5.0).
        beta:           Weight for recognizer CE loss (default 1.0).
        lr_gen:         SGD learning rate for generator.
        lr_fr:          SGD learning rate for recognizer.
        checkpoint_fn:  Optional callback(epoch, state_dict) called after each epoch.
        wandb_run:      Optional W&B run for per-epoch metric logging.
        start_epoch:    Epoch to resume from (0 = fresh start).
        scaler:         GradScaler instance (pre-restored from checkpoint if resuming).
                        If None, a new scaler is created (enabled only on CUDA).

    Returns:
        (gen_losses, fr_losses, val_accs) — one value per epoch trained.
    """
    use_amp = device.type == "cuda"
    if scaler is None:
        scaler = GradScaler(enabled=use_amp)

    opt = optim.SGD(
        [
            {"params": generator.parameters(), "lr": lr_gen},
            {"params": recognizer.parameters(), "lr": lr_fr},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit_gen = nn.L1Loss()
    crit_fr = nn.CrossEntropyLoss()

    # Fast-forward scheduler state when resuming mid-training
    for _ in range(start_epoch):
        scheduler.step()

    gen_losses: list[float] = []
    fr_losses: list[float] = []
    val_accs: list[float] = []

    epoch_times: list[float] = []

    epoch_bar = tqdm(
        range(start_epoch, epochs),
        desc="Stage 1",
        unit="epoch",
        initial=start_epoch,
        total=epochs,
    )

    for ep in epoch_bar:
        t_epoch_start = time.time()
        generator.train()
        recognizer.train()
        rg = rf = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"S1 Ep {ep + 1}/{epochs}",
            leave=False,
            unit="batch",
        )
        for imgs, lbls in batch_bar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()

            with autocast(enabled=use_amp):
                x = mapper.encode(imgs)
                x_prime = generator(x)        # ONE forward pass — reused below
                r = x - x_prime
                lg = crit_gen(x_prime, x)    # no second generator(x) call
                lf = crit_fr(recognizer(r), lbls)
                loss = alpha * lg + beta * lf

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            rg += lg.item()
            rf += lf.item()
            batch_bar.set_postfix(gen=f"{lg.item():.4f}", fr=f"{lf.item():.4f}")

        # Validation — also single encode per batch
        generator.eval()
        recognizer.eval()
        ok = tot = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with autocast(enabled=use_amp):
                    x = mapper.encode(imgs)
                    x_prime = generator(x)    # single encode, no double mapper.encode()
                    r = x - x_prime
                ok += recognizer(r).argmax(1).eq(lbls).sum().item()
                tot += lbls.size(0)

        n = len(train_loader)
        acc = 100.0 * ok / tot
        gl, fl = rg / n, rf / n
        gen_losses.append(gl)
        fr_losses.append(fl)
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
        epoch_bar.set_postfix(gen=f"{gl:.4f}", fr=f"{fl:.4f}", acc=f"{acc:.1f}%")
        tqdm.write(
            f"  Epoch {ep + 1:>2}/{epochs} | GenLoss {gl:.4f} | FRLoss {fl:.4f} "
            f"| ValAcc {acc:.2f}% | {timing_str}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "s1/gen_loss": gl,
                "s1/fr_loss": fl,
                "s1/val_acc": acc,
                "s1/epoch": ep + 1,
                "s1/epoch_time_sec": elapsed,
            })

        if checkpoint_fn is not None:
            checkpoint_fn(
                ep + 1,
                {
                    "generator": generator.state_dict(),
                    "recognizer": recognizer.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "gen_losses": gen_losses,
                    "fr_losses": fr_losses,
                    "val_accs": val_accs,
                    "epoch": ep + 1,
                },
            )

        scheduler.step()
        torch.cuda.empty_cache()

    tqdm.write(f"\nStage 1 peak val acc : {max(val_accs):.2f}%")
    tqdm.write(f"Final gen loss       : {gen_losses[-1]:.4f}  (target: < 0.05)")
    return gen_losses, fr_losses, val_accs
