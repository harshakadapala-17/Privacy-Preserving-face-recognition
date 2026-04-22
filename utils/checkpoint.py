"""
Google Drive checkpoint save/load utilities for Colab training.

Saves a full state dict (models + optimizer + scheduler + scaler) after every epoch.
On the next Colab session, auto-resume logic finds the latest checkpoint and skips
already-trained epochs. The GradScaler state is saved so mixed-precision training
resumes correctly without loss spikes.

Usage:
    ckpt_fn = make_checkpoint_fn("/content/drive/MyDrive/cancelable_minusface", "stage1")
    state   = load_latest_checkpoint("/content/drive/MyDrive/cancelable_minusface", "stage1")
    if state is not None:
        start_epoch = restore_stage1(state, generator, recognizer)
        scaler_s1.load_state_dict(state["scaler"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.cuda.amp import GradScaler


def make_checkpoint_fn(
    save_dir: str,
    prefix: str,
    keep_last: int = 2,
) -> Callable[[int, dict], None]:
    """Create a checkpoint save callback for use in training loops.

    Args:
        save_dir:  Directory path (Google Drive mount point on Colab).
        prefix:    File name prefix, e.g. 'stage1' or 'stage2'.
        keep_last: How many most recent checkpoints to retain.

    Returns:
        Callable checkpoint_fn(epoch, state_dict) that saves to save_dir.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    def _save(epoch: int, state: dict) -> None:
        path = save_path / f"{prefix}_epoch{epoch:03d}.pt"
        torch.save(state, path)
        print(f"  [ckpt] saved -> {path}")
        # Remove older checkpoints beyond keep_last
        existing = sorted(save_path.glob(f"{prefix}_epoch*.pt"))
        for old in existing[:-keep_last]:
            old.unlink()

    return _save


def load_latest_checkpoint(save_dir: str, prefix: str) -> dict | None:
    """Load the most recent checkpoint for a given prefix.

    Args:
        save_dir: Directory path containing checkpoints.
        prefix:   File name prefix used when saving.

    Returns:
        State dict from most recent checkpoint, or None if no checkpoint found.
    """
    candidates = sorted(Path(save_dir).glob(f"{prefix}_epoch*.pt"))
    if not candidates:
        print(f"  [ckpt] no checkpoint found for '{prefix}' in {save_dir}")
        return None
    latest = candidates[-1]
    print(f"  [ckpt] resuming from {latest}")
    return torch.load(latest, map_location="cpu")


def restore_stage1(
    state: dict,
    generator: torch.nn.Module,
    recognizer: torch.nn.Module,
    scaler: GradScaler | None = None,
) -> int:
    """Restore Stage 1 model weights (and optionally GradScaler) from a checkpoint.

    The scaler must be restored BEFORE the first backward pass of the resumed
    session, otherwise the loss scale starts at its default value and may spike.

    Args:
        state:      State dict loaded from checkpoint.
        generator:  UNetGenerator instance.
        recognizer: ResNet-18 residue recognizer instance.
        scaler:     GradScaler to restore. If provided and checkpoint has 'scaler'
                    key, the scaler state is loaded. Pass None to skip.

    Returns:
        The epoch number to resume from (= last completed epoch).
    """
    generator.load_state_dict(state["generator"])
    recognizer.load_state_dict(state["recognizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
        print(f"  [ckpt] restored Stage 1 scaler state")
    print(f"  [ckpt] restored Stage 1 at epoch {state['epoch']}")
    return state["epoch"]


def restore_stage2(
    state: dict,
    fp: torch.nn.Module,
    scaler: GradScaler | None = None,
) -> int:
    """Restore Stage 2 model weights (and optionally GradScaler) from a checkpoint.

    Args:
        state:  State dict loaded from checkpoint.
        fp:     MLP recognizer instance.
        scaler: GradScaler to restore. If provided and checkpoint has 'scaler'
                key, the scaler state is loaded. Pass None to skip.

    Returns:
        The epoch number to resume from.
    """
    fp.load_state_dict(state["fp"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
        print(f"  [ckpt] restored Stage 2 scaler state")
    print(f"  [ckpt] restored Stage 2 at epoch {state['epoch']}")
    return state["epoch"]
