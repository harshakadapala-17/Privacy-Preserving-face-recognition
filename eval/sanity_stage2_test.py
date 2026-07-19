"""
Pre-training sanity check for the Stage 2 training loop, isolated from the
generator/dataset/GPU.

WHY THIS EXISTS: a completed 8-epoch Stage 2 checkpoint showed
losses = [0.0]*8 and val_accs = [100.0]*8 from epoch 1, never changing. That
is NOT a sign of a broken training loop -- it's the deterministic,
mathematical signature of training a classifier with num_classes == 1:
softmax over a single logit is always exactly 1.0 regardless of its value, so
CrossEntropyLoss = -log(1.0) = 0.0 exactly, and its gradient is exactly zero
too (softmax_output - one_hot_target = 1 - 1 = 0), for every sample, every
epoch, independent of the frozen generator, the residue, the projection key,
or fp's weights. argmax of a single-output head is always class 0, which
trivially matches every label, since there's only one label that can exist.

Root cause: data/dataloader.py's `_find_image_root()` silently resolved to
the wrong directory for this Kaggle repackaging of VGGFace2, collapsing the
dataset to 1 identity ("class"), which the imgs_per_identity stratified cap
then shrank to 100 total images. That's fixed (see CHANGELOG.md), and
`build_dataloaders()` now raises a ValueError if num_classes < 2 instead of
silently producing a checkpoint that looks like a perfect classifier.

This script:
  1. Reproduces the exact zero-loss/100%-accuracy signature on synthetic
     num_classes=1 data, to document mathematically why it happens.
  2. Confirms build_dataloaders()'s guard actually rejects a synthetic
     single-identity image folder tree.
  3. Runs the same training-loop shape (build_mlp -> CrossEntropyLoss ->
     Adam) on a tiny but genuine multi-class synthetic dataset, confirming
     loss decreases from a nonzero starting point and accuracy improves
     gradually rather than being stuck at 100% from epoch 1 -- i.e. the
     actual Stage 2 training logic is sound.

Run: python -m eval.sanity_stage2_test
Exit code: 0 if the expected pattern holds in all three checks, 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training.stage2_train import build_mlp

PROJ_DIM = 512


def run_degenerate_single_class() -> tuple[list[float], list[float]]:
    """Train build_mlp(proj_dim, num_classes=1) on random data -- reproduces
    the exact zero-loss/100%-accuracy signature, regardless of the data.
    """
    torch.manual_seed(0)
    X = F.normalize(torch.randn(20, PROJ_DIM), p=2, dim=1)
    y = torch.zeros(20, dtype=torch.long)

    model = build_mlp(PROJ_DIM, num_classes=1)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    losses, accs = [], []
    for _ in range(5):
        model.train()
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item() * 100
        accs.append(acc)

    return losses, accs


def run_multiclass_training() -> tuple[list[float], list[float]]:
    """Train build_mlp on a small but genuine multi-class synthetic dataset --
    confirms the training loop shape itself is sound (loss actually falls,
    accuracy actually rises, neither is trivial from epoch 1).
    """
    torch.manual_seed(0)
    num_classes, n_per_class, noise = 6, 40, 0.9

    centers = torch.randn(num_classes, PROJ_DIM) * 2.0
    xs, ys = [], []
    for c in range(num_classes):
        xs.append(centers[c] + torch.randn(n_per_class, PROJ_DIM) * noise)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = F.normalize(torch.cat(xs), p=2, dim=1)
    y = torch.cat(ys)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    model = build_mlp(PROJ_DIM, num_classes=num_classes)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    losses, accs = [], []
    for _ in range(15):
        model.train()
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            acc = (model(X).argmax(1) == y).float().mean().item() * 100
        accs.append(acc)

    return losses, accs


def check_num_classes_guard() -> bool:
    """Build a synthetic single-identity image folder tree and confirm
    build_dataloaders() refuses it instead of silently training on it.
    """
    from data.dataloader import build_dataloaders

    tmp = tempfile.mkdtemp()
    try:
        identity_dir = os.path.join(tmp, "only_one_identity")
        os.makedirs(identity_dir)
        for i in range(20):
            # 1x1 pixel JPEGs -- the guard fires before any image is opened
            with open(os.path.join(identity_dir, f"{i:04d}.jpg"), "wb") as f:
                f.write(
                    bytes.fromhex(
                        "ffd8ffe000104a46494600010100000100010000ffdb004300"
                        "03020202020203020202030303030405080505040404080b06"
                        "0605080b0b09090a0a0b0a09090a0a0b0c0e0e0e0e0e0e0e0e"
                        "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e"
                        "0e0effc0000b080001000101011100ffc4001f0000010501010"
                        "101010100000000000000000102030405060708090a0bffda0"
                        "008010100003f00d2cf20ffd9"
                    )
                )
        try:
            build_dataloaders(tmp, batch_size=4, num_workers=0)
            return False  # should have raised
        except ValueError as e:
            return "single-class" in str(e) or "identity" in str(e)
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    degenerate_losses, degenerate_accs = run_degenerate_single_class()
    multiclass_losses, multiclass_accs = run_multiclass_training()
    guard_fires = check_num_classes_guard()

    print(f"{'check':<28} {'result':>25}")
    print("-" * 55)
    print(f"{'degenerate loss (all 0.0)':<28} {str(all(l == 0.0 for l in degenerate_losses)):>25}")
    print(f"{'degenerate acc (all 100.0)':<28} {str(all(a == 100.0 for a in degenerate_accs)):>25}")
    print(f"{'multiclass loss[0]':<28} {multiclass_losses[0]:>25.4f}")
    print(f"{'multiclass loss[-1]':<28} {multiclass_losses[-1]:>25.4f}")
    print(f"{'multiclass acc[0]':<28} {multiclass_accs[0]:>25.2f}")
    print(f"{'multiclass acc[-1]':<28} {multiclass_accs[-1]:>25.2f}")
    print(f"{'num_classes<2 guard fires':<28} {str(guard_fires):>25}")

    degenerate_reproduced = all(l == 0.0 for l in degenerate_losses) and all(
        a == 100.0 for a in degenerate_accs
    )
    multiclass_sound = (
        multiclass_losses[0] > 0.5
        and multiclass_losses[-1] < multiclass_losses[0]
        and multiclass_accs[0] < 90.0
        and multiclass_accs[-1] > multiclass_accs[0]
    )

    print()
    print("=" * 55)
    print("PASS/FAIL SUMMARY")
    print("=" * 55)
    print(f"[{'PASS' if degenerate_reproduced else 'FAIL'}] num_classes=1 reproduces the exact "
          f"loss=0.0/acc=100.0 signature (confirms root cause, not a training-loop bug)")
    print(f"[{'PASS' if guard_fires else 'FAIL'}] build_dataloaders() rejects a single-identity "
          f"dataset instead of silently training on it")
    print(f"[{'PASS' if multiclass_sound else 'FAIL'}] genuine multi-class data shows loss falling "
          f"from a nonzero start and accuracy rising gradually (not stuck at 100% from epoch 1)")

    all_pass = degenerate_reproduced and guard_fires and multiclass_sound
    print()
    if all_pass:
        print("OVERALL: PASS -- Stage 2 training loop is sound; the earlier checkpoint's")
        print("perfect numbers were a degenerate 1-class dataset, now rejected up front.")
    else:
        print("OVERALL: FAIL -- investigate before trusting any Stage 2 checkpoint.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())