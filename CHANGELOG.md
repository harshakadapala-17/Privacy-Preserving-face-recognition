# Changelog

Development history for Cancelable MinusFace. The README describes the system as it
stands today; this file documents how it got here — bugs found and fixed, and
capability added in each phase.

## Phase 2 — Performance Upgrades

1. **Stratified dataset subset** (`imgs_per_identity=100`): caps each identity at 100
   samples, retaining all 8,631 identities while reducing 3.31M → ~863K total samples.
   ~4× faster per epoch with no loss of class diversity.
2. **Mixed precision training** (`torch.cuda.amp`): `autocast()` + `GradScaler` for both
   stages. ~1.5–2× speedup on T4 Tensor Cores. Scaler state is saved in checkpoints so
   resuming does not cause loss spikes.
3. **`torch.compile`** (`mode='reduce-overhead'`): applied to the generator, recognizer,
   and MLP classifier, with a `try/except` fallback for PyTorch < 2.0. ~10–20%
   additional throughput after the first-epoch JIT warmup.
4. **Persistent workers + prefetch** (`persistent_workers=True`, `prefetch_factor=2`):
   removes DataLoader worker spawn overhead between batches.
5. **tqdm progress bars**: inner per-batch bar with live loss display; outer per-epoch
   bar with ETA.
6. **Training time estimate cell**: two warmup passes (second one used, first discarded
   for JIT tracing overhead) extrapolated to full epoch counts for both stages.
7. **Epoch timing in logs**: wall-clock elapsed and remaining-time estimate printed
   after every epoch; `epoch_time_sec` logged to W&B.

## Phase 1 — Experiment Tracking & Checkpointing

- **W&B integration**: `wandb.init` with full config dict; `wandb.log` per epoch for
  all metrics; pipeline/cancelability plots logged as `wandb.Image`.
- **Google Drive checkpointing**: saves after every epoch; auto-resume detects the
  latest checkpoint on session restart.
- **Convergence guard**: `assert s1_gloss[-1] < 0.05`, with a clear warning, blocks
  Stage 2 if the generator has not converged.
- **ROC curves with AUC**: same-key AUC (usability) and cross-key AUC (unlinkability,
  target ≈ 0.5) added to the cancelability experiment.
- **Six output plots**: `pipeline.png`, `stage1.png`, `stage2.png`, `cancelability.png`,
  `noninvert.png`, `cancellation_demo.png`.
- **Final summary cell**: prints all metrics in one table and pushes to `run.summary`.
- Batch size / workers bumped from 32/2 to 64/4.

## Phase 0 — Bugs Found & Fixed

### Bug 1 — Double generator forward pass
```python
# BEFORE (wrong) — two separate forward passes, inconsistent gradients
r  = x - generator(x)
lg = crit_gen(generator(x), x)   # second call

# AFTER (correct) — one pass, reused
x_prime = generator(x)
r        = x - x_prime
lg       = crit_gen(x_prime, x)
```

### Bug 2 — Double `mapper.encode()` in the validation loop
```python
# BEFORE (wrong) — two encode calls, wasteful and inconsistent
r = mapper.encode(imgs) - generator(mapper.encode(imgs))

# AFTER (correct) — one encode
x       = mapper.encode(imgs)
x_prime = generator(x)
r       = x - x_prime
```

### Bug 3 — Dataset too small (LFW → VGGFace2)
```python
# BEFORE
path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
# 423 identities after filtering — insufficient for meaningful training

# AFTER
path = kagglehub.dataset_download("yakhyokhuja/vggface2-112x112")
# 8,631 identities, 3.31M images, ~380 avg per identity
```

### Bug 4 — `lstsq` pseudo-inverse in the non-invertibility test
```python
# BEFORE (wrong) — ambiguous for underdetermined systems
Pp = torch.linalg.lstsq(P.T, torch.eye(512)).solution

# AFTER (correct) — guaranteed shape (512, 65856)
Pp = torch.linalg.pinv(P)
```

### Bug 5 — Hardcoded batch size in the non-invertibility test
```python
# BEFORE (crashes if batch size < 50)
test_imgs = next(iter(val_loader))[0][:50]
r_est = r_est_flat.reshape(50, 21, 56, 56)

# AFTER (uses the actual batch count)
test_imgs = next(iter(val_loader))[0]
N_ACTUAL  = test_imgs.shape[0]
r_est     = r_est_flat.reshape(N_ACTUAL, 21, 56, 56)
```

### Bug 6 — Non-invertibility test compared re-projected templates, not raw residues

`eval/non_invertibility.py`'s recovery-similarity metric compared
`ct.transform(r_true)` to `ct.transform(r_est)` instead of comparing `r_true` to
`r_est` directly. By the Moore-Penrose identity `P·P⁺·P = P`, the adversary's
recovery `r_est = tmpl @ P⁺` satisfies `r_est @ P == r_true @ P` **exactly**, for
*any* `r_true` — so re-projecting both through `ct.transform` and comparing was
tautologically 1.0, always, regardless of model quality or generator convergence.
This alone explains the earlier `recovery_sim = 1.0000` "complete fail" result; it
had nothing to do with the generator.

```python
# BEFORE (wrong) — always returns ~1.0 by construction, tests nothing
tmpl_rec = ct.transform(r_est.to(device), key=key).cpu()
rec_sim = (tmpl_cpu * tmpl_rec).sum(dim=1).numpy()

# AFTER (correct) — compares raw residues directly, in the original space
r_true_n = F.normalize(r_t.cpu().float().reshape(n_actual, -1), p=2, dim=1)
r_est_n  = F.normalize(r_est_flat, p=2, dim=1)
rec_sim  = (r_true_n * r_est_n).sum(dim=1).numpy()
```

Verified via `eval/sanity_transform_test.py` (added alongside this fix): with
synthetic pure-noise residues standing in for a converged generator's output,
`recovery_sim` now reports ~0.09 (correctly low, not a tautological 1.0).
`eval/cancelability.py` was audited for the same pattern and does **not** have this
bug — it compares templates directly, with no reconstruction step, which is the
correct convention for measuring unlinkability.

### Bug 7 — `_find_image_root()` silently picked the wrong dataset directory

Downstream of Bug 6, but a distinct code bug: `_find_image_root()` only recognized
identity folders whose names start with `"n"` (classic VGGFace2 convention, e.g.
`n000002`). The `yakhyokhuja/vggface2-112x112` Kaggle repackaging nests the real
identity folders one level deeper under a `vggface2_112x112/` wrapper directory
with different naming, so the `"n"`-prefix check never matched anything in the
whole tree and silently fell back to the wrong (too-shallow) root.
`ImageFolder` then saw that one wrapper folder as a single class, and the
`imgs_per_identity=100` stratified cap collapsed it to 100 images total —
producing `batches/epoch: 2` and `num_classes: 1` instead of an error.

That, in turn, produced a Stage 2 checkpoint with `losses = [0.0]*8` and
`val_accs = [100.0]*8` from epoch 1, never changing. This is not a training-loop
bug — it's the deterministic mathematical signature of `num_classes == 1`:
softmax over a single logit is always exactly `1.0` regardless of its value, so
`CrossEntropyLoss = -log(1.0) = 0.0` exactly (and its gradient is exactly zero
too), and `argmax` of a one-output head always matches the one label that can
exist. `eval/sanity_stage2_test.py` reproduces this signature on synthetic data
to document it, confirms `build_dataloaders()`'s new guard below rejects it, and
separately confirms the same training-loop shape (`build_mlp` → `CrossEntropyLoss`
→ `Adam`) behaves normally (loss falls, accuracy rises gradually) on genuine
multi-class synthetic data.

```python
# BEFORE (wrong) — only matches VGGFace2's classic "n000002" naming
def _find_image_root(base_path: str) -> str:
    for root, dirs, _files in os.walk(base_path):
        id_dirs = [d for d in dirs if d.startswith("n")]
        if len(id_dirs) > 10:
            return root
    return base_path

# AFTER (correct) — naming-convention-agnostic: pick the directory with the
# most immediate subdirectories, wherever it is in the tree
def _find_image_root(base_path: str) -> str:
    best_root, best_count = base_path, 0
    for root, dirs, _files in os.walk(base_path):
        if len(dirs) > best_count:
            best_root, best_count = root, len(dirs)
    return best_root
```

Also added a guard in `build_dataloaders()`: it now raises `ValueError` if
`num_classes < 2`, instead of silently building a degenerate one-class dataset
that trains "successfully" to a meaningless 0.0 loss / 100% accuracy.
