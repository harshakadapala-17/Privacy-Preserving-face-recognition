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
