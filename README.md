# Cancelable MinusFace

Privacy-preserving, cancelable face recognition system extending the MinusFace paper with:
1. **Haar Wavelet Encoder** — differentiable 2-level decomposition via `ptwt`
2. **Cancelable Biometric Templates** — key-seeded Gaussian projection T(r, K)

## Quick Start (Google Colab T4)

Open `cancelable_minusface_colab.ipynb` (self-contained) or `notebooks/cancelable_minusface_v2.ipynb` (imports modules).

**Manual steps:**
1. Approve Google Drive mount popup
2. Run `wandb.login()` once per session if not authenticated

The notebook runs top-to-bottom with no other manual steps.

## Pipeline

```
Face X (112x112x3)
  -> e(X) = x          WaveletMapper.encode()    (B, 21, 56, 56)
  -> g(x) = x'         UNetGenerator             (B, 21, 56, 56)
  -> r = x - x'        Residue: identity signal  (B, 65856)
  -> T(r, K)           CancelableTransform       (B, 512) L2-normalized
  -> f_p(T(r,K))       MLP classifier            class scores
```

## Training

| Stage | What trains | Loss | Epochs |
|-------|------------|------|--------|
| 1 | Generator + ResNet-18 residue recognizer jointly | 5·L1(x',x) + CE(f_r(r), y) | 15 |
| 2 | MLP f_p on frozen cancelable templates | CE(f_p(T(r,K)), y) | 8 |

**Estimated training time on T4 with all upgrades** (AMP + torch.compile + stratified subset):

| Configuration | Stage 1 | Stage 2 | Total |
|--------------|---------|---------|-------|
| Full 3.31M samples (original) | ~9 hrs | ~2.5 hrs | ~11.5 hrs |
| Stratified 100/identity (~863K) | ~2.4 hrs | ~0.7 hrs | ~3.1 hrs |

The stratified cap reduces wall-clock time ~4× with no reduction in identity diversity.

## Key Design Decisions

- **Projection on CPU**: P_K is 65856×512 = 128 MB. Placing it on GPU causes OOM on T4. Only the (B, 512) result moves to GPU.
- **Single forward pass**: `x_prime = generator(x)` is computed once per batch and reused for both the residue and the generator loss.
- **VGGFace2 over LFW**: 8631 identities vs ~1140, enabling meaningful identity discrimination.
- **Convergence guard**: Stage 2 is blocked by `assert gen_loss[-1] < 0.05` — if the generator has not converged, the residue still contains visible face structure and non-invertibility will fail.

## Security & Threat Model

### Threat Model

- **Server is honest-but-curious**: stores templates and may attempt passive inference, but does not actively collude with attackers
- **Raw face images are never stored or transmitted**: only T(r, K) reaches the server
- **Client-side key generation and template computation**: the projection happens before any data leaves the device
- **Key compromise scenario**: attacker has T(r, K) and K → pseudo-inverse attack fails (recovery_sim < 0.30) due to 65,344-dim null space of the projection
- **Template compromise without key**: useless — no key means no projection matrix means no comparison possible

### Security Properties (empirical, not cryptographic)

| Property | Mechanism | Verified by |
|----------|-----------|-------------|
| Non-invertibility | 65856→512 projection null space | Experiment 2 |
| Unlinkability | Different keys → independent T(r,K) | Experiment 1 cross-key AUC ≈ 0.5 |
| Cancelability | Delete key → template inaccessible | Experiment 3 |
| No raw storage | Only T(r,K) stored, never X or r | Architecture design |

## Security Properties

| Property | How | Pass criterion |
|----------|-----|----------------|
| Cancelability | Delete key K; issue K' | cross-key AUC ≈ 0.5 |
| Unlinkability | T(r,K) · T(r,K') ≈ 0 | \|cross-key mean − impostor mean\| < 0.05 |
| Non-invertibility | 65856→512 projection, 65344-dim null space | recovery_sim < 0.3 |

## Evaluation Benchmarks

| Benchmark | Protocol | Metrics reported |
|-----------|----------|-----------------|
| LFW | Standard 6,000-pair protocol (3,000 genuine + 3,000 impostor) from `pairs.txt` | AUC, TAR@FAR=0.1%, TAR@FAR=1% |
| AgeDB-30 | Balanced pairs with age gap ≤ 30 yr, filename-parsed identity | AUC, TAR@FAR=1% |
| Internal — Cancelability | 500 same-key and cross-key pairs from val set | Cross-key AUC ≈ 0.5 (unlinkability) |
| Internal — Non-invertibility | Pseudo-inverse attack on val batch | Recovery sim < 0.30 |

Implementations: `eval/lfw_verification.py`, `eval/agedb_verification.py`

LFW and AgeDB eval require trained models from Stages 1 and 2. Both blocks are wrapped in `try/except` in the notebook and skip gracefully if download fails.

## File Tree

```
Capstone/
├── cancelable_minusface_colab.ipynb   # Main self-contained Colab notebook (Phase 0+1)
├── notebooks/
│   └── cancelable_minusface_v2.ipynb  # Modular notebook (imports from packages)
├── models/
│   ├── __init__.py
│   ├── wavelet_mapper.py              # WaveletMapper: Haar encode/decode
│   ├── unet_generator.py             # UNetGenerator + ConvBlock
│   └── cancelable_transform.py       # CancelableTransform: T(r,K)
├── training/
│   ├── __init__.py
│   ├── stage1_train.py               # run_stage1(): joint gen + recognizer
│   └── stage2_train.py               # run_stage2(), build_mlp(): MLP on templates
├── data/
│   ├── __init__.py
│   └── dataloader.py                 # download_vggface2(), build_dataloaders()
├── eval/
│   ├── __init__.py
│   ├── cancelability.py              # run_cancelability() with ROC + AUC
│   ├── non_invertibility.py          # run_non_invertibility() pseudo-inverse attack
│   └── cancellation_demo.py          # run_cancellation_demo() lifecycle
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py                 # make_checkpoint_fn(), load/restore helpers
│   └── visualise.py                  # plot_pipeline/stage1/stage2/cancelability/...
└── README.md
```

## Bugs Fixed (Phase 0)

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

### Bug 2 — Double mapper.encode() in validation loop
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
# 423 identities after filtering, insufficient for meaningful training

# AFTER
path = kagglehub.dataset_download("yakhyokhuja/vggface2-112x112")
# 8631 identities, 3.31M images, avg ~380 per identity
```

### Bug 4 — lstsq pseudo-inverse (non-invertibility test)
```python
# BEFORE (wrong) — ambiguous for underdetermined systems
Pp = torch.linalg.lstsq(P.T, torch.eye(512)).solution

# AFTER (correct) — guaranteed shape (512, 65856)
Pp = torch.linalg.pinv(P)
```

### Bug 5 — Hardcoded batch size in non-invertibility test
```python
# BEFORE (crashes if batch < 50)
test_imgs = next(iter(val_loader))[0][:50]
r_est = r_est_flat.reshape(50, 21, 56, 56)

# AFTER (uses actual batch count)
test_imgs = next(iter(val_loader))[0]
N_ACTUAL  = test_imgs.shape[0]
r_est     = r_est_flat.reshape(N_ACTUAL, 21, 56, 56)
```

## What Was Added in Phase 1

- **W&B integration**: `wandb.init` with full config dict; `wandb.log` per epoch for all metrics; pipeline.png and cancelability.png logged as `wandb.Image`
- **Google Drive checkpointing**: saves after every epoch; auto-resume detects latest checkpoint on session restart
- **Convergence guard**: `assert s1_gloss[-1] < 0.05` with a clear warning message blocks Stage 2 if generator has not converged
- **ROC curves with AUC**: same-key AUC (usability) and cross-key AUC (unlinkability ≈ 0.5) in the cancelability experiment
- **All 6 output plots**: pipeline.png, stage1.png, stage2.png, cancelability.png, noninvert.png, cancellation_demo.png
- **Final summary cell**: prints all metrics in a clean table and pushes to `run.summary`
- **Batch size / workers**: bumped from 32/2 to 64/4

## Performance Upgrades (Phase 2 Round 2)

1. **Stratified dataset subset** (`imgs_per_identity=100`): caps each identity at 100 samples, retaining all 8631 identities while reducing 3.31M → ~863K total samples. ~4× faster per epoch.
2. **Mixed precision training** (`torch.cuda.amp`): `autocast()` + `GradScaler` for both stages. ~1.5–2× speedup on T4 Tensor Cores. Scaler state saved in checkpoints so resume does not cause loss spikes.
3. **torch.compile** (`mode='reduce-overhead'`): applied to generator, recognizer, and fp with try/except fallback for PyTorch < 2.0. Additional ~10–20% throughput after first-epoch JIT warmup.
4. **Persistent workers + prefetch** (`persistent_workers=True`, `prefetch_factor=2`): eliminates DataLoader worker spawn overhead between batches.
5. **tqdm progress bars**: inner per-batch bar with live loss display; outer per-epoch bar with ETA. Uses `tqdm.auto` for Colab and terminal compatibility.
6. **Training time estimate cell**: two warmup passes (second used, first discarded due to JIT tracing overhead) extrapolated to full epoch counts for Stage 1 and Stage 2.
7. **Epoch timing in logs**: wall-clock elapsed per epoch + remaining-time estimate printed after each epoch; `epoch_time_sec` logged to W&B.

## Dependencies

```bash
pip install ptwt wandb
# All other packages (torch, torchvision, sklearn, matplotlib, tqdm) pre-installed on Colab
```

## Known Limitations

- **GPU training required**: The full VGGFace2 dataset requires a GPU. CPU-only training will take many hours.
- **Wandb API key**: You must run `wandb.login()` once per session. Set `mode='disabled'` to skip logging.
- **Kaggle credentials**: `kagglehub` needs your Kaggle API key in `~/.kaggle/kaggle.json` or as environment variables.
- **15 epochs may not converge**: If `gen_loss[-1] >= 0.05` after Stage 1, increase `S1_EPOCHS` to 20-25 and re-run. The auto-resume logic will pick up where training left off.
- **Cancellation demo threshold**: The demo uses a cosine threshold of 0.2. Real systems would calibrate this threshold at a target FAR.

## Suggested Next Steps

1. **Verify gen_loss < 0.05**: After Stage 1, check that the decoded residue looks like noise (not faces). If faces are still visible, run more epochs.
2. **TAR@FAR evaluation**: Replace top-1 accuracy with a proper biometric metric (True Accept Rate at 0.1% False Accept Rate).
3. **Multi-key Stage 2**: Assign each identity a different key during Stage 2 training and verify that recognition accuracy is maintained.
4. **Larger generator**: Try a deeper U-Net (512→1024 bottleneck) or replace with a pretrained backbone for faster convergence.
5. **Threshold calibration**: Tune the authentication threshold per-FAR operating point using the ROC curves from the cancelability experiment.
6. **Paper writeup**: Sections to highlight: (1) novel wavelet encoder vs DCT, (2) cancelability property proof via Gaussian projection, (3) non-invertibility bound from null-space argument.
