# Cancelable MinusFace

A privacy-preserving, cancelable face recognition system extending the **MinusFace**
paper (subtractive biometrics) with two original contributions:

1. **Haar wavelet encoder** — a differentiable, invertible frequency-domain
   transform (via `ptwt`) replacing the paper's DCT encoder.
2. **Cancelable biometric templates** — a key-seeded Gaussian random projection
   `T(r, K)` that makes the stored template revocable, unlinkable across
   re-enrollments, and computationally infeasible to invert.

No raw face image, and no raw residue, is ever stored or transmitted — only the
512-dimensional projected template `T(r, K)` leaves the pipeline.

## Table of Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Running on Google Colab](#running-on-google-colab-primary-workflow)
  - [Pre-Training Sanity Check (local, no GPU)](#pre-training-sanity-check-local-no-gpu)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Security Properties](#evaluation--security-properties)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Changelog](#changelog)
- [Acknowledgments](#acknowledgments)

## How It Works

```
Face X (112×112×3)
  → e(X) = x          WaveletMapper.encode()    (B, 21, 56, 56)
  → g(x) = x'         UNetGenerator              (B, 21, 56, 56)
  → r = x − x'        Residue: identity signal   (B, 65,856)
  → T(r, K)           CancelableTransform        (B, 512), L2-normalized
  → f_p(T(r, K))      MLP classifier             class scores
```

Instead of storing a face embedding directly, the system stores only what a
generator *fails* to reconstruct — the residue `r`. The generator is trained to
capture appearance (lighting, background, pose), so what's left behind in `r` is
identity-discriminative structure. That residue is then passed through a
key-seeded random projection before it ever leaves the pipeline, giving three
properties for free:

- **Cancelability** — delete key `K` and the template `T(r, K)` is permanently
  unusable.
- **Unlinkability** — re-enroll with a new key `K'`; `T(r, K)` and `T(r, K')` are
  statistically independent, indistinguishable from an impostor pair.
- **Non-invertibility** — the 65,856 → 512 projection has a 65,344-dimensional
  null space, defeating pseudo-inverse reconstruction attacks even with the key.

## Repository Structure

```
Capstone/
├── cancelable_minusface_colab.ipynb   # Main, self-contained Colab notebook
├── notebooks/
│   └── cancelable_minusface_v2.ipynb  # Modular notebook (imports from packages below)
├── models/
│   ├── wavelet_mapper.py              # WaveletMapper: Haar encode/decode
│   ├── unet_generator.py              # UNetGenerator + ConvBlock
│   └── cancelable_transform.py        # CancelableTransform: T(r, K)
├── training/
│   ├── stage1_train.py                # run_stage1(): joint generator + recognizer
│   └── stage2_train.py                # run_stage2(), build_mlp(): MLP on templates
├── data/
│   └── dataloader.py                  # download_vggface2(), build_dataloaders()
├── eval/
│   ├── cancelability.py               # run_cancelability(): ROC + AUC
│   ├── non_invertibility.py           # run_non_invertibility(): pseudo-inverse attack
│   ├── cancellation_demo.py           # run_cancellation_demo(): full lifecycle demo
│   ├── lfw_verification.py            # LFW 6,000-pair verification benchmark
│   ├── agedb_verification.py          # AgeDB-30 verification benchmark
│   └── sanity_transform_test.py       # Pre-training sanity check (no GPU required)
├── utils/
│   ├── checkpoint.py                  # Checkpoint save/load/resume helpers
│   └── visualise.py                   # Plotting for every experiment
├── README.md
└── CHANGELOG.md                       # Full bug-fix and upgrade history
```

## Getting Started

### Prerequisites

| Requirement | Needed for |
|---|---|
| Google account | Colab runtime + Google Drive checkpoint persistence |
| Kaggle account + API key | Downloading VGGFace2 (and optionally AgeDB) via `kagglehub` |
| Weights & Biases account (optional) | Experiment tracking — set `mode='disabled'` in Block 7 to skip |
| A GitHub fork of this repo (optional) | Only if you want the notebook to clone *your* copy — otherwise point `REPO_URL` at this repo directly |

### Running on Google Colab (primary workflow)

The project is designed to run top-to-bottom in `cancelable_minusface_colab.ipynb`
on a single T4 GPU, in roughly 3 hours end-to-end.

1. **Open the notebook.** Upload `cancelable_minusface_colab.ipynb` to Colab, or
   open it directly from GitHub via *File → Open notebook → GitHub*.
2. **Select a GPU runtime.** *Runtime → Change runtime type → T4 GPU.*
3. **Set your repo URL** (Block 1). If you're working from your own fork, update
   `REPO_URL`; the notebook clones it and adds it to `sys.path` so the `models/`,
   `training/`, `data/`, `eval/`, and `utils/` packages import cleanly.
4. **Install dependencies** (Block 2) — runs automatically:
   `pip install ptwt wandb kagglehub tqdm --quiet`. Everything else (`torch`,
   `torchvision`, `scikit-learn`, `matplotlib`) is preinstalled on Colab.
5. **Authenticate W&B** (Block 4) — run `wandb.login()` once per session, or set
   the `WANDB_API_KEY` environment variable beforehand. To skip tracking
   entirely, change `mode='online'` to `mode='disabled'` in Block 7.
6. **Mount Google Drive** (Block 5) — approve the popup. Checkpoints are written
   to `/content/drive/MyDrive/minusface_checkpoints/` after every epoch, so
   training survives a disconnected Colab session.
7. **Set Kaggle credentials** before Block 8 — either export
   `KAGGLE_USERNAME` / `KAGGLE_KEY` as environment variables, or place your
   `kaggle.json` at `~/.kaggle/kaggle.json`. This is required to download the
   VGGFace2 112×112 dataset (8,631 identities, ~863K images after the
   stratified cap).
8. **Run all cells in order** (*Runtime → Run all*, or step through manually).
   Everything downstream — model init, Stage 1 training, the convergence guard,
   Stage 2 training, and all five evaluation experiments — runs without further
   input.
9. **Watch the convergence guard** (Block 14). Stage 2 is blocked with an
   assertion unless `gen_loss < 0.05`. If it fires, increase `S1_EPOCHS`
   (config in Block 6, e.g. 15 → 20–25) and re-run Block 12 — training
   auto-resumes from the last Drive checkpoint rather than starting over.
10. **Read the results** (Block 23) — a final summary table with every metric
    (accuracies, same-key/cross-key AUC, recovery similarity, LFW/AgeDB AUC)
    checked against its target, plus six saved plots
    (`pipeline.png`, `stage1.png`, `stage2.png`, `cancelability.png`,
    `noninvert.png`, `cancellation_demo.png`) and everything logged to your W&B
    run.

The LFW (Block 21) and AgeDB-30 (Block 22) benchmarks each download their own
dataset and are wrapped in `try/except` — they skip gracefully (rather than
crashing the run) if the download fails or credentials are missing.

### Pre-Training Sanity Check (local, no GPU)

Before spending GPU hours on a full Stage 1 + Stage 2 run, you can verify that
`CancelableTransform` itself is mathematically sound — independent of whether the
generator has converged — using synthetic residues on CPU:

```bash
pip install torch scikit-learn numpy
python -m eval.sanity_transform_test
```

This compares a pure-noise residue regime (what a converged generator should
produce) against a shared-bias regime (what an under-converged generator leaks)
and prints a pass/fail summary, exiting `0` on pass and `1` on fail. See
[`eval/sanity_transform_test.py`](eval/sanity_transform_test.py) for the full
rationale, and [CHANGELOG.md](CHANGELOG.md#bug-6--non-invertibility-test-compared-re-projected-templates-not-raw-residues)
for the bug it was built to catch.

## Training Pipeline

| Stage | What trains | Loss | Epochs |
|-------|------------|------|--------|
| 1 | Generator + ResNet-18 residue recognizer, jointly, with opposing objectives | `5·L1(x', x) + 1·CE(f_r(r), y)` | 15 |
| 2 | MLP `f_p` on frozen cancelable templates | `CE(f_p(T(r,K)), y)` | 8 |

**Estimated training time on a T4** (mixed precision + `torch.compile` +
stratified dataset subset):

| Configuration | Stage 1 | Stage 2 | Total |
|--------------|---------|---------|-------|
| Full 3.31M samples | ~9 hrs | ~2.5 hrs | ~11.5 hrs |
| Stratified 100 images/identity (~863K) | ~2.4 hrs | ~0.7 hrs | ~3.1 hrs |

The stratified cap keeps all 8,631 identities while cutting wall-clock time ~4×.

## Evaluation & Security Properties

### Threat Model

- **Server is honest-but-curious**: stores templates and may attempt passive
  inference, but does not actively collude with attackers.
- **Raw face images are never stored or transmitted** — only `T(r, K)` reaches
  the server.
- **Client-side key generation and template computation** — the projection
  happens before any data leaves the device.
- **Key compromise**: attacker has `T(r, K)` *and* `K` → pseudo-inverse attack
  fails (`recovery_sim < 0.30`) due to the 65,344-dimensional null space.
- **Template compromise without the key** — useless; no key means no
  projection matrix means no comparison possible.

### Security Properties (empirical, not cryptographic)

| Property | Mechanism | Pass criterion | Verified by |
|----------|-----------|-----------------|-------------|
| Cancelability | Delete key `K`; issue `K'` | cross-key AUC ≈ 0.5 | Cancellation demo |
| Unlinkability | Different keys → independent `T(r,K)` | \|cross-key mean − impostor mean\| < 0.05 | Cancelability experiment |
| Non-invertibility | 65,856 → 512 projection, 65,344-dim null space | `recovery_sim` < 0.30 | Non-invertibility experiment |

### Evaluation Benchmarks

| Benchmark | Protocol | Metrics reported |
|-----------|----------|-----------------|
| LFW | Standard 6,000-pair protocol (3,000 genuine + 3,000 impostor) from `pairs.txt` | AUC, TAR@FAR=0.1%, TAR@FAR=1% |
| AgeDB-30 | Balanced pairs with age gap ≤ 30 yr, filename-parsed identity | AUC, TAR@FAR=1% |
| Internal — Cancelability | 500 same-key and cross-key pairs from the validation set | Cross-key AUC ≈ 0.5 (unlinkability) |
| Internal — Non-invertibility | Pseudo-inverse attack on a validation batch | Recovery similarity < 0.30 |
| Pre-training sanity check | Synthetic residues, no generator/GPU/dataset needed — `python -m eval.sanity_transform_test` | Confirms `CancelableTransform` math is sound *before* spending GPU hours |

## Key Design Decisions

- **Projection on CPU**: `P_K` is 65,856×512 = 128 MB. Placing it on GPU causes
  an OOM on a T4. Only the `(B, 512)` result moves to GPU.
- **Single generator forward pass**: `x_prime = generator(x)` is computed once
  per batch and reused for both the residue and the generator loss.
- **VGGFace2 over LFW**: 8,631 identities vs. ~1,140, enabling meaningful
  identity discrimination at Stage 1 and Stage 2.
- **Convergence guard**: Stage 2 is blocked by `assert gen_loss[-1] < 0.05` — if
  the generator hasn't converged, the residue still contains visible face
  structure and non-invertibility will fail for real, model-quality reasons.

## Known Limitations

- **GPU training required** — the full VGGFace2 dataset needs a GPU; CPU-only
  training would take many hours per epoch.
- **W&B API key** — `wandb.login()` is required once per session unless you set
  `mode='disabled'`.
- **Kaggle credentials** — `kagglehub` needs your API key in
  `~/.kaggle/kaggle.json` or as environment variables.
- **15 epochs may not converge** — if `gen_loss[-1] >= 0.05` after Stage 1,
  increase `S1_EPOCHS` to 20–25 and re-run; auto-resume picks up where training
  left off.
- **Cancellation demo threshold** — uses a fixed cosine threshold of 0.2; a real
  system would calibrate this at a target FAR.

## Roadmap

1. **Verify `gen_loss < 0.05`** after Stage 1, and visually confirm the decoded
   residue looks like noise, not a face.
2. **TAR@FAR evaluation** — replace top-1 accuracy with a proper biometric
   metric (True Accept Rate at a fixed False Accept Rate).
3. **Multi-key Stage 2** — assign each identity a distinct key during Stage 2
   training and confirm recognition accuracy is maintained.
4. **Larger generator** — a deeper U-Net (512→1024 bottleneck) or a pretrained
   backbone, for faster convergence.
5. **Threshold calibration** — tune the authentication threshold per-FAR
   operating point using the cancelability experiment's ROC curves.
6. **Paper writeup** — novel wavelet encoder vs. DCT; cancelability proof via
   Gaussian projection; non-invertibility bound from the null-space argument.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full history of bugs found and fixed,
and capability added, phase by phase.

## Acknowledgments

Core idea (encode → regenerate → subtract → residue) comes from **MinusFace:
Privacy-Preserving Face Recognition via Subtractive Biometrics**. This project's
contributions on top of that paper are the Haar wavelet encoder (replacing DCT)
and the cancelable Gaussian projection transform `T(r, K)`.
