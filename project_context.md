# Cancelable MinusFace — Full Project Context

Use this document to onboard a new LLM instance to this project. It contains everything needed to continue work without losing context.

---

## 1. What This Project Is

A **capstone project** implementing a privacy-preserving, cancelable face recognition system. It extends the **MinusFace paper** with two novel contributions:

1. **Wavelet encoder** replacing the paper's DCT transform (using differentiable `ptwt` Haar wavelets)
2. **Cancelable biometric template** via key-seeded Gaussian random projection on the residue

The system is implemented as a Google Colab notebook running on a T4 GPU.

---

## 2. Core Concept: How It Works

### The MinusFace idea
Instead of storing a face embedding directly, the system stores only what a generator *fails* to reconstruct — the "minus" (residue).

```
Face X  →  e(X) = x  →  g(x) = x'  →  r = x - x'  →  T(r, K) = template
```

- `e(·)` — Wavelet encoder: maps face to frequency domain (21 channels × 56×56)
- `g(·)` — U-Net generator: tries to reconstruct x from x (learns to capture appearance, not identity)
- `r = x - x'` — Residue: what the generator fails to reconstruct = identity signal
- `T(r, K)` — Cancelable transform: Gaussian projection seeded by user key K → 512-dim template

### Why not just use raw pixels?
Raw pixels contain lighting, background, hair — mostly non-identity signal. The residue `r` strips all appearance away, leaving only identity-discriminative structure. It also lives in a 65,856-dim space, making inversion computationally infeasible.

### Cancelability
- **Cancel:** delete key K → template T(r,K) is permanently unusable
- **Re-enroll:** generate new key K' → T(r,K') is statistically unlinkable from T(r,K)
- **Unlinkability property:** cosine similarity between T(r,K) and T(r,K') ≈ 0, indistinguishable from impostor pairs

---

## 3. Architecture

| Component | Details |
|---|---|
| Input | 112×112×3 RGB face images |
| Encoder e(·) | Haar wavelet, 2-level decomposition → (B, 21, 56, 56) |
| Generator g(·) | U-Net with skip connections, 21→21 channels |
| Residue r | x − x', shape (B, 21, 56, 56), 65,856 values |
| Cancelable transform T(r,K) | Gaussian projection (65856→512), L2-normalised, on CPU |
| Recognizer f_p (Stage 2) | 3-layer MLP: 512→1024→512→num_classes |
| Dataset | VGGFace2 112×112 (yakhyokhuja/vggface2-112x112), 3.31M images, 8631 identities |
| Training platform | Google Colab, T4 GPU (15GB VRAM) |

---

## 4. Training: Two Stages

### Stage 1 — Joint training (15 epochs)
Train generator `g` and residue recognizer `f_r` jointly with opposing objectives:
- `L_gen = L1(x', x)` — push generator to reconstruct faithfully (α=5.0)
- `L_fr = CrossEntropy(f_r(r), labels)` — push recognizer to identify from residue (β=1.0)
- Combined: `Loss = 5·L_gen + 1·L_fr`
- Both trained with SGD (lr 5e-3 for g, 1e-2 for f_r), CosineAnnealingLR

**Goal:** gen loss < 0.05 → residue looks like noise (no visible faces)
**Known bug (now fixed):** original code called `generator(x)` twice per batch — now stored as `x_prime` and reused

### Stage 2 — Cancelable recognizer (10 epochs)
Freeze generator. Train fresh MLP `f_p` on cancelable templates T(r, K):
- Fixed key USER_KEY=12345 for all training samples
- Adam lr=1e-3, CosineAnnealingLR
- Projection happens on CPU (avoids 128MB CUDA OOM)

---

## 5. Key Implementation Decisions and Bugs Fixed

### Bug 1 — Double generator forward pass (FIXED)
```python
# OLD (wrong) — two separate forward passes
r  = x - generator(x)
lg = crit_gen(generator(x), x)   # second call, different graph

# NEW (correct) — one pass, reused
x_prime = generator(x)
r        = x - x_prime
lg       = crit_gen(x_prime, x)
```

### Bug 2 — CUDA OOM from projection matrix (FIXED)
Projection matrix P_K is (65856×512) = 128MB. Putting it on GPU causes OOM on T4.
Fix: generate P_K on CPU, do matmul on CPU, move only the (B, 512) result to GPU.
```python
def transform(self, r, key):
    P      = self._P(key).to(r.dtype)          # CPU
    r_flat = r.detach().cpu().reshape(r.shape[0], -1)  # CPU
    tmpl   = r_flat @ P                         # CPU matmul
    tmpl   = F.normalize(tmpl, p=2, dim=1)
    return tmpl.to(r.device)                    # only result moves to GPU
```

### Bug 3 — Non-invertibility cell hardcoded N=50 vs batch size 32 (FIXED)
```python
# OLD — crashes if loader returns fewer than N images
test_imgs = next(iter(val_loader))[0][:50]
r_est = r_est_flat.reshape(50, 21, 56, 56)    # shape mismatch

# NEW — use actual count
test_imgs = next(iter(val_loader))[0]
N_ACTUAL  = test_imgs.shape[0]
r_est     = r_est_flat.reshape(N_ACTUAL, 21, 56, 56)
```

### Bug 4 — lstsq giving wrong pinv shape (FIXED)
```python
# OLD — lstsq ambiguous with underdetermined systems
Pp = torch.linalg.lstsq(P.T, torch.eye(512)).solution

# NEW — pinv directly, shape guaranteed (512, 65856)
Pp = torch.linalg.pinv(P)   # recovery: tmpl @ Pp = (B,512)@(512,65856) = (B,65856)
```

### Dataset change: LFW → VGGFace2
LFW only has ~1140 identities with ≥5 images — too few classes, too few samples per class.
VGGFace2 has 8631 identities, 3.31M images, avg ~380 per identity.
No `transforms.Resize` needed — already 112×112.

---

## 6. Known Current Issue

**Faces visible in residue → NON-INVERTIBILITY: REVIEW**

The residue `r` decoded back to spatial domain should look like noise (no recognisable faces). If faces are visible, the generator has not converged — it's outputting near-zero so `r ≈ x - 0 = x`.

Root causes in order of likelihood:
1. Stage 1 gen loss not below 0.05 yet — needs more epochs
2. Double forward pass bug (now fixed) was preventing coherent gradients
3. LFW (old dataset) too small — VGGFace2 (new) should help significantly

Check: after Stage 1, print `s1_gloss[-1]`. If > 0.05, run more epochs before Stage 2.

---

## 7. Evaluation Experiments

### Cancelability experiment (Block 9)
Compares three cosine similarity distributions:
- **Genuine same-key:** same person, key A → key A (should be high, ~0.5+)
- **Cross-key:** same person, key A → key B (should be ≈ impostor)
- **Impostor:** different people, key A (baseline)

Pass condition: `|cross-key mean − impostor mean| < 0.05`
ROC: same-key AUC should be high (usability), cross-key AUC should be ≈ 0.5 (unlinkability)

### Non-invertibility experiment (Block 10)
Adversary mounts pseudo-inverse attack: given T(r,K) and K, compute `r_est = tmpl @ pinv(P_K)`.
Pass condition: `recovery_sim.mean() < 0.3`
Expected: fails because 65856-dim → 512-dim projection has a 65344-dim null space.

### Cancellation demo (Block 11)
Step-by-step: enroll K → auth K → cancel K → re-enroll K' → adversary attack → legitimate auth K'.
Key metric: `|T(r,K) · T(r,K')| ≈ 0` (unlinkable)

---

## 8. Full Notebook Code

### Cell 1 — Install
```python
!pip install ptwt --quiet

import torch
print(f"PyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 3 — Imports
```python
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pywt, ptwt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

### Cell 5 — WaveletMapper
```python
class WaveletMapper(nn.Module):
    """X (B,3,112,112) <-> x (B,21,56,56) via differentiable Haar wavelet."""
    def __init__(self, wavelet='haar', levels=2):
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        self.levels  = levels

    def encode(self, x):
        coeffs = ptwt.wavedec2(x, self.wavelet, level=self.levels, mode='reflect')
        th, tw = coeffs[-1][0].shape[-2:]
        out = []
        for i, c in enumerate(coeffs):
            if i == 0:
                out.append(F.interpolate(c, (th, tw), mode='bilinear', align_corners=False))
            else:
                for s in c:
                    if s.shape[-2:] != (th, tw):
                        s = F.interpolate(s, (th, tw), mode='bilinear', align_corners=False)
                    out.append(s)
        return torch.cat(out, 1)

    def decode(self, x):
        b, c, h, w = x.shape
        ll_sz = h // (2 ** (self.levels - 1))
        ll    = F.interpolate(x[:, :3], (ll_sz, ll_sz), mode='bilinear', align_corners=False)
        coeffs, ptr = [ll], 3
        for i in range(self.levels, 0, -1):
            sz = h // (2 ** (i - 1))
            lh = F.interpolate(x[:, ptr:ptr+3],   (sz, sz), mode='bilinear', align_corners=False)
            hl = F.interpolate(x[:, ptr+3:ptr+6], (sz, sz), mode='bilinear', align_corners=False)
            hh = F.interpolate(x[:, ptr+6:ptr+9], (sz, sz), mode='bilinear', align_corners=False)
            coeffs.append((lh, hl, hh)); ptr += 9
        return ptwt.waverec2(coeffs, self.wavelet)

mapper = WaveletMapper().to(device)
_t = torch.randn(2, 3, 112, 112, device=device)
assert mapper.decode(mapper.encode(_t)).shape == _t.shape
print("WaveletMapper OK  encode->", mapper.encode(_t).shape)
```

### Cell 7 — UNetGenerator
```python
class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.b(x)

class UNetGenerator(nn.Module):
    def __init__(self, ch=21):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(ch,64), ConvBlock(64,128), ConvBlock(128,256)
        self.pool = nn.MaxPool2d(2)
        self.bn   = ConvBlock(256, 512)
        self.u3, self.d3 = nn.ConvTranspose2d(512,256,2,2), ConvBlock(512,256)
        self.u2, self.d2 = nn.ConvTranspose2d(256,128,2,2), ConvBlock(256,128)
        self.u1, self.d1 = nn.ConvTranspose2d(128, 64,2,2), ConvBlock(128, 64)
        self.out = nn.Conv2d(64, ch, 1)

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2))
        b=self.bn(self.pool(e3))
        d3=self.d3(torch.cat([F.interpolate(self.u3(b),  e3.shape[2:]), e3], 1))
        d2=self.d2(torch.cat([F.interpolate(self.u2(d3), e2.shape[2:]), e2], 1))
        d1=self.d1(torch.cat([F.interpolate(self.u1(d2), e1.shape[2:]), e1], 1))
        return self.out(d1)

generator = UNetGenerator().to(device)
print(f"UNetGenerator OK  params: {sum(p.numel() for p in generator.parameters()):,}")
```

### Cell 9 — CancelableTransform
```python
class CancelableTransform:
    """
    T(r, K) = L2_norm( flatten(r) @ P_K )
    P_K: (65856, 512) Gaussian matrix seeded by key K, lives on CPU only.
    Cancelability: delete K -> template inaccessible.
    Unlinkability: T(r,K) and T(r,K') are statistically independent.
    Non-invertibility: 512 << 65856, huge null-space, pinv attack fails.
    """
    def __init__(self, r_ch=21, r_hw=56, proj_dim=512):
        self.in_dim  = r_ch * r_hw * r_hw   # 65856
        self.out_dim = proj_dim
        self._cache  = {}                    # key -> CPU tensor

    def _P(self, key: int) -> torch.Tensor:
        if key not in self._cache:
            rng = torch.Generator()
            rng.manual_seed(key)
            P = torch.randn(self.in_dim, self.out_dim, generator=rng) / (self.in_dim ** 0.5)
            self._cache[key] = P             # stays on CPU
        return self._cache[key]

    @torch.no_grad()
    def transform(self, r: torch.Tensor, key: int) -> torch.Tensor:
        P      = self._P(key).to(r.dtype)
        r_flat = r.detach().cpu().reshape(r.shape[0], -1)
        tmpl   = r_flat @ P
        tmpl   = F.normalize(tmpl, p=2, dim=1)
        return tmpl.to(r.device)

    def similarity(self, a, b):
        return (a * b).sum(dim=1)

ct = CancelableTransform()
```

### Cell 11 — Dataset download (VGGFace2)
```python
import kagglehub, os

path = kagglehub.dataset_download("yakhyokhuja/vggface2-112x112")
print(f"Downloaded to: {path}")

image_dir = None
for root, dirs, files in os.walk(path):
    id_dirs = [d for d in dirs if d.startswith("n")]
    if len(id_dirs) > 10:
        image_dir = root
        break
if image_dir is None:
    image_dir = path

print(f"Image root   : {image_dir}")
print(f"Sample ids   : {sorted(os.listdir(image_dir))[:5]}")
```

### Cell 12 — DataLoader
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

# Already 112x112 — no Resize needed
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])

full_ds = datasets.ImageFolder(root=image_dir, transform=tfm)

counts      = Counter(lbl for _, lbl in full_ds.samples)
valid_lbls  = {lbl for lbl, n in counts.items() if n >= 10}
valid_idx   = [i for i, (_, lbl) in enumerate(full_ds.samples) if lbl in valid_lbls]
label_list  = sorted(valid_lbls)
remap       = {old: new for new, old in enumerate(label_list)}
num_classes = len(label_list)

class RemapDataset(torch.utils.data.Dataset):
    def __init__(self, ds, idx, remap):
        self.ds, self.idx, self.remap = ds, idx, remap
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, lbl = self.ds[self.idx[i]]
        return img, self.remap[lbl]

remapped = RemapDataset(full_ds, valid_idx, remap)
n_tr = int(0.8 * len(remapped))
train_set, val_set = torch.utils.data.random_split(
    remapped, [n_tr, len(remapped) - n_tr],
    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

print(f"Identities       : {num_classes:,}")
print(f"Total images     : {len(remapped):,}")
print(f"Train / Val      : {len(train_set):,} / {len(val_set):,}")
print(f"Avg per identity : {len(remapped)//num_classes}")
```

### Cell 13 — Stage 1 Training
```python
generator  = UNetGenerator().to(device)
recognizer = models.resnet18(num_classes=num_classes).to(device)
recognizer.conv1 = nn.Conv2d(21, 64, 7, stride=2, padding=3, bias=False).to(device)

ALPHA, BETA = 5.0, 1.0
S1_EPOCHS   = 15

opt1 = optim.SGD([
    {"params": generator.parameters(),  "lr": 5e-3},
    {"params": recognizer.parameters(), "lr": 1e-2},
], momentum=0.9, weight_decay=1e-4)
sch1     = optim.lr_scheduler.CosineAnnealingLR(opt1, S1_EPOCHS)
crit_gen = nn.L1Loss()
crit_fr  = nn.CrossEntropyLoss()
s1_gloss, s1_floss, s1_acc = [], [], []

for ep in range(S1_EPOCHS):
    generator.train(); recognizer.train()
    rg = rf = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt1.zero_grad()
        x       = mapper.encode(imgs)
        x_prime = generator(x)           # ONE forward pass, reused below
        r       = x - x_prime
        lg      = crit_gen(x_prime, x)   # no second generator(x) call
        lf      = crit_fr(recognizer(r), lbls)
        (ALPHA * lg + BETA * lf).backward()
        opt1.step()
        rg += lg.item(); rf += lf.item()

    generator.eval(); recognizer.eval()
    ok = tot = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            x = mapper.encode(imgs)
            r = x - generator(x)
            ok  += recognizer(r).argmax(1).eq(lbls).sum().item()
            tot += lbls.size(0)
    acc = 100. * ok / tot
    n   = len(train_loader)
    s1_gloss.append(rg/n); s1_floss.append(rf/n); s1_acc.append(acc)
    print(f"Epoch {ep+1} | GenLoss {rg/n:.4f} | FRLoss {rf/n:.4f} | ValAcc {acc:.2f}%")
    sch1.step()
    torch.cuda.empty_cache()

print(f"Stage 1 peak val acc : {max(s1_acc):.2f}%")
print(f"Final gen loss       : {s1_gloss[-1]:.4f}  (target: < 0.05)")
```

### Cell 16 — Stage 2 Training
```python
generator.eval()
for p in generator.parameters(): p.requires_grad_(False)

PROJ_DIM  = 512
USER_KEY  = 12345
S2_EPOCHS = 10

fp = nn.Sequential(
    nn.Linear(PROJ_DIM, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(1024, 512),      nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, num_classes),
).to(device)

opt2  = optim.Adam(fp.parameters(), lr=1e-3, weight_decay=1e-4)
sch2  = optim.lr_scheduler.CosineAnnealingLR(opt2, S2_EPOCHS)
crit2 = nn.CrossEntropyLoss()
s2_loss, s2_acc = [], []

for ep in range(S2_EPOCHS):
    fp.train()
    rl = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt2.zero_grad()
        with torch.no_grad():
            x    = mapper.encode(imgs)
            r    = x - generator(x)
        tmpl = ct.transform(r, key=USER_KEY)   # CPU projection, GPU result
        loss = crit2(fp(tmpl), lbls)
        loss.backward()
        opt2.step()
        rl += loss.item()

    fp.eval()
    ok = tot = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            x    = mapper.encode(imgs)
            r    = x - generator(x)
            tmpl = ct.transform(r, key=USER_KEY)
            ok  += fp(tmpl).argmax(1).eq(lbls).sum().item()
            tot += lbls.size(0)
    acc = 100.*ok/tot
    n   = len(train_loader)
    s2_loss.append(rl/n); s2_acc.append(acc)
    print(f"Epoch {ep+1} | Loss {rl/n:.4f} | ValAcc {acc:.2f}%")
    sch2.step()
    torch.cuda.empty_cache()
```

---

## 9. Errors Encountered and Fixed

| Error | Cause | Fix |
|---|---|---|
| `OutOfMemoryError: Tried to allocate 130 MiB` | QR decomposition of 65856×512 matrix on GPU | Keep projection matrix on CPU entirely |
| `RuntimeError: shape '[50, 21, 56, 56]' invalid for size 2107392` | Hardcoded N=50 but loader returns 32 (batch_size=32) | Use `N_ACTUAL = test_imgs.shape[0]` |
| `lstsq` returning wrong shape for pinv | Underdetermined system truncation | Replace with `torch.linalg.pinv(P)` directly |
| Faces visible in residue / NON-INVERTIBILITY: REVIEW | Generator not converged, double forward pass bug | Fix double call, increase epochs to 15, switch to VGGFace2 |

---

## 10. What Is Still Pending / Could Be Improved

- **Residue quality check:** After Stage 1, visualise decoded `r` — should look like noise, no faces. If faces visible, run more epochs.
- **Threshold tuning:** The cancelability experiment uses a fixed threshold of 0.05 for unlinkability. Could be tuned per dataset.
- **Proper verification protocol:** Currently using classification accuracy. A proper biometric system should report TAR@FAR (True Accept Rate at a given False Accept Rate), not just top-1 accuracy.
- **Multi-user key experiment:** Stage 2 trains with a single global key. A stronger evaluation would assign each identity a different key and verify recognition still works.
- **Save/load checkpoints:** No checkpoint saving currently — if Colab disconnects, training restarts from scratch.

---

## 11. Paper Reference

**MinusFace: Privacy-Preserving Face Recognition via Subtractive Biometrics**
The system's core idea (encode → regenerate → subtract → residue) comes from this paper.
Your novel contributions: (1) Haar wavelet encoder instead of DCT, (2) cancelable Gaussian projection transform T(r,K).
