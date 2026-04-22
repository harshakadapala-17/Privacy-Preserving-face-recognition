"""
Differentiable Haar wavelet encoder/decoder for the Cancelable MinusFace pipeline.

Maps face images (B, 3, 112, 112) to wavelet coefficients (B, 21, 56, 56) and back.
Uses ptwt for GPU-compatible differentiable wavelet transforms.

Channel layout (21 total):
  [0:3]   LL approximation (level-2)
  [3:6]   LH detail (level-2)
  [6:9]   HL detail (level-2)
  [9:12]  HH detail (level-2)
  [12:15] LH detail (level-1)
  [15:18] HL detail (level-1)
  [18:21] HH detail (level-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import ptwt


class WaveletMapper(nn.Module):
    """Haar wavelet encoder/decoder.

    Performs 2-level 2D Haar wavelet decomposition, producing a 21-channel
    frequency representation at spatial size 56×56. Level-1 and level-2
    subbands are bilinearly upsampled to a common size before concatenation.
    """

    def __init__(self, wavelet: str = "haar", levels: int = 2) -> None:
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        self.levels = levels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode face images to wavelet coefficients.

        Args:
            x: Face images of shape (B, 3, 112, 112).

        Returns:
            Wavelet coefficients of shape (B, 21, 56, 56).
        """
        coeffs = ptwt.wavedec2(x, self.wavelet, level=self.levels, mode="reflect")
        th, tw = coeffs[-1][0].shape[-2:]
        out = []
        for i, c in enumerate(coeffs):
            if i == 0:
                out.append(F.interpolate(c, (th, tw), mode="bilinear", align_corners=False))
            else:
                for s in c:
                    if s.shape[-2:] != (th, tw):
                        s = F.interpolate(s, (th, tw), mode="bilinear", align_corners=False)
                    out.append(s)
        return torch.cat(out, dim=1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode wavelet coefficients back to spatial images.

        Args:
            x: Wavelet coefficients of shape (B, 21, 56, 56).

        Returns:
            Reconstructed images of approximately shape (B, 3, 112, 112).
        """
        b, c, h, w = x.shape
        ll_sz = h // (2 ** (self.levels - 1))
        ll = F.interpolate(x[:, :3], (ll_sz, ll_sz), mode="bilinear", align_corners=False)
        coeffs, ptr = [ll], 3
        for i in range(self.levels, 0, -1):
            sz = h // (2 ** (i - 1))
            lh = F.interpolate(x[:, ptr : ptr + 3], (sz, sz), mode="bilinear", align_corners=False)
            hl = F.interpolate(x[:, ptr + 3 : ptr + 6], (sz, sz), mode="bilinear", align_corners=False)
            hh = F.interpolate(x[:, ptr + 6 : ptr + 9], (sz, sz), mode="bilinear", align_corners=False)
            coeffs.append((lh, hl, hh))
            ptr += 9
        return ptwt.waverec2(coeffs, self.wavelet)
