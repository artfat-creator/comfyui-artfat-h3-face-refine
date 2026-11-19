# SPDX-License-Identifier: GPL-3.0-or-later
"""Artfat Face Tone Match: per-frame skin-tone correction of refined face crops toward the original crops.

A second H3 pass on a face crop comes back with the skin a little darker / cooler than the shot around it (measured on
18 Nov: L -3.4, b -3.7 in Lab). The stitch's own colour match works on mean/std of the whole paste mask (hair, brows,
eyes included), so the skin itself stays off. This node compares SKIN ONLY: inside the face mask it keeps the pixels
between two brightness percentiles (drops brows, eyes, lip lines, shadows, teeth and speculars), takes the per-channel
median in the original and in the refined crop, turns the ratio into an RGB gain, smooths the gain over time and applies
it to the refined crop. Detail is untouched (one gain per channel per frame). Put it right before the stitch.
"""
import torch
import torch.nn.functional as F


def _skin_median(img, mask, lo_pct, hi_pct):
    """img [H,W,3] 0..1, mask [H,W] bool -> per-channel median of the skin pixels, or None."""
    px = img[mask]
    if px.shape[0] < 64:
        return None
    lum = px.mean(dim=1)
    lo = torch.quantile(lum, lo_pct / 100.0)
    hi = torch.quantile(lum, hi_pct / 100.0)
    keep = (lum >= lo) & (lum <= hi)
    if int(keep.sum()) < 32:
        return None
    return px[keep].median(dim=0).values


class ArtfatFaceToneMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_crops": ("IMAGE", {"tooltip": "Face crops of the ORIGINAL video (e.g. H3 Face Track Crop 'crops')."}),
                "refined_crops": ("IMAGE", {"tooltip": "The refined crops, same size and frame count, right before the stitch."}),
                "masks": ("MASK", {"tooltip": "Per-frame face masks in crop (canvas) space, e.g. SAM3 'face'. Only skin inside them is measured."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                                       "tooltip": "1.0 = full correction toward the original skin tone, 0 = off."}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2,
                                          "tooltip": "Moving average of the gain over this many frames, so the tone cannot flicker."}),
                "low_percentile": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 95.0, "step": 1.0,
                                             "tooltip": "Darker pixels than this brightness percentile are ignored (brows, eyes, lip lines, shadows)."}),
                "high_percentile": ("FLOAT", {"default": 90.0, "min": 5.0, "max": 100.0, "step": 1.0,
                                              "tooltip": "Brighter pixels than this percentile are ignored (teeth, speculars)."}),
                "mask_erode": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1,
                                       "tooltip": "Shrink the mask by this many pixels before measuring, so hair and background at the edge stay out."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("refined_crops", "report")
    FUNCTION = "match"
    CATEGORY = "Artfat/H3"

    def match(self, original_crops, refined_crops, masks, strength, smooth_frames, low_percentile, high_percentile, mask_erode):
        n = min(original_crops.shape[0], refined_crops.shape[0])
        if refined_crops.shape[1:3] != original_crops.shape[1:3]:
            raise ValueError(f"crop sizes differ: original {tuple(original_crops.shape[1:3])}, refined {tuple(refined_crops.shape[1:3])}")
        h, w = refined_crops.shape[1:3]
        m = masks
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[1:] != (h, w):
            m = F.interpolate(m.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        m = m.float()
        if mask_erode > 0:
            k = 2 * mask_erode + 1
            m = -F.max_pool2d(-m.unsqueeze(1), kernel_size=k, stride=1, padding=mask_erode).squeeze(1)
        gains, measured = [], 0
        for i in range(n):
            mi = m[min(i, m.shape[0] - 1)] > 0.5
            a = _skin_median(original_crops[i, ..., :3].float(), mi, low_percentile, high_percentile)
            b = _skin_median(refined_crops[i, ..., :3].float(), mi, low_percentile, high_percentile)
            if a is None or b is None:
                gains.append(None)
                continue
            gains.append(a / b.clamp_min(1e-3))
            measured += 1
        if measured == 0:
            return (refined_crops, "Face Tone Match: no skin measured in any frame, passed through unchanged")
        # frames without a measurement take the nearest measured gain
        idx = [i for i, g in enumerate(gains) if g is not None]
        g = torch.stack([gains[min(idx, key=lambda j: abs(j - i))] for i in range(n)])          # [n,3]
        if smooth_frames > 1:
            pad = smooth_frames // 2
            gp = F.pad(g.t().unsqueeze(0), (pad, pad), mode="replicate")
            g = F.avg_pool1d(gp, kernel_size=smooth_frames, stride=1).squeeze(0).t()[:n]
        g = 1.0 + (g - 1.0) * float(strength)
        out = refined_crops.clone()
        out[:n, ..., :3] = (refined_crops[:n, ..., :3].float() * g.view(n, 1, 1, 3)).clamp(0, 1).to(refined_crops.dtype)
        mg = g.mean(dim=0).tolist()
        report = (f"Face Tone Match: measured {measured}/{n} frames, mean RGB gain "
                  f"R {mg[0]:.3f} G {mg[1]:.3f} B {mg[2]:.3f}, smooth {smooth_frames}, strength {strength}")
        print("[llm-prompter] " + report)
        return (out, report)
