#!/usr/bin/env python3
"""
micro_anim -- green-screen still -> seamless-loop micro-animation.

Turns ONE photograph that was shot against a bright chroma-green screen into a
short, seamlessly looping "living photograph" video.

Why this exists
---------------
Hosted image-to-video models *regenerate* every pixel, so identity, wardrobe,
colour and the green screen can drift. This renderer never invents a pixel: it
only *moves* the pixels that are already there.

Guarantees enforced and verified at runtime (every number below is written to
<out>.report.json, and a failed check makes the process exit non-zero):
  1. Background is bit-identical.  Every pixel the chroma key classifies as
     green screen is copied from the source untouched, in every frame
     (sampled every frame, re-checked in full on the final frame).
  2. Seamless loop.  All motion is a sum of sinusoids with integer cycle counts
     across the clip, so frame[N-1] flows into frame[0] with no jump.  The
     wrap-around step is measured against the ordinary per-frame step.
  3. Composition, pose and framing lock.  There is no camera model at all: the
     displacement field is identity outside the subject mask, so no zoom, pan,
     crop or perspective change is possible by construction.
  4. No added or removed elements.  Only the subject's own pixels are resampled.
  5. Motion never stalls.  Components carry different phases; if they shared
     one, every term would vanish together and the loop would visibly freeze
     twice.  The minimum displacement in the loop is checked against the peak.

Usage
-----
  python renderer/micro_anim.py --input photo.png --out out/loop
  python renderer/micro_anim.py --input photo.png --out out/loop \
      --zones zones.json --debug-dir out/debug --frames 120 --fps 30

Outputs   <out>.mp4 (H.264)  <out>.webp (animated preview)  <out>.report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #

Array = np.ndarray


def smoothstep(e0: float, e1: float, x: Array) -> Array:
    """Hermite ramp between e0 and e1 (vectorised, clamped)."""
    if e1 == e0:
        return (x >= e1).astype(np.float32)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _gauss_kernel(sigma: float) -> Array:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _convolve1d(arr: Array, kernel: Array, axis: int) -> Array:
    r = len(kernel) // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    src = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr)
    n = arr.shape[axis]
    for i, wgt in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + n)
        out += float(wgt) * src[tuple(sl)]
    return out


def blur_f(arr: Array, radius: float) -> Array:
    """Separable Gaussian blur of a float array (numpy only, any PIL version)."""
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    kernel = _gauss_kernel(max(radius, 0.4) / 2.0)
    out = _convolve1d(arr.astype(np.float32), kernel, 0)
    return _convolve1d(out, kernel, 1)


def upsample_2d(arr: Array, w: int, h: int) -> Array:
    """Bilinear resize of a float 2-D array to (h, w)."""
    ah, aw = arr.shape
    if (ah, aw) == (h, w):
        return arr.astype(np.float32, copy=True)
    xs = np.linspace(0.0, aw - 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, ah - 1.0, h, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, aw - 1)
    y1 = np.minimum(y0 + 1, ah - 1)
    fx = (xs - x0)[None, :]
    fy = (ys - y0)[:, None]
    src = arr.astype(np.float32)
    top = src[y0][:, x0] * (1.0 - fx) + src[y0][:, x1] * fx
    bot = src[y1][:, x0] * (1.0 - fx) + src[y1][:, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def blur_rgb(arr: Array, radius: float) -> Array:
    return np.stack([blur_f(arr[..., c], radius) for c in range(arr.shape[2])], axis=-1)


def smooth_noise(shape: Tuple[int, int], cell: float, rng: np.random.Generator,
                 blur: float = 1.5) -> Array:
    """Band-limited random field in roughly [-1, 1] with feature size ~`cell` px.

    Used to give neighbouring hair strands / fabric areas slightly different
    phases, so motion does not look like a single rigid warp.
    """
    h, w = shape
    ch, cw = max(2, int(round(h / max(cell, 1.0)))), max(2, int(round(w / max(cell, 1.0))))
    small = rng.normal(0.0, 1.0, (ch, cw)).astype(np.float32)
    big = upsample_2d(small, w, h)
    big = blur_f(big, blur)
    peak = float(np.max(np.abs(big))) or 1.0
    return np.clip(big / peak, -1.0, 1.0).astype(np.float32)


def sample_bilinear(img: Array, dx: Array, dy: Array,
                    xs: Array, ys: Array) -> Array:
    """Sample `img` at (x + dx, y + dy) with bilinear interpolation.

    `xs` / `ys` are the (already cropped) integer coordinate grids. Works for
    2-D (H, W) and 3-D (H, W, C) images.
    """
    h, w = img.shape[0], img.shape[1]
    if (xs.min() < 0.0 or ys.min() < 0.0 or xs.max() > w - 1.0
            or ys.max() > h - 1.0 or xs.shape != ys.shape
            or xs.shape != (h, w)):
        raise ValueError(
            "sample_bilinear: the coordinate grid must be relative to the image "
            f"being sampled (grid {xs.shape}, image {(h, w)}, "
            f"x in [{xs.min():.0f}, {xs.max():.0f}], y in [{ys.min():.0f}, {ys.max():.0f}]). "
            "Did you forget to subtract the crop origin?")
    sx = np.clip(xs + dx, 0.0, w - 1.0)
    sy = np.clip(ys + dy, 0.0, h - 1.0)

    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)

    # One-pixel edge pad turns the x1/y1 clamping into "just read one column
    # further", and flat indices make it a single gather per corner — much
    # faster than four fancy-indexed gathers on a stacked image.
    if img.ndim == 2:
        flat = np.pad(img, ((0, 1), (0, 1)), mode="edge").reshape(-1)
        base = y0 * (w + 1) + x0
        a, b = flat[base], flat[base + 1]
        c, d = flat[base + w + 1], flat[base + w + 2]
        top = a + (b - a) * fx
        bot = c + (d - c) * fx
        return (top + (bot - top) * fy).astype(np.float32)

    ch = img.shape[2]
    flat = np.pad(img, ((0, 1), (0, 1), (0, 0)), mode="edge").reshape(-1, ch)
    base = (y0 * (w + 1) + x0).ravel()
    a, b = flat[base], flat[base + 1]
    c, d = flat[base + w + 1], flat[base + w + 2]
    fxv = fx.ravel()[:, None]
    fyv = fy.ravel()[:, None]
    top = a + (b - a) * fxv
    bot = c + (d - c) * fxv
    out = top + (bot - top) * fyv
    return out.reshape(img.shape[0], img.shape[1], ch).astype(np.float32)


# --------------------------------------------------------------------------- #
# background / subject separation
# --------------------------------------------------------------------------- #


@dataclass
class MatteParams:
    key_low: float = 0.10     # ratio below this -> pure subject
    key_high: float = 0.22    # ratio above this -> pure background
    feather: float = 1.2      # px, softens the classification edge
    despeckle: int = 3        # median filter size on the hard mask (0 = off)
    grow: int = 24            # px, subject colours are extended outward this far


def greenness(rgb: Array) -> Array:
    """Normalised chroma-key metric.

    (G - max(R, B)) / (G + max(R, B)). Normalising matters: a studio green
    screen is rarely one flat colour (vignettes, uneven lighting, shadow
    gradients), but the *ratio* stays almost constant, whereas the raw channel
    difference does not. Skin, hair, denim and most warm fabrics score at or
    below zero, so the separation stays clean.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(r, b)
    return (g - mx) / (g + mx + 1.0)


def chroma_matte(rgb: Array, p: MatteParams) -> Array:
    """Soft subject matte in [0, 1] (1 = subject), feathered for compositing."""
    return blur_f(suppress_speckle(raw_matte(rgb, p), p.despeckle), p.feather)


def raw_matte(rgb: Array, p: MatteParams) -> Array:
    """Unblurred linear ramp from the chroma key, in [0, 1]."""
    gn = greenness(rgb)
    return np.clip((p.key_high - gn) / max(p.key_high - p.key_low, 1e-6), 0.0, 1.0)


def suppress_speckle(m: Array, k: int) -> Array:
    """Median-filter the hard mask and push the soft matte to agree with it."""
    if k is None or k < 3:
        return m
    hard = ((m > 0.5).astype(np.uint8) * 255)
    k = int(k) | 1
    hard = np.asarray(Image.fromarray(hard, "L").filter(ImageFilter.MedianFilter(k)))
    m = np.where((m > 0.5) & (hard == 0), 0.0, m)
    return np.where((m < 0.5) & (hard == 255), 1.0, m)


def matte_support(rgb: Array, p: MatteParams) -> Array:
    """Hard subject mask (bool) = the only region the renderer may write to."""
    return suppress_speckle(raw_matte(rgb, p), p.despeckle) > 0.5


def block_mean(a: Array, s: int) -> Array:
    """Mean-pool an array by an integer factor (used by the pyramid fill)."""
    if s <= 1:
        return a.astype(np.float32)
    h, w = a.shape[:2]
    ph, pw = (-h) % s, (-w) % s
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2)
    ap = np.pad(a.astype(np.float32), pad, mode="edge")
    sh, sw = ap.shape[0] // s, ap.shape[1] // s
    if a.ndim == 2:
        return ap.reshape(sh, s, sw, s).mean(axis=(1, 3))
    return ap.reshape(sh, s, sw, s, a.shape[2]).mean(axis=(1, 3))


def extend_subject_colours(rgb: Array, support: Array, width: int = 24,
                           levels: Sequence[int] = (32, 16, 8, 4, 2)) -> Array:
    """Paint the subject's own colours a little way past its silhouette.

    Without this, resampling near the edge of the subject would drag green
    screen pixels into the hair/body edge when it moves. The extension is
    built on a coarse pyramid (block-weighted by the subject mask, then spread)
    so it costs milliseconds instead of seconds, and it is only ever read at
    sample points that the alpha channel weights near zero.
    """
    support = support.astype(np.float32)
    layer = rgb.astype(np.float32, copy=True)
    if width <= 0:
        return layer
    neigh = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for s in levels:
        if s <= 1 or s > max(width, 2):
            continue
        num = block_mean(rgb * support[..., None], s)
        den = block_mean(support, s)
        have = den > 1e-4
        if not have.any():
            continue
        col = np.zeros_like(num)
        col[have] = num[have] / den[have][..., None]
        wgt = have.astype(np.float32)
        # a few cheap 8-neighbour sweeps spread colour outward by ~s px each
        for _ in range(4):
            acc = np.zeros_like(col)
            wacc = np.zeros_like(wgt)
            for dy, dx in neigh:
                acc += np.roll(col * wgt[..., None], (dy, dx), axis=(0, 1))
                wacc += np.roll(wgt, (dy, dx), axis=(0, 1))
            spread = (wacc > 1e-6) & (wgt < 0.5)
            if not spread.any():
                break
            col[spread] = (acc / np.maximum(wacc, 1e-6)[..., None])[spread]
            wgt = np.where(spread | (wgt > 0.5), 1.0, 0.0)
        up = np.stack([upsample_2d(col[..., c], rgb.shape[1], rgb.shape[0])
                       for c in range(rgb.shape[2])], axis=-1)
        layer = np.where(support[..., None] > 0.5, layer, up)
    outside = support <= 0.5
    if outside.any():
        sm = blur_rgb(layer, 1.2)
        layer = np.where(outside[..., None], sm, layer).astype(np.float32)
    return layer


# --------------------------------------------------------------------------- #
# motion
# --------------------------------------------------------------------------- #


@dataclass
class MotionParams:
    breath: float = 1.0        # master multiplier for breathing
    hair: float = 1.0          # master multiplier for hair sway
    fabric: float = 1.0        # master multiplier for fabric / strings
    sway: float = 1.0          # whole-subject weight shift
    edge_lock: float = 0.5     # 1.0 = silhouette pinned, 0.0 = free edges
    amplitude: float = 1.0     # global output amplitude scaler
    z_jitter: float = 0.0      # optional additive output-space jitter (px rms), off by default


@dataclass
class Zone:
    """A hand-placed motion zone (earrings, bracelets, lenses, ...)."""
    name: str
    mode: str                  # 'rotate_top' | 'sway' | 'glint'
    rect: Tuple[float, float, float, float]  # x, y, w, h (px, or normalised 0-1)
    amplitude: float = 1.0
    cycles: int = 2
    phase: float = 0.0

    # resolved at runtime
    px_rect: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0), repr=False)


class MotionModel:
    """Builds the per-frame displacement field for a specific image."""

    def __init__(self, rgb: Array, matte: Array, params: MotionParams,
                 zones: Sequence[Zone], seed: int = 7):
        self.p = params
        self.zones = list(zones)
        self.h, self.w = matte.shape
        self.rng = np.random.default_rng(seed)

        ys, xs = np.nonzero(matte > 0.5)
        if len(ys) == 0:
            raise SystemExit("error: the chroma key found no subject pixels. "
                             "Try --key-low / --key-high, or check that the "
                             "background really is a bright green screen.")
        self.x0, self.x1 = int(xs.min()), int(xs.max()) + 1
        self.y0, self.y1 = int(ys.min()), int(ys.max()) + 1
        self.sub_h = float(self.y1 - self.y0)
        self.sub_w = float(self.x1 - self.x0)

        # --- head estimate: centroid + lateral extent of the top 18 % of the subject
        top_rows = matte[self.y0:self.y0 + max(1, int(0.18 * self.sub_h))] > 0.5
        txs = np.nonzero(top_rows.any(axis=0))[0]
        if len(txs):
            self.head_x0, self.head_x1 = int(txs.min()), int(txs.max()) + 1
        else:
            self.head_x0, self.head_x1 = self.x0, self.x1
        self.head_cx = 0.5 * (self.head_x0 + self.head_x1)
        self.head_half = max(0.5 * (self.head_x1 - self.head_x0), 0.05 * self.sub_w)

        # --- feature sizes: peak displacement, as a fraction of subject height.
        # A 1.5 kpx-tall subject therefore breathes ~3 px and its curls sway
        # ~2-4 px, which is what "subtle micro-motion" means at that size.
        self.a_breath = 0.0022 * self.sub_h
        self.a_hair = 0.0016 * self.sub_h
        self.a_fabric = 0.0013 * self.sub_h

        # --- spatially varying noise for strand-level variation
        noise_shape = (self.h, self.w)
        self.hair_noise = smooth_noise(noise_shape, max(8.0, self.sub_h / 16.0), self.rng, 1.6)
        self.hair_noise2 = smooth_noise(noise_shape, max(6.0, self.sub_h / 24.0), self.rng, 1.2)
        self.fab_noise = smooth_noise(noise_shape, max(6.0, self.sub_h / 26.0), self.rng, 1.0)
        self.fab_noise2 = smooth_noise(noise_shape, max(5.0, self.sub_h / 30.0), self.rng, 0.9)

        # --- region weights (computed on the full frame, cropped later)
        Y = np.arange(self.h, dtype=np.float32)[:, None] * np.ones((1, self.w), np.float32)
        X = np.ones((self.h, 1), np.float32) * np.arange(self.w, dtype=np.float32)[None, :]

        subj = (matte > 0.02).astype(np.float32)
        interior = blur_f(matte, radius=max(1.5, 0.004 * self.sub_h))
        edge_prox = np.clip(1.0 - interior * 1.8, 0.0, 1.0)   # 1 near silhouette

        # breathing: a smooth bump centred on the chest, fading out towards the
        # hips (which real breathing hardly moves) and towards the arms
        self.chest_y = self.y0 + 0.42 * self.sub_h
        self.sigma_y = 0.30 * self.sub_h
        self.w_torso = subj * np.exp(-0.5 * ((Y - self.chest_y) / self.sigma_y) ** 2)
        # chest: the same bump, used for the subtle lateral breath expansion
        self.w_chest = self.w_torso
        xn = np.clip((X - self.head_cx) / max(self.head_half * 2.0, 1.0), -1.0, 1.0)
        self.xn = xn.astype(np.float32)

        # head: confined to the head's horizontal band so a raised arm is untouched
        head_band = smoothstep(self.head_x0 - 0.10 * self.head_half,
                               self.head_x0 + 0.10 * self.head_half, X) \
            * (1.0 - smoothstep(self.head_x1 - 0.10 * self.head_half,
                                self.head_x1 + 0.10 * self.head_half, X))
        self.w_head = subj * head_band.astype(np.float32) \
            * (1.0 - smoothstep(self.y0 + 0.16 * self.sub_h,
                                self.y0 + 0.34 * self.sub_h, Y))

        # hair: top of the subject + the loose silhouette, within the head band
        hair_y = 1.0 - smoothstep(self.y0 + 0.24 * self.sub_h,
                                  self.y0 + 0.58 * self.sub_h, Y)
        self.w_hair = subj * hair_y * head_band.astype(np.float32) \
            * (0.45 + 0.55 * edge_prox)

        self.w_fab = subj * smoothstep(self.y0 + 0.40 * self.sub_h,
                                       self.y0 + 0.62 * self.sub_h, Y) \
            * (0.55 + 0.45 * edge_prox)

        # edge damping: silhouette keeps `edge_lock` of the motion, interior is free
        self.edge_scale = (params.edge_lock + (1.0 - params.edge_lock)
                           * smoothstep(0.25, 0.90, interior)).astype(np.float32)

        # zone weights / pivots
        self.zone_masks: List[Array] = []
        for z in self.zones:
            x, y, zw, zh = z.rect
            if max(z.rect) <= 1.0:                     # normalised input
                x, y, zw, zh = x * self.w, y * self.h, zw * self.w, zh * self.h
            ix0, iy0 = int(round(x)), int(round(y))
            ix1, iy1 = int(round(x + zw)), int(round(y + zh))
            ix0, iy0 = max(ix0, 0), max(iy0, 0)
            ix1, iy1 = min(ix1, self.w), min(iy1, self.h)
            z.px_rect = (ix0, iy0, ix1, iy1)
            m = np.zeros((self.h, self.w), np.float32)
            if ix1 > ix0 and iy1 > iy0:
                m[iy0:iy1, ix0:ix1] = 1.0
                m = blur_f(m, 2.0)
            self.zone_masks.append(m)

        # work box: subject bbox + padding, so the rest of the frame is a pure copy
        # enough headroom for the largest displacement (so sampling never needs
        # pixels from outside the work box) but no more: the box stays small
        pad = int(math.ceil(0.02 * self.sub_h * max(params.amplitude, 1e-3))) + 4
        self.wx0, self.wy0 = max(0, self.x0 - pad), max(0, self.y0 - pad)
        self.wx1, self.wy1 = min(self.w, self.x1 + pad), min(self.h, self.y1 + pad)
        self.local_xs = np.arange(self.wx0, self.wx1, dtype=np.float32)[None, :] \
            * np.ones((self.wy1 - self.wy0, 1), np.float32)
        self.local_ys = np.ones((1, self.wx1 - self.wx0), np.float32) \
            * np.arange(self.wy0, self.wy1, dtype=np.float32)[:, None]
        self.full_xs = np.ones((self.h, 1), np.float32) * np.arange(
            self.w, dtype=np.float32)[None, :]
        self.full_ys = np.arange(self.h, dtype=np.float32)[:, None] * np.ones(
            (1, self.w), np.float32)

    # -- crops ------------------------------------------------------------- #
    def _crop(self, a: Array) -> Array:
        return a[self.wy0:self.wy1, self.wx0:self.wx1]

    def displacement(self, t: float) -> Tuple[Array, Array, Array]:
        """Displacement (dx, dy) and a brightness multiplier for phase t in [0, 1).

        All three arrays come back cropped to the work box
        ``[wy0:wy1, wx0:wx1]`` — the only region of the frame that can differ
        from the source — with coordinates expressed in that crop's index space.
        """
        p = self.p
        w = 2.0 * math.pi
        # Every term is a sinusoid of an integer number of cycles per loop, so
        # the field is perfectly periodic and the loop closes with no jump.
        # The phases differ per term: if they did not, every component would
        # vanish at the same instant and the animation would visibly freeze
        # twice per loop. Frame 0 is therefore not bit-identical to the still
        # (it is off by a fraction of the amplitude) and the report says by how
        # much — that is the honest trade for motion that never stalls.
        s = lambda k, ph=0.0: float(math.sin(w * k * t + ph))      # noqa: E731

        bone = s(1)                              # one breath per loop
        bone_late = s(1, -0.35)                  # spine lags the chest
        s2 = s(2, 0.7)
        s3 = s(3, 1.9)

        dy = (-self.a_breath * p.breath * bone) * self.w_torso \
            * self.edge_scale
        dx = (self.a_breath * 0.35 * p.breath * s2) * self.w_chest * self.xn \
            * self.edge_scale
        # head / shoulders ride the breath a little, with a natural lag
        dy = dy + (self.a_breath * 0.42 * p.breath * bone_late) * self.w_head

        # ---- hair: curl groups move, the spatial noise de-synchronises them --- #
        base = 0.72 * s(1, 0.9) + 0.28 * s2
        dx = dx + self.a_hair * p.hair * (base + 0.62 * self.hair_noise * s3) * self.w_hair
        dy = dy + self.a_hair * p.hair * (0.42 * s(2, 1.3) + 0.30 * self.hair_noise2 * s3) \
            * self.w_hair

        # ---- fabric / strings -------------------------------------------- #
        dx = dx + self.a_fabric * p.fabric * (0.75 + 0.5 * self.fab_noise) \
            * np.sin(w * 4 * t) * self.w_fab
        dy = dy + self.a_fabric * p.fabric * (0.62 + 0.45 * self.fab_noise2) \
            * np.sin(w * 4 * t + 0.8) * self.w_fab

        # ---- gentle weight shift, strongest through the upper body -------- #
        if p.sway:
            dx = dx + self.a_breath * 0.35 * p.sway * s(1) * self.w_torso * self.edge_scale
            dy = dy + self.a_breath * 0.20 * p.sway * s(2) * self.w_torso * self.edge_scale

        # ---- zones (earrings / bracelets / lenses) ------------------------ #
        mult = np.ones_like(dx)
        for z, zm in zip(self.zones, self.zone_masks):
            if zm.max() <= 0.0:
                continue
            ix0, iy0, ix1, iy1 = z.px_rect
            cyc = max(1, int(z.cycles))
            swing = math.sin(w * cyc * t + z.phase)
            if z.mode == "rotate_top":
                cy = float(iy0)                      # pivot = top of the zone
                cx = 0.5 * (ix0 + ix1)
                theta = 0.035 * z.amplitude * swing
                dx = dx + (-theta * (self.full_ys - cy)) * zm
                dy = dy + (theta * (self.full_xs - cx)) * zm
            elif z.mode == "sway":
                amp = 0.015 * max(1.0, float(iy1 - iy0)) * z.amplitude * swing
                dx = dx + amp * zm
                dy = dy + 0.45 * amp * math.cos(w * 2 * cyc * t + z.phase) * zm
            elif z.mode == "glint":
                span = max(1.0, float(ix1 - ix0))
                sweep = np.sin(2.0 * math.pi * 0.75 * (self.full_xs - ix0) / span)
                mult = mult * (1.0 + 0.030 * z.amplitude * zm
                               * sweep * math.sin(w * cyc * t + z.phase))
            else:
                raise SystemExit(f"error: unknown zone mode '{z.mode}' "
                                 f"(use rotate_top | sway | glint)")

        if p.z_jitter > 0.0:
            dx = dx + self.rng.normal(0.0, p.z_jitter, dx.shape).astype(np.float32) * self.w_torso
            dy = dy + self.rng.normal(0.0, p.z_jitter, dy.shape).astype(np.float32) * self.w_torso

        amp = p.amplitude
        return self._crop(dx) * amp, self._crop(dy) * amp, self._crop(mult)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


@dataclass
class RenderResult:
    frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    matte_coverage: float
    subject_bbox: Tuple[int, int, int, int]
    background_bit_exact: bool
    background_pixels_checked: int
    max_background_delta: int
    background_check_mode: str
    silhouette_lock: bool
    loop_wrap_step: float
    mean_step: float
    max_step: float
    motion_subject_peak: float
    motion_subject_mean: float
    motion_px_max: float
    motion_px_mean: float
    motion_px_min: float
    max_step_frame: Optional[int]
    frame0_max_delta_from_source: int
    frame0_mean_delta_from_source_pct: float
    files: Dict[str, str] = field(default_factory=dict)
    checks_passed: bool = True
    notes: List[str] = field(default_factory=list)


def attach_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_chroma(source: Array, matte_params: MatteParams) -> Tuple[Array, Array, Array]:
    """Return (extended colour layer, soft matte, hard support mask)."""
    raw = suppress_speckle(raw_matte(source, matte_params), matte_params.despeckle)
    matte = blur_f(raw, matte_params.feather)
    support = raw > 0.5
    # the extended colour layer is only needed within a few px of the subject,
    # so build it on a crop: same result, a fraction of the work
    ys, xs = np.nonzero(support)
    if len(ys) == 0:
        layer = source.astype(np.float32, copy=True)
    else:
        m = int(max(matte_params.grow, 12) + 8)
        y0, y1 = max(0, int(ys.min()) - m), min(source.shape[0], int(ys.max()) + 1 + m)
        x0, x1 = max(0, int(xs.min()) - m), min(source.shape[1], int(xs.max()) + 1 + m)
        sub = extend_subject_colours(source[y0:y1, x0:x1], support[y0:y1, x0:x1],
                                     matte_params.grow)
        layer = source.astype(np.float32, copy=True)
        layer[y0:y1, x0:x1] = sub
    return layer, matte, support


def render(source_u8: Array, matte_params: MatteParams, motion: MotionParams,
           zones: Sequence[Zone], frames: int, fps: float, seed: int = 7,
           out_mp4: Optional[str] = None, ffmpeg: Optional[str] = None,
           crf: int = 15, out_webp: Optional[str] = None, webp_max_width: int = 720,
           webp_quality: int = 90, webp_lossless: bool = False,
           debug_dir: Optional[str] = None, bg_lock: bool = True,
           debug_frames: Sequence[int] = (0, 1, 2)) -> RenderResult:
    """Render the loop.

    bg_lock=True (default) is a strict guarantee: every pixel that the chroma key
    classifies as green screen is copied from the source untouchable, so the
    background of the video is bit-identical to the still. The subject may then
    move only inside its own silhouette (plus the soft matte feather), which is
    why the reported background check is exact.

    bg_lock=False lets the silhouette edge breathe freely; the subject can then
    move 2-3 px past its original outline, at the cost of a few green-screen
    pixels near the edge being rewritten. The report will say so.
    """
    h, w = source_u8.shape[:2]
    rng_check = np.random.default_rng(seed + 991)
    source = source_u8.astype(np.float32)
    layer, matte, support = run_chroma(source, matte_params)
    model = MotionModel(source, matte, motion, zones, seed=seed)

    wx0, wy0, wx1, wy1 = model.wx0, model.wy0, model.wx1, model.wy1
    src_local = source[wy0:wy1, wx0:wx1]
    layer_local = layer[wy0:wy1, wx0:wx1]
    matte_local = matte[wy0:wy1, wx0:wx1]
    # coordinate grids *relative to the work box*: sample_bilinear indexes the
    # cropped arrays, so absolute image coordinates would be out of range
    xs = model.local_xs - wx0
    ys = model.local_ys - wy0

    proc = None
    if out_mp4:
        if ffmpeg is None:
            raise SystemExit("error: ffmpeg not found. `pip install imageio-ffmpeg` "
                             "or install ffmpeg, or pass --no-mp4.")
        os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{w}x{h}", "-framerate", f"{fps}", "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    webp_frames: List[Image.Image] = []
    webp_scale = 1.0
    if out_webp and webp_max_width and w > webp_max_width:
        webp_scale = webp_max_width / float(w)

    subj_local = matte_local > 0.5
    support_local = support[wy0:wy1, wx0:wx1]
    if not bg_lock:
        # free-edge mode: allow the silhouette to move a few px past its outline
        grown = blur_f(support_local.astype(np.float32), max(2.0, matte_params.grow / 3.0))
        support_local = grown > 0.02
    src_local_u8 = source_u8[wy0:wy1, wx0:wx1]

    # background verification: a fixed random sample every frame, everything at the end
    bg_flat = np.flatnonzero(~support_local.ravel())
    bg_sample = (rng_check.choice(bg_flat, size=min(50000, bg_flat.size), replace=False)
                 if bg_flat.size else np.array([], dtype=np.int64))
    bg_ys, bg_xs = (np.unravel_index(bg_sample, support_local.shape)
                    if bg_sample.size else (np.array([], int), np.array([], int)))
    n_bg_checked = int(bg_flat.size)
    bg_sample_checks = int(bg_sample.size) * frames
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    debug_frame_set = set(int(v) for v in (debug_frames or ()))

    prev = None
    max_bg_delta = 0
    steps: List[float] = []
    step_frames: List[int] = []
    first_frame = None
    last_frame = None
    motion_acc = np.zeros((wy1 - wy0, wx1 - wx0), np.float32)
    px_max = 0.0
    px_sum = 0.0
    px_min = float("inf")
    frame0_max = 0
    frame0_mean = 0.0

    for i in range(frames):
        t = i / float(frames)
        dx, dy, mult = model.displacement(t)

        warped_rgb = sample_bilinear(layer_local, dx, dy, xs, ys)
        warped_m = np.clip(sample_bilinear(matte_local, dx, dy, xs, ys), 0.0, 1.0)
        if np.isscalar(mult):
            pass
        else:
            warped_rgb = warped_rgb * mult[..., None]

        if bg_lock:
            # never let a semi-transparent matte pixel gain opacity: keeps the
            # silhouette from growing a hard rim, and keeps frame 0 exact
            alpha = np.where(support_local, np.minimum(warped_m, matte_local), 0.0)
        else:
            alpha = warped_m * support_local
        m3 = alpha[..., None]
        # backdrop is the *extended* subject colour, so a pixel on the silhouette
        # that the subject moves away from shows soft subject colour rather than
        # a hole of green screen
        local = warped_rgb * m3 + layer_local * (1.0 - m3)

        out_local = np.clip(np.rint(local), 0, 255).astype(np.uint8)
        out_local = np.where(support_local[..., None], out_local, src_local_u8)

        frame = source_u8.copy()
        frame[wy0:wy1, wx0:wx1] = out_local

        # --- verification: background must be bit-identical --------------- #
        if bg_sample.size:
            delta = np.abs(out_local[bg_ys, bg_xs].astype(np.int16)
                           - src_local_u8[bg_ys, bg_xs].astype(np.int16))
            max_bg_delta = max(max_bg_delta, int(delta.max()) if delta.size else 0)

        if proc is not None:
            proc.stdin.write(frame.tobytes())

        if out_webp:
            im = Image.fromarray(frame)
            if webp_scale != 1.0:
                im = im.resize((int(round(w * webp_scale)), int(round(h * webp_scale))),
                               Image.LANCZOS)
            webp_frames.append(im)

        if prev is not None:
            steps.append(float(np.abs(out_local.astype(np.int16)
                                      - prev.astype(np.int16)).mean()))
            step_frames.append(i)
        prev = out_local
        # displacement actually applied inside the subject, in pixels
        mag = np.sqrt(dx * dx + dy * dy)
        frame_px = float(mag[subj_local].max()) if subj_local.any() else 0.0
        px_max = max(px_max, frame_px)
        px_sum += frame_px
        px_min = min(px_min, frame_px)
        if i == 0:
            first_frame = frame.copy()
            zero_delta = np.abs(out_local.astype(np.int16) - src_local_u8.astype(np.int16))
            frame0_max = int(zero_delta.max())
            frame0_mean = float(zero_delta.mean()) / 255.0 * 100.0
        last_frame = frame
        if debug_dir and i in debug_frame_set:
            Image.fromarray(frame).save(os.path.join(debug_dir, "frame_%04d.png" % i))
        motion_acc = np.maximum(motion_acc, np.abs(local - src_local).max(axis=-1))

    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("error: ffmpeg failed while encoding the mp4")

    # full-frame background verification on the last rendered frame
    if n_bg_checked:
        crop_all = frame[wy0:wy1, wx0:wx1]
        full_delta = np.abs(crop_all.astype(np.int16) - src_local_u8.astype(np.int16))
        mask3 = np.repeat((~support_local)[..., None], 3, axis=2)
        max_bg_delta = max(max_bg_delta, int(full_delta[mask3].max()) if mask3.any() else 0)

    files: Dict[str, str] = {}
    if out_mp4:
        files["mp4"] = out_mp4
    if out_webp and webp_frames:
        os.makedirs(os.path.dirname(os.path.abspath(out_webp)) or ".", exist_ok=True)
        webp_frames[0].save(
            out_webp, save_all=True, append_images=webp_frames[1:],
            duration=int(round(1000.0 / max(fps, 1e-6))), loop=0,
            lossless=bool(webp_lossless), quality=int(webp_quality), method=4)
        files["webp"] = out_webp

    loop_wrap = float(np.abs(first_frame.astype(np.int16)
                             - last_frame.astype(np.int16)).mean())
    mean_step = float(np.mean(steps)) if steps else 0.0
    max_step = float(np.max(steps)) if steps else 0.0

    res = RenderResult(
        frames=frames, fps=fps, width=w, height=h, duration_s=frames / max(fps, 1e-6),
        matte_coverage=float((matte > 0.5).mean()),
        subject_bbox=(model.x0, model.y0, model.x1, model.y1),
        background_bit_exact=(max_bg_delta == 0),
        background_pixels_checked=n_bg_checked,
        max_background_delta=max_bg_delta,
        background_check_mode=("every frame: %s sampled pixels | final frame: all %s pixels"
                               % (f"{bg_sample_checks:,}", f"{n_bg_checked:,}")),
        silhouette_lock=bool(bg_lock),
        loop_wrap_step=loop_wrap, mean_step=mean_step, max_step=max_step,
        motion_subject_peak=float(motion_acc[subj_local].max()) if subj_local.any() else 0.0,
        motion_subject_mean=float(motion_acc[subj_local].mean()) if subj_local.any() else 0.0,
        motion_px_max=px_max,
        motion_px_mean=px_sum / max(frames, 1),
        motion_px_min=0.0 if px_min == float("inf") else px_min,
        max_step_frame=(step_frames[int(np.argmax(steps))] if steps else None),
        frame0_max_delta_from_source=frame0_max,
        frame0_mean_delta_from_source_pct=frame0_mean,
        files=files,
    )

    if res.matte_coverage > 0.92:
        res.notes.append("WARN: the chroma key found almost no background — is this really "
                         "a green screen shot? Try --key-low / --key-high.")
    elif res.matte_coverage < 0.02:
        res.notes.append("WARN: the chroma key found almost no subject — the key may be "
                         "swallowing the subject. Try a lower --key-high.")
    bx0, by0, bx1, by1 = res.subject_bbox
    if bx0 <= 1 or by0 <= 1 or bx1 >= w - 1 or by1 >= h - 1:
        res.notes.append("WARN: the subject touches the frame edge, so the silhouette cannot "
                         "be pinned there; those pixels may be written in every frame. Keep a "
                         "green margin around the subject when shooting.")

    if not res.background_bit_exact:
        if bg_lock:
            res.checks_passed = False
            res.notes.append("FAIL: background pixels are not bit-identical to the source.")
        else:
            res.notes.append("bg_lock off: background may differ from the source near the "
                             "silhouette (free-edge mode requested).")
    if res.motion_px_max < 0.2:
        res.checks_passed = False
        res.notes.append("FAIL: no measurable motion inside the subject — check the key.")
    if res.motion_px_min < 0.15 * max(res.motion_px_max, 1e-6):
        res.checks_passed = False
        res.notes.append("FAIL: the motion stalls at some point in the loop "
                         f"(min {res.motion_px_min:.2f} px vs peak {res.motion_px_max:.2f} px).")
    if res.motion_px_max > 12.0:
        res.checks_passed = False
        res.notes.append(f"FAIL: motion is no longer subtle ({res.motion_px_max:.1f} px peak).")
    if steps and loop_wrap > 1.5 * max(max_step, 1e-6):
        res.checks_passed = False
        res.notes.append("FAIL: loop seam is not continuous (wrap step is an outlier).")

    if debug_dir:
        Image.fromarray((matte * 255).astype(np.uint8), "L").save(
            os.path.join(debug_dir, "matte.png"))
        amp = 30.0
        vis = np.clip(128.0 + motion_acc * amp, 0, 255).astype(np.uint8)
        Image.fromarray(vis, "L").save(os.path.join(debug_dir, "motion_x%d.png" % int(amp)))
        Image.fromarray(last_frame).save(os.path.join(debug_dir, "frame_last.png"))
        files["debug_dir"] = debug_dir

    return res


def parse_zones(path: Optional[str]) -> List[Zone]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data["zones"] if isinstance(data, dict) else data
    zones = []
    for z in raw:
        rect = z.get("rect") or [z["x"], z["y"], z["w"], z["h"]]
        zones.append(Zone(name=z.get("name", "zone"), mode=z.get("mode", "sway"),
                          rect=tuple(float(v) for v in rect),
                          amplitude=float(z.get("amplitude", 1.0)),
                          cycles=int(z.get("cycles", 2)),
                          phase=float(z.get("phase", 0.0))))
    return zones


def load_source(path: str, scale: float) -> Array:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 177, 64, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    if scale != 1.0:
        im = im.resize((max(1, int(round(im.width * scale))),
                        max(1, int(round(im.height * scale)))), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="micro_anim",
        description="Turn a green-screen still into a seamlessly looping micro-animation "
                    "without regenerating a single pixel.")
    ap.add_argument("--input", "-i", required=True, help="source image (green screen)")
    ap.add_argument("--out", "-o", required=True,
                    help="output base path; writes <out>.mp4 and <out>.webp")
    ap.add_argument("--frames", type=int, default=96, help="frames in the loop (default 96)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--scale", type=float, default=1.0, help="resize source before render")
    ap.add_argument("--amplitude", type=float, default=1.0, help="global motion scaler")
    ap.add_argument("--breath", type=float, default=1.0)
    ap.add_argument("--hair", type=float, default=1.0)
    ap.add_argument("--fabric", type=float, default=1.0)
    ap.add_argument("--sway", type=float, default=1.0)
    ap.add_argument("--edge-lock", type=float, default=0.5,
                    help="1.0 pins the silhouette, 0.0 lets edges move freely")
    ap.add_argument("--key-low", type=float, default=0.10,
                    help="normalised chroma ratio below which a pixel is pure subject")
    ap.add_argument("--key-high", type=float, default=0.22,
                    help="normalised chroma ratio above which a pixel is pure green screen")
    ap.add_argument("--feather", type=float, default=1.2)
    ap.add_argument("--grow", type=int, default=10)
    ap.add_argument("--zones", help="JSON file with extra motion zones (earrings, lenses...)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--allow-bg-motion", action="store_true",
                    help="let the silhouette edge move freely; a few green-screen pixels "
                         "near the subject may then change (default: strict bit-exact bg)")
    ap.add_argument("--crf", type=int, default=15, help="H.264 quality (lower = better)")
    ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--no-webp", action="store_true")
    ap.add_argument("--webp-max-width", type=int, default=720)
    ap.add_argument("--webp-quality", type=int, default=90)
    ap.add_argument("--webp-lossless", action="store_true",
                    help="pixel-exact WebP (large files)")
    ap.add_argument("--debug-dir", help="write matte.png / motion map / first+last frames")
    ap.add_argument("--debug-frames", default="0,1,2",
                    help="comma-separated frame indices to dump into --debug-dir")
    args = ap.parse_args(argv)

    if args.frames < 2:
        raise SystemExit("error: --frames must be >= 2")

    source = load_source(args.input, args.scale)
    mp = MatteParams(key_low=args.key_low, key_high=args.key_high,
                     feather=args.feather, grow=args.grow)
    mm = MotionParams(breath=args.breath, hair=args.hair, fabric=args.fabric,
                      sway=args.sway, edge_lock=args.edge_lock, amplitude=args.amplitude)
    zones = parse_zones(args.zones)

    ffmpeg = None if args.no_mp4 else attach_ffmpeg()
    out_mp4 = None if args.no_mp4 else args.out + ".mp4"
    out_webp = None if args.no_webp else args.out + ".webp"

    res = render(source, mp, mm, zones, frames=args.frames, fps=args.fps, seed=args.seed,
                 out_mp4=out_mp4, ffmpeg=ffmpeg, crf=args.crf,
                 out_webp=out_webp, webp_max_width=args.webp_max_width,
                 webp_quality=args.webp_quality, webp_lossless=args.webp_lossless,
                 debug_dir=args.debug_dir, bg_lock=not args.allow_bg_motion,
                 debug_frames=[int(v) for v in args.debug_frames.split(",") if v.strip()])

    report_path = args.out + ".report.json"
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    payload = asdict(res)
    payload["files"]["report"] = report_path
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(json.dumps(payload, indent=2))
    return 0 if res.checks_passed else 2


if __name__ == "__main__":
    sys.exit(main())
