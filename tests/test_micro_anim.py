#!/usr/bin/env python3
"""Tests for the green-screen micro-animation renderer.

Run from the repo root:   python -m pytest tests -q
or without pytest:        python tests/test_micro_anim.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renderer"))

import micro_anim as ma  # noqa: E402

GREEN = (0, 177, 64)


def synthetic_scene(w: int = 220, h: int = 300, seed: int = 3) -> np.ndarray:
    """A crude green-screen 'portrait': green field + noisy subject silhouette.

    High-frequency noise inside the subject makes resampling measurable.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = GREEN
    img[:, :, 1] = 177 + rng.integers(-2, 3, (h, w)).astype(np.uint8)
    img[:, :, 0] = rng.integers(0, 4, (h, w)).astype(np.uint8)
    img[:, :, 2] = rng.integers(0, 4, (h, w)).astype(np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0
    head = (((xx - cx) / (0.17 * w)) ** 2 + ((yy - 0.16 * h) / (0.13 * h)) ** 2) < 1.0
    torso = (np.abs(xx - cx) < 0.24 * w) & (yy > 0.30 * h) & (yy < 0.95 * h)
    arm = (np.abs(xx - (cx + 0.42 * w)) < 0.05 * w) & (yy > 0.18 * h) & (yy < 0.55 * h)
    body = head | torso | arm

    skin = np.stack([
        205 + rng.integers(-25, 26, (h, w)),
        150 + rng.integers(-25, 26, (h, w)),
        125 + rng.integers(-25, 26, (h, w)),
    ], axis=-1).astype(np.uint8)
    img[body] = skin[body]
    dark = head & (yy < 0.14 * h)                       # "hair" patch
    img[dark] = np.stack([
        60 + rng.integers(-20, 21, (h, w)),
        40 + rng.integers(-20, 21, (h, w)),
        30 + rng.integers(-20, 21, (h, w)),
    ], axis=-1).astype(np.uint8)[dark]
    return img


class TestChromaKey(unittest.TestCase):
    def test_matte_separates_screen_from_subject(self):
        img = synthetic_scene()
        matte = ma.chroma_matte(img.astype(np.float32), ma.MatteParams())
        self.assertLess(matte[10, 10], 0.01, "corner (pure screen) must key out to background")
        self.assertGreater(matte[150, 110], 0.99, "torso centre must be opaque subject")
        self.assertGreater(matte[45, 110], 0.99, "hair patch must be opaque subject")
        self.assertAlmostEqual(float(matte.min()), 0.0, places=5)
        self.assertAlmostEqual(float(matte.max()), 1.0, places=5)

    def test_greenness_ignores_subject_warmth(self):
        warm = np.array([[[205.0, 150.0, 125.0]]])
        yellow = np.array([[[254.0, 210.0, 21.0]]])
        hair = np.array([[[6.0, 6.0, 3.0]]])
        green = np.array([[[0.0, 177.0, 64.0]]])
        self.assertLess(ma.greenness(warm)[0, 0], 0.0)
        self.assertLess(ma.greenness(yellow)[0, 0], 0.0, "a yellow top must never key out")
        self.assertLess(ma.greenness(hair)[0, 0], ma.MatteParams().key_low,
                        "near-neutral dark hair must stay on the subject side")
        self.assertGreater(ma.greenness(green)[0, 0], 0.4)

    def test_key_survives_an_uneven_green_screen(self):
        """Regression: a real studio screen has a lighting gradient, not one colour.

        The raw channel-difference key left ~half a million semi-transparent
        pixels on the background; the normalised ratio key must stay clean, so
        the soft ring should be roughly the same as for a perfectly flat screen.
        """
        flat = synthetic_scene(200, 260)
        h, w = flat.shape[:2]
        ramp = np.linspace(0.72, 1.0, w, dtype=np.float32)[None, :, None]
        # background = the pixels the key should classify as screen (ratio metric)
        bg = ma.greenness(flat.astype(np.float32)) > 0.3
        self.assertGreater(bg.mean(), 0.4, "sanity: most of the scene is green screen")
        grad = np.where(bg[..., None],
                        (flat.astype(np.float32) * ramp).clip(0, 255),
                        flat.astype(np.float32)).astype(np.uint8)

        p = ma.MatteParams()
        m_flat = ma.chroma_matte(flat.astype(np.float32), p)
        m_grad = ma.chroma_matte(grad.astype(np.float32), p)

        soft = lambda m: int(((m > 0.001) & (m < 0.999)).sum())   # noqa: E731
        self.assertLess(float(m_grad[5:40, 5:40].max()), 0.05,
                        "the dark corner of the green field must still key out")
        self.assertGreater(float(m_grad[150, 110]), 0.99, "torso must stay opaque")
        self.assertLess(soft(m_grad), 1.5 * soft(m_flat) + 200,
                        f"gradient doubled the soft edge: {soft(m_flat)} -> {soft(m_grad)}")
        # ignore the intended feather right next to the subject
        near_subject = ma.blur_f((~bg).astype(np.float32), 4.0) > 0.02
        bg_far = bg & ~near_subject
        self.assertGreater(bg_far.sum(), 1000)
        self.assertLess(float(m_grad[bg_far].max()), 0.05,
                        "no green-screen pixel may key as subject, however lit it is")
        self.assertLess(int((np.abs(m_flat - m_grad) > 0.05).sum()), 60,
                        "only a handful of silhouette pixels should change with the gradient")


class TestResampling(unittest.TestCase):
    def setUp(self):
        self.img = synthetic_scene(60, 80).astype(np.float32)
        h, w = self.img.shape[:2]
        self.xs = np.arange(w, dtype=np.float32)[None, :] * np.ones((h, 1), np.float32)
        self.ys = np.ones((1, w), np.float32) * np.arange(h, dtype=np.float32)[:, None]

    def test_identity_when_displacement_is_zero(self):
        out = ma.sample_bilinear(self.img, np.zeros_like(self.xs), np.zeros_like(self.ys),
                                 self.xs, self.ys)
        np.testing.assert_allclose(out, self.img, atol=1e-3)

    def test_one_pixel_shift_uses_neighbours(self):
        dx = np.ones_like(self.xs)
        out = ma.sample_bilinear(self.img, dx, np.zeros_like(self.ys), self.xs, self.ys)
        np.testing.assert_allclose(out[:, :-1], self.img[:, 1:], atol=1e-3)

    def test_matches_naive_bilinear_on_random_fields(self):
        """The fast flat-index sampler must agree with a straightforward one."""
        rng = np.random.default_rng(0)
        dx = rng.uniform(-3, 3, self.img.shape[:2]).astype(np.float32)
        dy = rng.uniform(-3, 3, self.img.shape[:2]).astype(np.float32)
        got = ma.sample_bilinear(self.img, dx, dy, self.xs, self.ys)

        h, w = self.img.shape[:2]
        sx = np.clip(self.xs + dx, 0, w - 1)
        sy = np.clip(self.ys + dy, 0, h - 1)
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
        top = self.img[y0, x0] * (1 - fx) + self.img[y0, x1] * fx
        bot = self.img[y1, x0] * (1 - fx) + self.img[y1, x1] * fx
        want = top * (1 - fy) + bot * fy
        np.testing.assert_allclose(got, want, atol=2e-3)

    def test_edges_are_clamped_not_wrapped(self):
        dx = -5.0 * np.ones_like(self.xs)
        out = ma.sample_bilinear(self.img, dx, np.zeros_like(self.ys), self.xs, self.ys)
        np.testing.assert_allclose(out[:, 0], self.img[:, 0], atol=1e-3)


class TestMotionModel(unittest.TestCase):
    def setUp(self):
        self.img = synthetic_scene(220, 300)
        self.f = self.img.astype(np.float32)
        self.matte = ma.chroma_matte(self.f, ma.MatteParams())

    def test_displacement_zero_on_background(self):
        m = ma.MotionModel(self.f, self.matte, ma.MotionParams(), [])
        bg = self.matte < 1e-6
        for t in (0.0, 0.13, 0.37, 0.5, 0.81):
            dx, dy, mult = m.displacement(t)
            dxf = np.zeros_like(self.matte)
            dyf = np.zeros_like(self.matte)
            dxf[m.wy0:m.wy1, m.wx0:m.wx1] = dx
            dyf[m.wy0:m.wy1, m.wx0:m.wx1] = dy
            self.assertEqual(float(np.abs(dxf[bg]).max()), 0.0, "background must never move")
            self.assertEqual(float(np.abs(dyf[bg]).max()), 0.0, "background must never move")

    def test_motion_is_measurable_and_subtle(self):
        m = ma.MotionModel(self.f, self.matte, ma.MotionParams(), [])
        peak = 0.0
        for i in range(24):
            dx, dy, _ = m.displacement(i / 24.0)
            peak = max(peak, float(np.abs(dx).max()), float(np.abs(dy).max()))
        self.assertGreater(peak, 0.15, "motion should be visible")
        self.assertLess(peak, 6.0, "motion must stay subtle (px)")

    def test_motion_never_freezes(self):
        """No instant in the loop may stall: phases must differ per component.

        With a single shared phase every term would vanish together at two
        points of the loop and the animation would visibly pause.
        """
        m = ma.MotionModel(self.f, self.matte, ma.MotionParams(), [])
        subj = self.matte[m.wy0:m.wy1, m.wx0:m.wx1] > 0.5
        n = 48
        peaks = []
        for i in range(n):
            dx, dy, _ = m.displacement(i / n)
            mag = np.sqrt(dx * dx + dy * dy)
            peaks.append(float(mag[subj].max()))
        peak, floor = max(peaks), min(peaks)
        # the envelope is allowed to relax (a real breath slows at the top of the
        # inhale) but it must never collapse to a standstill
        self.assertGreater(floor, 0.15 * peak,
                           f"motion stalls: min {floor:.3f} px vs peak {peak:.3f} px")

    def test_frame_zero_stays_faithful_to_the_source(self):
        """t = 0 need not be bit-identical, but it must be a micro-motion away."""
        m = ma.MotionModel(self.f, self.matte, ma.MotionParams(), [])
        dx, dy, mult = m.displacement(0.0)
        mag = np.sqrt(dx * dx + dy * dy)
        subj = self.matte[m.wy0:m.wy1, m.wx0:m.wx1] > 0.5
        self.assertLess(float(mag[subj].max()), 3.0,
                        "frame 0 must stay within a couple of px of the photograph")
        np.testing.assert_allclose(mult, 1.0, atol=0.01)

    def test_loop_wraps_smoothly(self):
        m = ma.MotionModel(self.f, self.matte, ma.MotionParams(), [])
        n = 48
        fields = [m.displacement(i / n) for i in range(n)]
        wrap = np.abs(fields[0][0] - fields[-1][0]).max() + \
            np.abs(fields[0][1] - fields[-1][1]).max()
        steps = [np.abs(fields[i + 1][0] - fields[i][0]).max() +
                 np.abs(fields[i + 1][1] - fields[i][1]).max() for i in range(n - 1)]
        self.assertLessEqual(wrap, 1.6 * max(steps) + 1e-6)

    def test_rotate_top_zone_pivots_at_its_top_edge(self):
        zone = ma.Zone(name="earring", mode="rotate_top", rect=(90, 60, 20, 40), amplitude=2.0)
        p = ma.MotionParams(breath=0, hair=0, fabric=0, sway=0)
        m = ma.MotionModel(self.f, self.matte, p, [zone])
        # t = 0.125 is a sine peak for cycles=2; t = 0.25 would sit on a zero crossing
        dx, dy, _ = m.displacement(0.125)
        # local coordinates of the zone inside the cropped work box
        cy = 60 - m.wy0
        cx = 100 - m.wx0
        self.assertAlmostEqual(float(dx[cy, cx]), 0.0, places=4,
                               msg="pivot row must not translate")
        self.assertGreater(abs(float(dx[cy + 30, cx])), 0.3,
                           "points below the pivot must swing")

    def test_glint_only_touches_brightness(self):
        zone = ma.Zone(name="lens", mode="glint", rect=(80, 40, 60, 20), amplitude=1.0)
        still = ma.MotionParams(breath=0, hair=0, fabric=0, sway=0)
        m = ma.MotionModel(self.f, self.matte, still, [zone])
        dx0, dy0, mult0 = m.displacement(0.0)
        dx1, _, mult1 = m.displacement(0.5)
        np.testing.assert_allclose(dx0, dx1, atol=1e-6)
        self.assertLess(float(np.abs(mult1 - 1.0).max()), 0.05)


class TestEndToEnd(unittest.TestCase):
    def test_render_report(self):
        img = synthetic_scene(180, 240)
        with tempfile.TemporaryDirectory() as td:
            res = ma.render(img, ma.MatteParams(), ma.MotionParams(), [], frames=12, fps=12,
                            seed=5, out_mp4=None, out_webp=os.path.join(td, "preview.webp"),
                            webp_max_width=120, debug_dir=td)
            self.assertTrue(res.background_bit_exact,
                            f"background drifted by {res.max_background_delta}")
            self.assertGreater(res.background_pixels_checked, 0)
            self.assertGreater(res.motion_subject_peak, 1.0)
            self.assertLess(res.loop_wrap_step, 1.5 * max(res.max_step, 1e-6))
            self.assertGreater(res.motion_px_min, 0.15 * res.motion_px_max,
                               "the loop must not stall")
            self.assertTrue(res.checks_passed, res.notes)
            self.assertTrue(os.path.exists(os.path.join(td, "preview.webp")))
            # frame 0 is the photograph with one frame's worth of micro-motion
            f0 = np.asarray(Image.open(os.path.join(td, "frame_0000.png")))
            self.assertLess(float(np.abs(f0.astype(np.int16)
                                         - img.astype(np.int16)).mean()), 8.0)
            self.assertGreater(res.frame0_max_delta_from_source, 0,
                               "a frame of zero motion would mean the loop stalls")
            # background sampled per frame AND fully re-checked on the last frame
            self.assertIn("sampled pixels", res.background_check_mode)
            self.assertTrue(res.silhouette_lock)
            self.assertTrue(os.path.exists(os.path.join(td, "matte.png")))
            with open(os.path.join(td, "report.json"), "w") as fh:
                json.dump({"ok": True}, fh)

    def test_render_matches_independent_reference(self):
        """Cross-check a rendered frame against a full-frame reference warp.

        The reference deliberately uses no cropping and no work-box arithmetic,
        so any mix-up between absolute and crop-relative coordinates (which once
        silently clamped whole regions of the subject to the work-box edge)
        shows up here.
        """
        img = synthetic_scene(200, 260)
        frames, fps = 12, 12
        with tempfile.TemporaryDirectory() as td:
            ma.render(img, ma.MatteParams(), ma.MotionParams(), [], frames=frames, fps=fps,
                      seed=3, out_mp4=None, out_webp=None, debug_dir=td)
            got1 = np.asarray(Image.open(os.path.join(td, "frame_0001.png")))
            got0 = np.asarray(Image.open(os.path.join(td, "frame_0000.png")))

            src = img.astype(np.float32)
            mp = ma.MatteParams()
            raw = ma.suppress_speckle(ma.raw_matte(src, mp), mp.despeckle)
            matte = ma.blur_f(raw, mp.feather)
            support = raw > 0.5
            layer, _, _ = ma.run_chroma(src, mp)
            model = ma.MotionModel(src, matte, ma.MotionParams(), [], seed=3)

            h, w = matte.shape
            xs = np.ones((h, 1), np.float32) * np.arange(w, dtype=np.float32)[None, :]
            ys = np.arange(h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)

            def reference(t):
                # displacement() is cropped to the work box: expand it back to
                # full-frame fields so the reference needs no cropping at all
                dxc, dyc, _ = model.displacement(t)
                dx = np.zeros((h, w), np.float32)
                dy = np.zeros((h, w), np.float32)
                dx[model.wy0:model.wy1, model.wx0:model.wx1] = dxc
                dy[model.wy0:model.wy1, model.wx0:model.wx1] = dyc
                wrgb = ma.sample_bilinear(layer, dx, dy, xs, ys)
                wm = np.clip(ma.sample_bilinear(matte, dx, dy, xs, ys), 0.0, 1.0)
                alpha = np.where(support, np.minimum(wm, matte), 0.0)[..., None]
                out = np.clip(np.rint(wrgb * alpha + layer * (1 - alpha)), 0, 255)
                out = np.where(support[..., None], out, src)
                return out.astype(np.uint8)

            # t = 0 is a micro-motion of the photograph: close, not identical
            d0 = np.abs(got0.astype(np.int16) - img.astype(np.int16))
            self.assertLess(float(d0.mean()), 8.0, "frame 0 drifted too far from the still")
            dref = np.abs(got0.astype(np.int16) - reference(0.0).astype(np.int16))
            self.assertLess(int(dref.max()), 3,
                            "the reference warp must agree with the renderer at t = 0")
            # frame 1 of the loop must match the independent full-frame reference
            diff = np.abs(got1.astype(np.int16) - reference(1.0 / frames).astype(np.int16))
            self.assertLess(int(diff.max()), 3,
                            f"rendered frame diverges from the reference warp: max {diff.max()}")

    def test_output_size_and_dtype(self):
        img = synthetic_scene(120, 150)
        res = ma.render(img, ma.MatteParams(), ma.MotionParams(), [], frames=4, fps=8,
                        seed=1, out_mp4=None, out_webp=None)
        self.assertEqual((res.height, res.width), (150, 120))
        self.assertEqual(res.frames, 4)
        self.assertEqual(len(res.subject_bbox), 4)

    def test_no_subject_raises(self):
        blank = np.zeros((40, 40, 3), np.uint8)
        blank[:, :] = GREEN
        with self.assertRaises(SystemExit):
            ma.render(blank, ma.MatteParams(), ma.MotionParams(), [], frames=4, fps=8,
                      out_mp4=None, out_webp=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
