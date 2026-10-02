# micro_anim — green-screen still → seamless micro-animation loop

A self-contained renderer that turns **one photograph shot against a green screen** into a short,
seamlessly looping "living photograph" video.

It exists because image-to-video models *regenerate* every pixel: identity, wardrobe, colour and the
green screen all drift. `micro_anim` never invents a pixel — it only **moves the pixels that are
already there**. That makes the locks in the prompt spec
([`../prompts/`](../prompts/)) structural rather than hoped-for.

```
python renderer/micro_anim.py --input photo.png --out out/loop --frames 120 --fps 30
→ out/loop.mp4  out/loop.webp  out/loop.report.json
```

---

## What it guarantees (and proves, every run)

| Guarantee | How it is enforced | How it is verified |
| --- | --- | --- |
| **Green screen is untouched** | A chroma-key mask decides which pixels may be written; everything else is copied from the source | Every frame: 50 000 background pixels sampled and compared. Final frame: **all** of them. Any difference fails the run |
| **Seamless loop** | Every motion term is a sinusoid with an integer number of cycles per loop, so frame *N* is the natural continuation of frame *N−1* | `loop_wrap_step` is reported and must not be an outlier against the per-frame step |
| **No camera movement** | There is no camera model at all. The displacement field is identity outside the subject, so zoom / pan / crop / perspective change are *impossible by construction* | `motion_px_max` inside the subject stays in the 0.2–12 px band |
| **No added or removed elements** | Only the subject's own pixels are resampled | `motion_subject_peak` confirms the frame is not frozen |
| **Motion never stalls** | Components use different phases; if they shared one, all of them would vanish together and the animation would visibly pause twice per loop | `motion_px_min` must stay above 15 % of the peak |

`report.json` carries all of these numbers, and the process exits non-zero if a check fails, so it
drops straight into a CI step or a batch job.

## What it animates

* **Breathing** — a smooth rise-and-fall bump centred on the chest (fading towards the hips and the
  arms, because that is where a real breath moves), with the head and shoulders riding a slightly
  lagged secondary term.
* **Hair** — curl groups sway with soft physics; a band-limited noise field de-synchronises
  neighbouring strands so it never reads as one rigid warp. Motion is strongest on the loose
  silhouette and calmer on the scalp.
* **Fabric** — the garment and any strings drift on their own slower cycle.
* **Jewellery / accessory zones** — hand-placed, see below.
* **Silhouette discipline** — `--edge-lock` blends between a pinned outline (default `0.5`, so hair
  edges stay put while the interior moves) and a free outline.

## Motion zones (earrings, bracelets, lens glints)

Anything needing its own pivot can be placed by hand in a JSON file (see
[`../examples/zones.example.json`](../examples/zones.example.json)):

| Mode | Effect | Use it for |
| --- | --- | --- |
| `rotate_top` | Rigid swing around the **top edge** of the rect | Hoop earrings, dangling charms — put the rect's top edge where the jewellery hangs from |
| `sway` | Soft horizontal drift plus a small vertical bob | Bracelets, pendants, loose straps |
| `glint` | A faint sweep of brightness, no geometry change (the only non-warp effect) | Sunglass lenses, polished metal — keep `amplitude` ≤ 1 |

```bash
python renderer/micro_anim.py -i photo.png -o out/loop --zones examples/fixture_zones.json
```

## Tuning

| Flag | Meaning |
| --- | --- |
| `--amplitude` | Global motion scale. `1.0` is the calibrated default; `0.5` is barely-there, `1.5` is the upper end of "subtle" |
| `--breath`, `--hair`, `--fabric`, `--sway` | Per-channel multipliers (`0` to switch a channel off) |
| `--edge-lock` | `1.0` pins the silhouette completely, `0.0` lets the outline float freely |
| `--key-low`, `--key-high` | Chroma-key thresholds on the normalised green ratio (defaults `0.10` / `0.22`) |
| `--grow` | How far subject colours are extended past the silhouette (px). Raise it if the subject has very fine detail, e.g. flyaway hair, against the screen — at a resolution cost |
| `--allow-bg-motion` | Opt out of the bit-exact background guarantee and let the silhouette breathe 2–3 px past its outline. The report then says the background was not bit-exact |
| `--frames`, `--fps` | Loop length. Keep the default 120 @ 30 fps (4 s) — longer clips need higher cycle counts, not bigger amplitudes |
| `--debug-dir`, `--debug-frames` | Writes `matte.png`, a motion heat map, `frame_last.png` and any frame indices you list |

## How the chroma key works

`(G − max(R, B)) / (G + max(R, B))`, thresholded with a linear ramp. Normalising is what makes it
robust: a real studio screen is never one flat colour — vignettes, uneven lighting and shadow
gradients are normal — but the *ratio* stays almost constant, while a raw channel difference does
not. On an uneven test screen the naive metric left ~540 000 semi-transparent pixels on the
background; this one leaves ~2 400, i.e. just the intended feather at the silhouette.

## Honest limits

This renderer moves existing pixels. It cannot invent new ones, so:

* **It cannot blink, move lips, or add a new specular highlight to an eye or lens.** A blink means
  painting an eyelid that is not in the photograph; anything that fakes it looks wrong. Only
  *existing* highlights can shift, and `glint` zones can sweep an existing lens highlight.
* **The silhouette never grows.** In the default strict mode the subject can only move *within* its
  own outline. Use `--allow-bg-motion` if you would rather have edge movement than a bit-exact
  background.
* **Motion is a warp, not a simulation.** Secondary motion (hair, earrings, fabric) is procedural,
  tuned to look like physics — it is not a physical simulation of individual strands.
* **Frame 0 is a micro-motion of the still**, not bit-identical to it (mean deviation is reported,
  typically well under 1 % of the range). That is the price of an animation that never freezes; a
  bit-exact frame 0 would force every motion term to be zero at the same instant, which is exactly
  the stall the report checks for.

For face micro-motion (blinks, lips) pair this with a hosted image-to-video model using the prompt
spec in [`../prompts/`](../prompts/) — or accept that those specific motions are out of scope here.

## Requirements

```bash
python -m pip install -r renderer/requirements.txt
```

`ffmpeg` is used for the MP4; if it is not on your `PATH`, the bundled
[`imageio-ffmpeg`](https://pypi.org/project/imageio-ffmpeg/) binary is used automatically
(`--no-mp4` skips the video entirely). Tests need `pytest`:

```bash
python -m pytest tests -q      # 18 tests, no ffmpeg required
```

Typical cost: a 1024 × 1536 still, 120 frames, ≈ 1.5 min on one CPU core, ≈ 40 MB peak RAM.
