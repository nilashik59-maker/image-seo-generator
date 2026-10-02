# Image → Video: Green-Screen Micro-Animation

A reusable prompt package for animating a **still photograph shot on a chroma-green screen** so it
looks like the photograph came alive — while identity, pose, wardrobe, composition, colour and the
green background stay locked.

| File | Purpose |
| --- | --- |
| [`image-to-video__green-screen-micro-animation.json`](image-to-video__green-screen-micro-animation.json) | Full structured spec (works with models that accept JSON-ish structured prompts, and as the canonical reference). |
| [`negative-prompt.txt`](negative-prompt.txt) | Ready-to-paste negative prompt. |
| `renderer/` (repo root) | Deterministic alternative: renders the same result locally from the still, with **zero** regeneration of pixels. |

---

## 1. Plain-prose prompt (use this in most video models)

> Animate the provided photograph exactly as it is. Lock the camera completely: no zoom, no pan, no
> tilt, no dolly, no shake, no reframe, no change of crop. Keep the woman's identity, face structure,
> body proportions, pose, hairstyle, sunglasses, earrings, bracelets, swimwear design, skin tone and
> body position pixel-for-pixel as in the source frame. Nothing is added and nothing is removed.
>
> Add only subtle, photorealistic micro-motion: slow natural breathing that gently lifts the chest,
> shoulders and torso; tiny head micro-movement; individual curly hair strands and curls settling,
> bouncing and swaying with soft hair physics; both hoop earrings swinging lightly with realistic
> jewellery physics; bracelets shifting subtly with the raised arm; the swimwear fabric and strings
> moving with the breath; extremely subtle natural blinks and lip micro-movement; a faint moving light
> reflection travelling across the existing sunglass lenses; and small natural shifts in skin
> highlights as the body moves.
>
> The bright green chroma background must stay perfectly static — same exact green, no gradient, no
> texture, no light change, no new shadows, no motion. The result must look like the original
> photograph coming alive, not like the image has been recreated, restyled or re-lit. Photorealistic,
> no deformation, no flicker, no warping.

## 2. Shot parameters

| Parameter | Value |
| --- | --- |
| Mode | image-to-video / image-reference (frame 0 = source image) |
| Duration | 4–6 s |
| Resolution | same aspect ratio as the source (never letterbox) |
| Motion strength | **lowest available** (low / subtle / 1–3 of 10) |
| Camera motion | none / locked / static |
| Seed | fixed; re-roll the seed instead of raising motion strength |
| Loop | request seamless loop if the model supports it, otherwise see §4 |

## 3. Negative prompt

Copy [`negative-prompt.txt`](negative-prompt.txt). If the model has a character budget, keep at least
these: `identity change, pose change, hairstyle change, wardrobe change, colour shift, relighting,
camera movement, zoom, pan, crop, extra fingers, warped body, background motion, cartoon, 3d render`.

## 4. Model notes

* **Veo 3 / Veo 3.1 (image-to-video)** — accepts a first-frame image. Feed the prose prompt; keep
  prompt length moderate. "Locked-off camera" phrasing is respected well; ask for a 4 s clip.
* **Kling 2.x / Kling 2.5** — has explicit camera-movement controls; set every axis to *static* and
  set creativity to the lowest value. Kling is strong on hair physics.
* **Runway Gen-4 / Gen-4 Turbo** — use *image to video* with the still as the first frame and
  camera-motion = none. Lower the motion slider; frame-0 fidelity is generally high.
* **Sora / Pika / Hailuo / others** — same rules: still as first frame, lowest motion, no camera
  move. Pika has a "motion 1–4" dial — stay at 1–2 for this look.
* **Seamless loops** — most hosted video models cannot guarantee an exact loop. Generate 4 s, then
  either ping-pong the clip (forward + reversed) or trim to the nearest matching frame. For a
  *guaranteed* perfect loop with bit-exact identity lock, use the local renderer in
  [`../renderer/`](../renderer/README.md).

## 5. QC checklist before you accept a take

- [ ] Frame 0 is indistinguishable from the source still.
- [ ] Sunglasses, earrings and bracelets are unchanged in shape, size and position.
- [ ] Swimwear colour, cut and knot placement are unchanged.
- [ ] No new objects, no missing objects, no extra fingers or limbs.
- [ ] Background green is uniform and identical in every frame — no gradient, no shadows, no drift.
- [ ] No zoom, pan, crop change or perspective shift anywhere in the clip.
- [ ] Hair motion reads as physics, not as a mesh warp; hair stays attached to the scalp.
- [ ] Face never morphs or "breathes" unnaturally; no beauty-filter smoothing appears.
- [ ] Motion is visible when you look for it, invisible when you don't.

## 6. Known limits of hosted video models

Face micro-motion (blinks, lips) and jewellery physics are the most common failure points: models
tend to either freeze them or exaggerate them. If a take fails, prefer lowering motion strength and
changing the seed over rewriting the prompt. Anything that requires inventing new pixels (closing
eyelids, adding specular highlights) will always be approximate — the local renderer avoids the
problem by only moving pixels that already exist.
