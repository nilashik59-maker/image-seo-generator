# image-seo-generator

Tools for turning a single source image into a complete, well-described asset — including an
image-to-video prompt package and a renderer that animates a green-screen still without
regenerating a single pixel.

## Contents

| Path | What it is |
| --- | --- |
| [`prompts/`](prompts/README.md) | **Image → video prompt package** for animating a green-screen portrait with the camera locked: structured JSON spec, plain-prose prompt, negative prompt, shot parameters and a QC checklist |
| [`renderer/`](renderer/README.md) | **`micro_anim`** — a self-contained Python renderer that turns a green-screen still into a seamlessly looping micro-animation MP4 + animated WebP, with runtime proof that the background is untouched and the loop is continuous |
| [`examples/`](examples/) | A green-screen test photograph, example motion zones, and the rendered demo loop |
| [`tests/`](tests/) | 18 tests covering the chroma key, the resampler, the motion model and the end-to-end render |

## Quick start

```bash
python -m pip install -r renderer/requirements.txt

# render a 4 s seamless loop from any green-screen still
python renderer/micro_anim.py --input examples/fixture_green_screen.jpg \
    --out out/loop --frames 120 --fps 30 --zones examples/fixture_zones.json

python -m pytest tests -q
```

Full documentation: [`renderer/README.md`](renderer/README.md) ·
Prompt spec: [`prompts/README.md`](prompts/README.md)
