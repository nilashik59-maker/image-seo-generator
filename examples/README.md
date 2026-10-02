# Examples

| File | What it is |
| --- | --- |
| `fixture_green_screen.jpg` | A synthetic green-screen test photograph used by the demo and the tests. Not a real person — it stands in for a shoot frame so the renderer can be exercised end to end. |
| `fixture_zones.json` | Hand-placed motion zones for that fixture: two hoop earrings (`rotate_top`), the bracelets on the raised arm (`sway`) and a lens glint (`glint`). |
| `zones.example.json` | The annotated template for writing your own zones, including what each mode does. |
| `out/` | Render output (git-ignored): `fixture_loop.mp4`, `fixture_loop.webp`, `fixture_loop.report.json`, `debug/`. |

## Reproduce the demo loop

```bash
python -m pip install -r renderer/requirements.txt
./renderer/demo.sh                       # or PYTHON=/path/to/venv/bin/python ./renderer/demo.sh
```

It writes `examples/renders/fixture_loop.mp4` (4 s, 120 frames, 30 fps, seamless), a 720 px animated
WebP preview, and a report whose checks all pass.

## Use your own photograph

Any green-screen (chroma) still works:

```bash
python renderer/micro_anim.py --input my_photo.jpg --out out/my_loop --frames 120 --fps 30
```

1. Run once with `--debug-dir out/dbg` and look at `dbg/matte.png`. Hair, jewellery and dark
   clothing should be white (subject) and the whole screen black. Adjust `--key-low` /
   `--key-high` if not.
2. Place a zone file next to it for earrings, bracelets or lens glints. Coordinates are pixels;
   put a `rotate_top` rect's top edge where the jewellery hangs from.
3. Re-render and read `out/my_loop.report.json` — if `checks_passed` is false, the reasons are in
   `notes`.

The fixture's own zones live in `fixture_zones.json` and use the 1024 × 1536 frame of the fixture;
for a different image aspect those coordinates need to be moved.
