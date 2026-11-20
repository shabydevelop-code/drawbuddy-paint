# DrawBuddy — Microsoft Paint automation

## Run

Open classic Microsoft Paint maximized, Home tab visible, and a new blank canvas.

- `python .\main.py dog-outline` — **new** diagnostic: one continuous smooth dog silhouette drawn by the real Paint Pencil tool, using actual mouse movements. No image import, no paste, no fill.
- `python .\main.py dog` — previous 20-step geometric dog, unchanged.

Do not touch the mouse or keyboard during drawing. Move the cursor to a screen corner to trigger PyAutoGUI failsafe if needed.

## Drawing speed

- `python .\main.py dog-outline --speed 2` — twice the default Pencil stroke speed.
- `python .\main.py dog-outline --speed 0.5` — half speed.
- `config.json` contains `drawing_speed` (default `1.0`), used when `--speed` is omitted.
- Allowed speed range: greater than 0 through 20. Command-line option overrides config.
- Speed applies **only to continuous Pencil strokes**. Tool/color selection, shape drags,
  UI focus and verification delays stay unchanged for reliability.
- High speeds may produce less consistent Paint strokes depending on Windows input processing.

## Architecture

- `drawings/*.json`: drawing instructions; no UI coordinates for tools.
- `drawbuddy/drawing_engine.py`: validates and executes shape and stroke commands; interpolates smooth paths.
- `drawbuddy/elements.py`: single registry of Paint UI tool coordinates.
- `drawbuddy/components.py`: tools, colors, canvas.
- `drawbuddy/controller.py`: window focus, mouse, keyboard, pixel sampling.

`stroke` accepts `color`, `points` and optional `smooth: true`. The Pencil tool is at (244,70), based on the existing Paint UI calibration. This new operation has not yet been tested against the user's Windows Paint. Existing drawing and color selection routines are unchanged.

`paint-diagnostic.png` is saved after each run, including failures. The new silhouette intentionally has no fill; closed-path filling and richer illustration are later stages.

## Filled continuous silhouette

Open a new white canvas in classic Paint, then run:

```powershell
python .\main.py dog-filled --speed 4
```

The `closed_path` tool draws a Pencil outline, closes it, selects the bucket, fills at `fill_point`, and checks both the fill pixel and an optional white `outside_point` for leakage. `dog-outline` remains unchanged. The canvas must be large enough for all points.

JSON step example: `{"name":"body","tool":"closed_path","color":"black","fill":"orange","smooth":true,"points":[[300,300],[500,300],[500,450],[300,450]],"fill_point":[400,375],"outside_point":[550,500]}`.

## Detailed dog (Paint UI only)

Run on a new blank canvas: `python .\main.py dog-detailed --speed 4`

The drawing first creates and fills the validated orange silhouette, then adds a brown ear and muzzle, black eye and nose, and contour details. Each operation is performed with actual Paint tools and mouse movements; no image is pasted.

