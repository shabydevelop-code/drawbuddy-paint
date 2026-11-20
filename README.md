# DrawBuddy — reference drawing in Microsoft Paint

DrawBuddy converts source image linework to a drawing definition and reproduces it in classic Paint using mouse operations. No AI is used at runtime.

## Requirements
Windows, classic Paint, Python, and Python packages `pyautogui`, `opencv-python`, `numpy`.

## Main workflow
Open Paint maximized on a blank canvas, with the Home tab visible.

```powershell
python trace_reference.py assets/dog-reference.png
python main.py dog --speed 2
```

After each successful run, DrawBuddy captures the Paint drawing area and automatically commits and pushes `output/dog.png` to GitHub `main` (requires working Git credentials and network). Use `--no-publish-result` to draw without uploading. The PNG is a screenshot crop of the Paint canvas area, not a Paint Save As export.\n\nThe generated drawing definition is `drawings/dog.json` (local working output, not tracked in Git). Keep your working drawing definitions when cleaning local files.

## Optional working drawing builders
- `build_closure_drawing.py`: conservative closure joins; outputs `drawings/dog-closure-test.json`.
- `build_leak_drawing.py`: validated leak joins with optimized drawing order; outputs `drawings/dog-leak-test.json`.
- `plan_reference_colors.py`: produces a color plan for review, not automatic coloring.

These builders retain their required geometry-validation modules. Do not delete a working drawing definition without checking which version you currently draw.

## Core
- `trace_reference.py`: source tracing and drawing JSON generation.
- `drawing_sequence_planner.py`: stroke ordering.
- `drawbuddy/`: Paint control, drawing engine, result publishing.
- `main.py`: execution.
- `output/dog.png`: latest published Paint screenshot.

Experimental diagnostic scripts and diagnostic images have been removed from the current branch. Git history remains available.

## Experimental binary 0/1 mode

Start with a new blank white Paint canvas. This mode thresholds the original image into a pixel mask (0=white, 1=black), compresses consecutive black pixels into horizontal runs, and draws those runs with Paint's pencil. It does not call the vector tracer or use guessed contours. It is black-and-white only, limited to 240x180 pixels for practical Paint mouse speed, and can take a long time on dense images. It is not yet a full-color or performance-optimized raster engine.

```powershell
python main.py --binary-source assets/dog-reference.png --speed 2
```

Optional: `--binary-threshold 110` controls the dark-pixel cutoff; `--no-publish-result` disables GitHub result upload. The original `python main.py dog-vector --speed 2` remains available and unchanged.

## BinaryPainter V2 (experimental)

`python main.py --binary-source assets/dog-reference.png --speed 2` now uses larger (up to 500x350) connected-region contour + bucket fills, including white nested holes. Legacy line-by-line printing is available with `--binary-runs`. Small/thin regions may be skipped. White palette coordinate is based on the existing Paint ribbon layout and must be verified on the target machine. Use a fresh blank white Paint canvas. This is experimental and not yet validated on the user's Paint installation.
