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
