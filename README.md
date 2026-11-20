# DrawBuddy — Paint reference tracing

DrawBuddy traces linework from `assets/dog-reference.png` and draws it in classic Microsoft Paint using real mouse movements. It does not paste the reference into Paint.

## Requirements

Windows, Python, classic Paint, and `pyautogui`, `opencv-python`, `numpy` (plus existing project dependencies).

## Run

Open Paint maximized with a **new blank canvas** and the Home tab visible.

```powershell
python -m pip install opencv-python numpy
python trace_reference.py assets/dog-reference.png
python main.py dog --speed 2
```

The generator writes `drawings/dog.json` and `drawings/dog-audit.json`. These are **generated working outputs**, not versioned drawing history. Regenerate them after pulling changes. The audit predicts enclosed regions but **does not guarantee that Paint's bucket will not leak**; inspect the actual drawing before coloring.

## Architecture

- `trace_reference.py`: image ink mask, Zhang–Suen skeletonization, pixel-path tracing, source width analysis, conservative short-bridge repair, visual stroke order, raster connectivity audit, and JSON export; separated into OOP classes.
- `drawbuddy/drawing_engine.py`: validated JSON execution.
- `drawbuddy/components.py`: Paint tools, colors, stroke sizes and canvas.
- `drawbuddy/controller.py`: actual mouse and keyboard interaction.
- `drawbuddy/elements.py`: Paint UI coordinate registry.
- `assets/dog-reference.png`: canonical source illustration.

Move the mouse to a screen corner for PyAutoGUI failsafe. The final drawing screenshot is saved as `paint-diagnostic.png`.

There is one active tracing script and no retained version-numbered scripts or legacy dog drawings on the current branch. Git commit history remains available.
