# DrawBuddy Paint

DrawBuddy draws a source image in Microsoft Paint using real mouse-driven Pencil contours and Bucket fills. The current drawing pipeline is shared by the dog and house. It does not paste bitmaps into Paint.

## Requirements

Windows, Microsoft Paint, Python, and Python packages `pyautogui`, `opencv-python`, `numpy`. Start with a blank white Paint canvas and leave Paint maximized.

## House — fixed reference

The canonical image is `assets/house-reference.png`.

```powershell
python -u main.py --house --speed 50
```

## Any exact-color source image

```powershell
python -u main.py --source-image assets/house-reference.png --speed 50
```

This mode preserves source RGB values. The image must be no larger than 500×547 and have at most 16 distinct colors. It rejects images requiring resizing or palette approximation.

## Dog — established plan workflow

Generate the existing 16-color plan from the dog reference:

```powershell
python pixel_map.py assets/dog-reference.png --width 300 --height 328 --paint-plan-16
```

Draw using the established contour/Bucket engine:

```powershell
python -u main.py --full-dog-bucket output/pixel-map/pixel-map-paint-plan16.json --speed 50
```

The plan uses image-derived colors and therefore approximates a reference with more than 16 colors. Inspect the generated preview and compare the real Paint result with the intended image before claiming fidelity.

## Files

- `assets/dog-reference.png` and `assets/house-reference.png`: canonical source images.
- `main.py`: entry point and established contour/Bucket implementation.
- `pixel_map.py`: pixel-map and 16-color plan generation.
- `drawbuddy/`: Paint mouse controller, drawing components, and legacy experimental painters.
- `output/`: locally regenerated plans, previews, comparisons and diagnostic captures.
- `diagnostics/latest/`: Git-tracked diagnostic snapshots, retained for remote review.
- `AGENTS.md`: mandatory drawing and fidelity rules.

Older drawing modes still exist in `main.py` for compatibility; they are experimental and are not the current house/dog workflow. Do not treat offline expected images as proof of Paint execution.

## Updating from GitHub

```powershell
git pull --ff-only origin main
```
