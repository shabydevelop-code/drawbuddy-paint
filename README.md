# DrawBuddy Paint

DrawBuddy draws a source image in Microsoft Paint using real mouse-driven Pencil contours and Bucket fills. The drawing pipeline accepts any supported source image. It does not paste bitmaps into Paint.

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

## Existing 16-color plan workflow

Generate a 16-color plan from a source image:

```powershell
python pixel_map.py assets/house-reference.png --width 430 --height 330 --paint-plan-16
```

Draw using the established contour/Bucket engine:

```powershell
python -u main.py --image-bucket output/pixel-map/pixel-map-paint-plan16.json --speed 50
```

The plan uses image-derived colors and therefore approximates a reference with more than 16 colors. Inspect the generated preview and compare the real Paint result with the intended image before claiming fidelity.

## Files

- `assets/`: reference images (the drawing engine is image-independent).
- `main.py`: entry point and established contour/Bucket implementation.
- `pixel_map.py`: pixel-map and 16-color plan generation.
- `drawbuddy/`: Paint mouse controller, drawing components, and legacy experimental painters.
- `output/`: locally regenerated plans, previews, comparisons and diagnostic captures.
- `diagnostics/latest/`: Git-tracked diagnostic snapshots, retained for remote review.
- `AGENTS.md`: mandatory drawing and fidelity rules.

The entry point now exposes only `--house`, `--source-image`, and `--image-bucket`. Legacy experimental modules remain in the repository pending dependency verification; they are not invoked by these modes. Do not treat offline expected images as proof of Paint execution.

## Updating from GitHub

```powershell
git pull --ff-only origin main
```
