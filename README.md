# DrawBuddy Paint

DrawBuddy draws an exact-color source image in Microsoft Paint using mouse-driven Pencil contours and Bucket fills. It does not paste bitmaps into Paint.

## Requirements

Windows, Microsoft Paint, Python, and Python packages `pyautogui`, `opencv-python`, `numpy`. Open a blank white maximized Paint canvas.

## Draw any source image

```powershell
python -u main.py --source-image assets/house-reference.png --speed 50
```

Use any supported image path with `--source-image`. The drawing area is limited to **550 pixels wide × 500 pixels high**. Larger source images are automatically scaled down proportionally to fit both bounds, without cropping or upscaling smaller images. Scaling uses OpenCV INTER_AREA; it may introduce blended RGB colors and lose source-resolution detail. Images already within the bounds retain their original pixel values. There is no 16-color palette limit; more distinct colors may greatly increase planning and drawing time. Screenshot comparison checks the actual Paint result against the **scaled drawing target**, not the original high-resolution image.

## Optional logs

By default, no drawing plans or diagnostic files are saved. Screenshot comparison and strict 1:1 validation still run.

```powershell
python -u main.py --source-image assets/house-reference.png --speed 50 --logs
```

Each run with `--logs` writes its generated plan, comparison images, reports, and timing diagnostics to its own `output/YYYY-MM-DD_HH-MM-SS_microseconds/` directory. Earlier runs are not overwritten. `assets/` contains source images only and is never used for logs.

## Project files

- `assets/`: source images, including the sample house.
- `main.py`: command-line entry point and contour/Bucket drawing pipeline.
- `compare_images.py`: Paint screenshot comparison and strict pixel validation.
- `drawbuddy/`: Paint controls and drawing components.
- `output/`: locally generated optional diagnostics, excluded from Git.
- `AGENTS.md`: drawing method and fidelity requirements.

The public drawing interface consists only of `--source-image`, `--speed`, and `--logs`. Internal stroke-plan processing is not a separate CLI mode.

## Updating from GitHub

```powershell
git pull --ff-only origin main
```
