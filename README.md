# DrawBuddy Paint

DrawBuddy draws an exact-color source image in Microsoft Paint using mouse-driven Pencil contours and Bucket fills. It does not paste bitmaps into Paint.

## Requirements

Windows, Microsoft Paint, Python, and Python packages `pyautogui`, `opencv-python`, `numpy`, `Pillow`. Open a blank white maximized Paint canvas.

## Draw any source image

```powershell
python -u main.py --source-image assets/house-reference.png --speed 50 --colors 50
```

Use any supported image path with `--source-image`. The drawing area is limited to **550 pixels wide × 500 pixels high**. Larger source images are automatically scaled down proportionally to fit both bounds, without cropping or upscaling smaller images. Scaling uses OpenCV INTER_AREA; it may introduce blended RGB colors and lose source-resolution detail. After scaling, the image is quantized using Median Cut to at most **50 RGB colors by default**, configurable with `--colors N` (2–256). Pure black and white pixels are preserved when present and count toward the limit. Quantization changes some RGB values and may lose fine color details; the original file is not modified. More colors may greatly increase planning and drawing time. Screenshot comparison checks the actual Paint result against the **scaled, quantized drawing target**, not the original image.

## Diagnostic preview without drawing

```powershell
python -u main.py --source-image assets/dog-reference.png --speed 50 --colors 50 --preview-only
```

This saves `output/preview.png` (overwritten on the next preview) showing the exact image after proportional resize and color quantization. Paint is not launched and no drawing plan is generated. Compare the preview with the Paint result to distinguish preprocessing defects from drawing defects.

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

The public drawing interface consists only of `--source-image`, `--speed`, `--colors`, `--preview-only`, and `--logs`. Internal stroke-plan processing is not a separate CLI mode.

## Updating from GitHub

```powershell
git pull --ff-only origin main
```
