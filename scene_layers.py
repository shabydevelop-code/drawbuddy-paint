"""Separate visible dark ink from color regions without treating every color seam as a stroke.

Outputs diagnostic layers only. Original pixel ownership is unchanged.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _skeleton(binary):
    """Morphological centerline, independent of optional opencv-contrib."""
    remaining = binary.copy()
    result = np.zeros_like(remaining)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(remaining):
        eroded = cv2.erode(remaining, kernel)
        opened = cv2.dilate(eroded, kernel)
        result = cv2.bitwise_or(result, cv2.subtract(remaining, opened))
        remaining = eroded
    return result


def export_visual_layers(scene, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(scene['labels'], dtype=np.uint8)
    palette = np.asarray(scene['palette'], dtype=np.uint8)
    rgb = palette[labels]
    # Darkness is a property of the rendered color, not a palette index.
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] +
           0.0722 * rgb[:, :, 2])
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    # Neutral dark outlines; saturated brown/orange fills are not ink.
    ink_raw = np.uint8((lum < 105) & (spread < 100)) * 255
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        ink_raw, connectivity=8)
    kept = np.zeros(count, dtype=np.uint8)
    if count > 1:
        kept[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 10).astype(np.uint8)
    ink = kept[components] * 255
    centerline = _skeleton(ink)

    # Color-only preview uses original palette values; ink pixels are masked
    # in white, so color boundaries are not mistakenly drawn as outlines.
    color_layer = rgb.copy()
    color_layer[ink != 0] = (255, 255, 255)
    ink_layer = np.full_like(rgb, 255)
    ink_layer[ink != 0] = (20, 20, 20)
    stroke_layer = np.full_like(rgb, 255)
    stroke_layer[centerline != 0] = (20, 20, 20)
    combined = rgb.copy()
    combined[ink != 0] = (20, 20, 20)

    def save(name, image):
        cv2.imwrite(str(out / name),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    save('color-regions.png', color_layer)
    save('ink-mask.png', ink_layer)
    save('ink-centerlines.png', stroke_layer)
    save('layered-preview.png', combined)

    # Mask-level reconstruction validation, not vector or Paint validation.
    reconstructed = color_layer.copy()
    reconstructed[ink != 0] = (20, 20, 20)
    expected = combined
    mismatch = int(np.count_nonzero(np.any(reconstructed != expected, axis=2)))
    report = {
        'method': 'dark-neutral-ink-separation',
        'ink_luminance_threshold': 105,
        'ink_chroma_threshold': 100,
        'min_ink_component_pixels': 10,
        'ink_pixels': int(np.count_nonzero(ink)),
        'centerline_pixels': int(np.count_nonzero(centerline)),
        'excluded_dark_noise_pixels': int(np.count_nonzero(ink_raw & ~ink)),
        'layer_reconstruction_mismatch_pixels': mismatch,
        'color_region_boundaries_are_not_strokes': True,
        'ink_centerlines_are_diagnostic_only': True,
        'vector_strokes_validated': False,
        'paint_ready': False,
    }
    (out / 'layer-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
