"""Separate visible dark ink from color regions without treating every color seam as a stroke.

Outputs diagnostic layers only. Original pixel ownership is unchanged.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scene_strokes import export_strokes


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
    # Trace ink from the ORIGINAL image, not the k-means palette.
    # Quantization can erase thin toe separators, mouth shadows and
    # small markings before the skeletonizer ever sees them.
    source_path = Path(scene['source'])
    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_bgr is None or source_bgr.shape[:2] != labels.shape:
        raise ValueError('Original scene source is missing or dimensions changed')
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    lum = (0.2126 * source_rgb[:, :, 0] +
           0.7152 * source_rgb[:, :, 1] +
           0.0722 * source_rgb[:, :, 2])
    spread = (source_rgb.max(axis=2).astype(np.int16) -
              source_rgb.min(axis=2).astype(np.int16))
    # Preserve faint anti-aliased ink while excluding saturated color.
    ink_raw = np.uint8((lum < 155) & (spread < 100)) * 255
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        ink_raw, connectivity=8)
    kept = np.zeros(count, dtype=np.uint8)
    if count > 1:
        # Two-pixel details are meaningful at the dog's eyes and paws.
        kept[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 2).astype(np.uint8)
    ink = kept[components] * 255
    # Source fidelity is measured against raw detected ink; removed noise
    # is explicit rather than silently lost during color quantization.
    missing_source_ink = int(np.count_nonzero((ink_raw != 0) & (ink == 0)))
    centerline = _skeleton(ink)
    stroke_report = export_strokes(ink, centerline, out)

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
        'ink_luminance_threshold': 155,
        'ink_chroma_threshold': 100,
        'min_ink_component_pixels': 2,
        'ink_detection_source': 'original_rgb_before_palette_quantization',
        'raw_detected_ink_pixels': int(np.count_nonzero(ink_raw)),
        'source_ink_pixels_removed_as_noise': missing_source_ink,
        'source_ink_retention_ratio': round(float(np.count_nonzero(ink)) / max(1, np.count_nonzero(ink_raw)), 6),
        'ink_pixels': int(np.count_nonzero(ink)),
        'centerline_pixels': int(np.count_nonzero(centerline)),
        'excluded_dark_noise_pixels': int(np.count_nonzero(ink_raw & ~ink)),
        'layer_reconstruction_mismatch_pixels': mismatch,
        'color_region_boundaries_are_not_strokes': True,
        'ink_centerlines_are_diagnostic_only': True,
        'vector_strokes_validated': False,
        'stroke_reconstruction': stroke_report,
        'paint_ready': False,
    }
    (out / 'layer-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
