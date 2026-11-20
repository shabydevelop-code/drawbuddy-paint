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

    # Compare actual source ink with reconstructed stroke ink at the SAME
    # resolution used by Paint. This exposes losses hidden by source-scale IoU.
    rebuilt_bgr = cv2.imread(str(out / 'stroke-reconstruction.png'))
    if rebuilt_bgr is None or rebuilt_bgr.shape[:2] != ink.shape:
        raise RuntimeError('Stroke reconstruction preview is missing or invalid')
    rebuilt_ink = np.uint8(np.all(rebuilt_bgr < 128, axis=2))
    target_w, target_h = 360, 270
    factor = min(target_w / ink.shape[1], target_h / ink.shape[0], 1.0)
    shape = (max(1, round(ink.shape[1] * factor)),
             max(1, round(ink.shape[0] * factor)))
    source_small = cv2.resize(np.uint8(ink > 0), shape,
                              interpolation=cv2.INTER_AREA) > 0.20
    rebuilt_small = cv2.resize(rebuilt_ink, shape,
                               interpolation=cv2.INTER_AREA) > 0.20
    lost = source_small & ~rebuilt_small
    added = rebuilt_small & ~source_small
    overlay = np.full((shape[1], shape[0], 3), 255, np.uint8)
    overlay[source_small & rebuilt_small] = (30, 30, 30)
    overlay[lost] = (0, 0, 255)  # red: source ink not reconstructed
    overlay[added] = (255, 0, 0)  # blue: reconstructed outside source
    cv2.imwrite(str(out / 'stroke-difference.png'), overlay)
    missing_count, missing_labels, missing_stats, _ = (
        cv2.connectedComponentsWithStats(np.uint8(lost), connectivity=8))
    missing_regions = []
    for i in range(1, missing_count):
        x, y, w, h, area = map(int, missing_stats[i])
        if area >= 2:
            missing_regions.append({'x': x, 'y': y, 'width': w,
                                    'height': h, 'pixels': area})
    missing_regions.sort(key=lambda item: -item['pixels'])
    target_fidelity = {
        'resolution': list(shape),
        'source_ink_pixels': int(np.count_nonzero(source_small)),
        'reconstructed_ink_pixels': int(np.count_nonzero(rebuilt_small)),
        'missing_ink_pixels': int(np.count_nonzero(lost)),
        'extra_ink_pixels': int(np.count_nonzero(added)),
        'iou': round(float(np.count_nonzero(source_small & rebuilt_small)) /
                     max(1, np.count_nonzero(source_small | rebuilt_small)), 6),
        'largest_missing_regions': missing_regions[:20],
        'note': 'Offline reconstructed strokes at Paint scale, not a Paint screenshot',
    }
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
        'paint_scale_stroke_fidelity': target_fidelity,
        'paint_ready': False,
    }
    (out / 'layer-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
