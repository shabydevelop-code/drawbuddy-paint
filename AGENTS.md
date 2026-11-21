# DrawBuddy Paint — Mandatory Project Rules

These rules apply to every change in this repository. Read this file before modifying drawing code. If a proposed change conflicts with a rule, stop and ask the user for explicit approval; do not silently substitute a different drawing method.

## Mission

Draw a recognizable, complete, accurately colored dog in **Microsoft Paint by physically controlling the mouse and Paint tools**. The output is a human-like drawing process, not a bitmap transfer or a raster printer. Preserve the reference's outlines, shapes, colors, and small details. The final Paint canvas must have no unintended white gaps between adjacent colored or outlined cells.

## Non-negotiable drawing workflow

1. **Outer contours first.** Trace the dog's exterior silhouette with meaningful, continuous pencil/line paths: ears, head, body, legs, paws, and tail. Do not start with interior colors or scanline strokes.
2. **Inner contours second.** Draw the structural boundaries separating color regions and facial details. Ensure boundaries are closed where Paint Bucket requires closure.
3. **Bucket fills third.** Use Paint's Bucket to fill bounded regions, choosing seeds inside the correct region. Respect nesting, holes, and the order of overlapping regions.
4. **Local pixel repairs last.** Use Pencil only for thin areas, tiny features, and verified remaining gaps that cannot be safely filled by Bucket. Repairs must be local and proportionate, not a substitute for the drawing algorithm.
5. **Verify the result.** Compare the actual Paint canvas against the source and investigate visible defects. A perfect simulated raster is NOT proof that Paint drew correctly.

## Forbidden regressions

- **No printer method:** no per-pixel, row-by-row, serpentine, DFS component coverage, or bulk raster scanning as the primary drawing method, even if it simulates to zero mismatches.
- **No bitmap insertion:** no clipboard image/DIB paste, image import, or direct canvas replacement to produce the drawing.
- Do not replace contour-and-bucket drawing with full-image Pencil coverage to bypass open boundaries, holes, or fill failures.
- Do not erase earlier drawing stages by filling an enclosing contour over already completed inner regions without a validated layering strategy.
- Do not add arbitrary stroke-count limits, remove limits, or change the method merely to get past a preflight failure. Diagnose the actual issue.
- Do not declare a defect fixed based only on source inspection, a planning simulation, or a successful Git commit.

## Continuous contour color discipline (mandatory)

- **Never sample the source color at every contour pixel to decide the mouse stroke color.** A contour is a geometric path, not a sequence of independently colored pixels.
- **Outer silhouette:** select the intended outline/ink color once, then trace each continuous outer contour as a continuous mouse stroke. Do not break it into tiny strokes or switch colors at source-pixel boundaries.
- **Inner boundaries:** plan continuous paths and select a stable, explicitly planned color per boundary/path or region; never switch colors per pixel or per contour point.
- **Color changes are permitted only between meaningful drawing operations** (e.g. distinct contours, region boundaries, or Bucket fills), not during a single contour.
- Before committing, inspect the actual drawing loop for point-by-point RGB sampling, repeated color selection inside contour-point loops, and accidental fragmentation of paths. Reject such changes as regressions.
- If a boundary has multiple source colors, resolve the contour/region ownership in the planner; do not solve it by switching Paint colors on every pixel.

## Correctness requirements

- Source color masks and intended canvas coordinates must be handled consistently.
- Contours with holes or nested islands require explicit hierarchy-aware planning; an external-only contour is not sufficient evidence of a safe fill.
- A Bucket operation is allowed only when its seed and expected filled footprint have been validated to avoid spilling into other regions.
- Pencil contours must actually close on the Paint pixel grid; visually close is not enough.
- Black/near-black outlines must not be mistaken for large solid regions simply because they form connected components.
- Do not intentionally omit source details or leave unintended white pixels.
- Preflight must reject provably unsafe fills. Runtime screenshots/comparisons are needed to validate real behavior.

## Change discipline

- Before coding, identify which drawing stage is broken and why.
- Prefer the smallest root-cause fix that preserves the established workflow.
- Review the resulting drawing sequence, not just the final pixel coverage.
- Keep existing canonical paths and filenames; update files in place instead of creating version-suffixed copies.
- Use the existing `main` branch and GitHub repository `shabydevelop-code/drawbuddy-paint`.
- Document durable operational commands and recurring errors in existing documentation as appropriate, not a running status log.
- When reporting work, state what was changed, what was actually tested, and what remains unverified. Do not claim success without an actual Paint run and visual/result comparison.

## Exact 1:1 fidelity to the source (mandatory)

- **The reference image is the single source of truth.** The drawing must reproduce it **1:1**, not merely look similar or recognizable.
- Preserve every original shape, outline, boundary, proportion, position, color, and detail. No artistic interpretation, simplification, smoothing, stylization, or approximations unless the user explicitly approves them.
- Preserve source dimensions and pixel alignment in the intended Paint canvas; do not silently resize, crop, shift, or change the palette.
- The target is **zero differing pixels** between the captured Paint result and the source at the agreed pixel coordinates, with no missing white/colored/black pixels or unexpected marks.
- If Paint's actual drawing behavior prevents exact fidelity, report the measured differences and limitations; do not declare 1:1 success.
- Contour-and-Bucket mouse drawing remains mandatory even when pixel-perfect reproduction is challenging. Exactness does not authorize switching to printer-style raster drawing or bitmap pasting.

## Acceptance criteria

1. A visible Paint drawing starts with continuous **outer dog contours**, then internal contours.
2. Large bounded regions are colored with **Bucket**, not raster printing.
3. No unintended white gaps, missing areas, or color spills in the completed dog.
4. The mouse is responsible for all drawing operations; no image pasting.
5. A screenshot or equivalent captured Paint result is checked against the reference.
6. Any future change must preserve criteria 1–5 unless the user explicitly authorizes an exception.

**Priority:** Preserve the drawing method and visual fidelity before optimizing speed or stroke count. A faster incorrect method is a regression.
