"""Draw a reference image in Microsoft Paint using contour and Bucket operations."""
import argparse
import json
import math
from pathlib import Path
from datetime import datetime
from drawbuddy.drawing_engine import DrawingEngine
from compare_images import compare_paint_canvas


def draw_image_bucket(plan_path, speed, pixel_finish=False, finish_limit=1500, black_gap_finish=False, black_gap_limit=120, logs=False, output_dir=None):
    """Full image: all color components, closed contours, Paint bucket, no raster strokes."""
    import time
    startup_clock = time.perf_counter()
    import cv2
    import numpy as np
    from drawbuddy.controller import check_cancel
    import_seconds = time.perf_counter() - startup_clock
    stage_clock = time.perf_counter()
    data = (plan_path if isinstance(plan_path, dict)
            else json.loads(Path(plan_path).read_text(encoding='utf-8')))
    read_seconds = time.perf_counter() - stage_clock
    stage_clock = time.perf_counter()
    w, h = data.get('width'), data.get('height')
    if (data.get('format') != 'drawbuddy-paint-stroke-plan-v1'
            or not isinstance(w,int) or not isinstance(h,int)
            or not 1 <= w <= 500 or not 1 <= h <= 547):
        raise ValueError('Invalid 16-color pixel plan')
    # White is implicit canvas background. The stroke plan may omit its runs;
    # do not require one explicit group entry for every pixel.
    source = np.full((h,w,3),255,np.uint8)
    groups = []
    occupied = np.zeros((h,w),np.uint8)
    for group in data['groups']:
        color = group.get('rgb')
        if not isinstance(color,str) or len(color) != 7 or not color.startswith('#'):
            raise ValueError('Invalid RGB color in plan')
        try:
            rgb = tuple(bytes.fromhex(color[1:]))
        except ValueError as exc:
            raise ValueError('Invalid RGB color in plan') from exc
        mask = np.zeros((h,w),np.uint8)
        for x0,y0,x1,y1 in group['strokes']:
            if not 0 <= x0 <= x1 < w or not 0 <= y0 == y1 < h:
                raise ValueError('Invalid run')
            mask[y0,x0:x1+1] = 255
        # Explicit white runs, when present, are background, not paint work.
        if rgb == (255,255,255):
            continue
        overlap = (occupied != 0) & (mask != 0)
        if np.any(overlap):
            raise ValueError(f'Overlapping nonwhite color regions: {color}')
        occupied[mask != 0] = 1
        source[mask != 0] = rgb[::-1]
        groups.append((rgb,mask))
    # Include the implicit white pixels so enclosed white regions can be
    # recovered by the contour/bucket planner when necessary.
    white_mask = np.uint8(occupied == 0)*255
    groups.append(((255,255,255),white_mask))
    mask_seconds = time.perf_counter() - stage_clock
    stage_clock = time.perf_counter()
    # Stage 1: paint broad color surfaces. Stage 2: draw black
    # source pixels LAST, so feet/toe lines are never covered by later fills.
    regions = []
    details = []
    omitted_pixels = 0
    narrow_regions = []
    for rgb,mask in groups:
        n,labels,stats,_ = cv2.connectedComponentsWithStats(mask,8)
        for idx in range(1,n):
            area = int(stats[idx,cv2.CC_STAT_AREA])
            component = np.uint8(labels == idx)*255
            if rgb == (255,255,255):
                if np.any(component[0,:]) or np.any(component[-1,:]) or np.any(component[:,0]) or np.any(component[:,-1]):
                    continue
            contours,_ = cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours,key=cv2.contourArea)
            dist = cv2.distanceTransform(component,cv2.DIST_L2,3)
            _,radius,_,seed = cv2.minMaxLoc(dist)
            is_ink = max(rgb) <= 24
            if is_ink:
                # Both true black and near-black are outline ink, not
                # bucket-fill regions. Their connected components can
                # surround large holes; RETR_EXTERNAL would flood them.
                # The dedicated final ink stage draws these details.
                continue
            # The relevant width is the free interior AFTER a 1-pixel
            # boundary is drawn, not the overall width of the source region.
            # Do not discard a long, narrow enclosed color corridor merely
            # because its distance-transform maximum is below two pixels.
            free_interior = cv2.erode(
                component,np.ones((3,3),np.uint8),
                borderType=cv2.BORDER_CONSTANT,borderValue=0)
            free_count = int(np.count_nonzero(free_interior))
            if area < 4 or free_count == 0:
                # No safe Bucket seed remains after the contour. These
                # are genuine line-only details or local repair candidates.
                if area >= 2:
                    narrow_regions.append({'rgb':rgb,'area':area,'mask':component,
                                           'free_pixels':free_count})
                else:
                    omitted_pixels += area
                continue
            # Seed on the actual unpainted interior, not a boundary pixel.
            _,_,_,seed = cv2.minMaxLoc(
                cv2.distanceTransform(free_interior,cv2.DIST_L2,3))
            # Preserve inner holes as explicit closed boundaries. The old
            # RETR_EXTERNAL-only planner left these openings unprotected:
            # Paint Bucket could flood the islands inside an outer contour.
            all_contours,hierarchy = cv2.findContours(
                component,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            paths = []
            for boundary in all_contours:
                boundary_pts = [tuple(map(int,p[0])) for p in boundary]
                if len(boundary_pts) > 1:
                    paths.append(boundary_pts+[boundary_pts[0]])
            regions.append({'area':area,'rgb':rgb,'paths':paths,
                            'seed':seed,'mask':component})
    # Pixel-level plan verification: a closed contour alone is not proof
    # that Paint bucket will cover the component. Simulate each region mask
    # and reject unsafe bucket plans before any UI operation.
    safe_regions = []
    unsafe_regions = []
    for region in regions:
        pts = np.asarray(region['paths'][0],dtype=np.int32).reshape(-1,1,2)
        footprint = np.zeros((h,w),np.uint8)
        cv2.drawContours(footprint,[pts],-1,255,thickness=cv2.FILLED)
        rgb = region['rgb']
        intended = np.all(source == np.asarray(rgb[::-1],np.uint8),axis=2)
        # A bucket must not cover a pixel assigned to a different color.
        extra = int(np.count_nonzero((footprint != 0) & ~intended))
        if extra:
            unsafe_regions.append((region,extra))
        else:
            safe_regions.append(region)
    # Locate the exact source-color boundaries crossed by each proposed
    # bucket. This is a diagnostic, not a drawing workaround.
    boundary_diagnostics = []
    for region,extra in unsafe_regions:
        pts = np.asarray(region['paths'][0],dtype=np.int32).reshape(-1,1,2)
        footprint = np.zeros((h,w),np.uint8)
        cv2.drawContours(footprint,[pts],-1,255,thickness=cv2.FILLED)
        intended = np.all(source == np.asarray(region['rgb'][::-1],np.uint8),axis=2)
        ys,xs = np.where((footprint != 0) & ~intended)
        boundary_diagnostics.append({
            'rgb': '#%02X%02X%02X' % region['rgb'],
            'area': int(region['area']),
            'crossed_pixels': int(extra),
            'crossed_bbox_xyxy': [int(xs.min()),int(ys.min()),
                                  int(xs.max()),int(ys.max())],
            'sample_xy': [[int(x),int(y)] for x,y in zip(xs[:30],ys[:30])]
        })
    if logs:
        diagnostic_dir = output_dir or Path('output/comparison/image-bucket')
        diagnostic_dir.mkdir(parents=True,exist_ok=True)
        (diagnostic_dir/'boundary-diagnostics.json').write_text(
            json.dumps(boundary_diagnostics,indent=2),encoding='utf-8')
        print('BOUNDARY DIAGNOSTICS: output/comparison/image-bucket/'
              'boundary-diagnostics.json',flush=True)
    # Holes are not missing pixels: RETR_EXTERNAL deliberately surrounds
    # inner islands. Paint outer areas first, then repaint their inner islands.
    # This is a layered bucket plan, not thousands of pixel-by-pixel strokes.
    print(f'BUCKET PREFLIGHT: {len(safe_regions)} simple regions, '
          f'{len(unsafe_regions)} regions with interior islands',flush=True)
    fallback_jobs = []
    fallback_limit = 1500
    def tier(area):
        if area >= 1000: return 0
        if area >= 100: return 1
        return 2
    analysis_seconds = time.perf_counter() - stage_clock
    stage_clock = time.perf_counter()
    # Coarse-to-fine, group by RGB inside each tier to reduce UI dialogs.
    # Large outer silhouettes first, small interior color islands last.
    # This avoids a large bucket erasing smaller regions drawn earlier.
    regions.sort(key=lambda x:-x['area'])
    # Only true black source pixels become black marks; never trace both sides of a thin black component.
    details.sort(key=lambda x:(x['rgb'] == (0,0,0),x['rgb'],-x['area']))
    left,top = 250+(500-w)//2,155+(547-h)//2
    output = output_dir or Path('output/comparison/image-bucket')
    if logs:
        output.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(output/'expected.png'),source)
    print(f'IMAGE DRAW: {len(regions)} fill regions, {len(details)} thin details, '
          f'{omitted_pixels} noise pixels omitted; black details last')
    print(f'BUCKET WIDTH CHECK: {len(narrow_regions)} components have no '
          '3x3 free interior after boundaries; all other enclosed '
          'components retain Bucket candidates',flush=True)
    print('Open a NEW blank white maximized Paint canvas. F8 cancels.',flush=True)
    planning_seconds = time.perf_counter() - stage_clock
    stage_clock = time.perf_counter()
    print(f'STARTUP: imports={import_seconds:.3f}s, map={read_seconds:.3f}s, masks={mask_seconds:.3f}s, regions={analysis_seconds:.3f}s, plan={planning_seconds:.3f}s',flush=True)
    print('STARTUP: initializing Paint...',flush=True)
    engine = DrawingEngine(speed=speed)
    # Reset Paint's persistent tool state before the first canvas action.
    # Paint may reopen with Bucket and black selected from a prior run.
    engine.tools.select('pencil')
    engine.stroke_size.select(1)
    startup_ink = max(
        ((rgb,int(np.count_nonzero(mask))) for rgb,mask in groups
         if max(rgb) <= 24 and rgb != (255,255,255)),
        key=lambda entry:entry[1],default=None)
    if startup_ink is None:
        raise RuntimeError('Cannot initialize Paint: no outline ink color')
    engine.colors.select_rgb(startup_ink[0])
    print(f'STARTUP: Pencil, size=1, ink={startup_ink[0]} selected',flush=True)
    print(f'STARTUP: Paint initialized in {time.perf_counter()-stage_clock:.3f}s',flush=True)
    paint_init_seconds = time.perf_counter() - stage_clock
    startup_total_seconds = time.perf_counter() - startup_clock
    startup_timings = {
        'imports_seconds': round(import_seconds,3),
        'read_pixel_map_seconds': round(read_seconds,3),
        'build_color_masks_seconds': round(mask_seconds,3),
        'analyze_regions_seconds': round(analysis_seconds,3),
        'prepare_plan_and_preview_seconds': round(planning_seconds,3),
        'initialize_paint_seconds': round(paint_init_seconds,3),
        'total_before_drawing_seconds': round(startup_total_seconds,3),
    }
    print('STARTUP TIMING:', json.dumps(startup_timings,ensure_ascii=False),flush=True)
    if logs:
        (output/'startup-timing.json').write_text(json.dumps(startup_timings,indent=2),encoding='utf-8')
    started = time.perf_counter()
    completed = 0
    thick_outline_regions = 0
    color_switches = 0
    previous_rgb = None
    # Do not retrace filled region boundaries: it thickens and pixelates edges.
    repaired = 0
    repair_strokes = 0
    skipped = 0

    # Enforce the drawing sequence at runtime: external silhouette FIRST,
    # internal region boundaries SECOND, bucket operations THIRD.
    # A source-colored silhouette contour does not invent black pixels.
    def draw_contour_path(path,rgb):
        nonlocal previous_rgb,color_switches
        if rgb != previous_rgb:
            engine.colors.select_rgb(rgb)
            previous_rgb = rgb
            color_switches += 1
        engine.tools.select('pencil')
        engine.stroke_size.select(1)
        mapped = [(left+x,top+y) for x,y in path]
        engine.paint.validate_canvas_points(mapped)
        engine.paint.stroke(mapped if len(mapped)>1 else mapped*2,duration=0.012)

    # One stable ink color per continuous outer contour. Never sample the
    # source color at every contour point: that causes thousands of color
    # changes and breaks the mouse path into tiny fragments.
    ink_colors = [(rgb,int(np.count_nonzero(mask))) for rgb,mask in groups
                  if max(rgb) <= 24 and rgb != (255,255,255)]
    if not ink_colors:
        raise RuntimeError('No outline ink color found in source plan')
    outline_rgb = max(ink_colors,key=lambda entry:entry[1])[0]
    silhouette = np.uint8(occupied != 0)*255
    exterior_contours,_ = cv2.findContours(
        silhouette,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print(f'DRAW STAGE 1/3: OUTER CONTOURS ({len(exterior_contours)}), '
          f'fixed ink color={outline_rgb}',flush=True)
    for contour in sorted(exterior_contours,key=cv2.contourArea,reverse=True):
        check_cancel()
        points = [tuple(map(int,p[0])) for p in contour]
        if len(points) > 1:
            draw_contour_path(points+[points[0]],outline_rgb)
    # Black-base workflow: after closing each exterior silhouette, use
    # Paint Bucket to make its entire interior dark. Later color-region
    # boundaries are drawn IN THEIR TARGET COLOR over this black base;
    # each closed colored boundary then isolates a black pocket for Bucket.
    # Any small missed edge stays dark rather than becoming a white seam.
    # Fill each disconnected silhouette separately; never bucket the canvas.
    import pyautogui as pg
    silhouette_count,silhouette_labels,silhouette_stats,_ = (
        cv2.connectedComponentsWithStats(silhouette,connectivity=8))
    engine.tools.select('bucket')
    for silhouette_id in range(1,silhouette_count):
        check_cancel()
        component = np.uint8(silhouette_labels == silhouette_id)*255
        if int(silhouette_stats[silhouette_id,cv2.CC_STAT_AREA]) < 9:
            continue
        inner = cv2.erode(component,np.ones((3,3),np.uint8))
        if not np.any(inner):
            continue
        _,_,_,seed = cv2.minMaxLoc(
            cv2.distanceTransform(inner,cv2.DIST_L2,3))
        point = (left+seed[0],top+seed[1])
        engine.paint.validate_canvas_points([point])
        engine.canvas.fill(*point)
        print(f'BLACK BASE BUCKET: silhouette {silhouette_id}/'
              f'{silhouette_count-1}, area='
              f'{int(silhouette_stats[silhouette_id,cv2.CC_STAT_AREA])}',
              flush=True)
    # The black base is a foundation, not a substitute for the source
    # color masks. Every nonblack region still needs its own closed
    # contour, Bucket fill, and screenshot verification.
    color_batches = {}
    for region in regions:
        color_batches.setdefault(region['rgb'],[]).append(region)
    # White islands are processed last among light colors, while dark ink
    # is restored after all bucket operations.
    batch_order = sorted(color_batches,
                         key=lambda rgb:(rgb == (255,255,255),
                                         -sum(r['area'] for r in color_batches[rgb])))
    # A white highlight INSIDE dark ink must be painted after that ink,
    # while dark pupils INSIDE white remain part of the final ink pass.
    # Detect containment from source geometry, not a global color ranking.
    dark_mask = np.zeros((h,w),np.uint8)
    for ink_rgb,ink_mask in groups:
        if max(ink_rgb) <= 24:
            dark_mask = cv2.bitwise_or(dark_mask,ink_mask)
    dark_boundaries,_ = cv2.findContours(
        dark_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    white_after_ink = []
    for region in color_batches.get((255,255,255),[]):
        x,y = region['seed']
        if any(cv2.pointPolygonTest(contour,(float(x),float(y)),False)>0
               for contour in dark_boundaries):
            white_after_ink.append(region)
    if white_after_ink:
        color_batches[(255,255,255)] = [
            region for region in color_batches[(255,255,255)]
            if all(region is not highlight for highlight in white_after_ink)]
        batch_order = [rgb for rgb in batch_order
                       if color_batches.get(rgb)]
    print(f'NESTED LAYERS: {len(white_after_ink)} white highlights '
          'scheduled after dark ink',flush=True)
    print('DRAW STAGE 2/3: COLOR CONTOURS + BUCKET OVER BLACK BASE',flush=True)
    print(f'DRAW STAGE 2/3: COMPLETE REGIONS BY COLOR '
          f'({len(batch_order)} colors, {len(regions)} regions)',flush=True)
    for rgb in batch_order:
        batch = sorted(color_batches[rgb],key=lambda region:-region['area'])
        if rgb != previous_rgb:
            engine.colors.select_rgb(rgb)
            previous_rgb = rgb
            color_switches += 1
        print(f'COLOR BATCH {rgb}: {len(batch)} regions',flush=True)
        for index,region in enumerate(batch,1):
            check_cancel()
            engine.tools.select('pencil')
            engine.stroke_size.select(1)
            # All enclosing and hole boundaries must exist before Bucket.
            for path in region['paths']:
                mapped = [(left+x,top+y) for x,y in path]
                engine.paint.validate_canvas_points(mapped)
                engine.paint.stroke(mapped if len(mapped)>1 else mapped*2,
                                    duration=0.012)
            seed = region['seed']
            fill_point = (left+seed[0],top+seed[1])
            engine.paint.validate_canvas_points([fill_point])
            # Fill the current completed contour. Region completion is
            # checked below; no separate BUCKET SPILL rejection layer.
            engine.tools.select('bucket')
            engine.canvas.fill(*fill_point)
            # Verify the region's interior, not its edge (the latter may
            # legitimately be repainted by a later neighboring region).
            core = cv2.erode(region['mask'],
                             np.ones((3,3),np.uint8)) != 0
            if np.any(core):
                sx,sy = engine.paint.point(left,top)
                screenshot = pg.screenshot()
                if sx < 0 or sy < 0 or sx+w > screenshot.width or sy+h > screenshot.height:
                    raise RuntimeError('Region verification crop outside display')
                actual = np.asarray(
                    screenshot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
                expected = np.asarray(rgb,dtype=np.uint8)
                # A final ink pass may overlay some boundary pixels, but
                # a filled region's interior must already match its color.
                mismatch = core & np.any(actual != expected,axis=2)
                wrong = int(np.count_nonzero(mismatch))
                if wrong:
                    # Up to four mismatched pixels, including adjacent ones,
                    # are not sufficient evidence of an open Bucket boundary.
                    # Log them for final comparison without repairing them.
                    # Larger mismatches remain a hard stop.
                    n, labels, stats, _ = cv2.connectedComponentsWithStats(
                        np.uint8(mismatch), connectivity=8)
                    largest = max(
                        (int(stats[k,cv2.CC_STAT_AREA]) for k in range(1,n)),
                        default=0)
                    ys,xs = np.where(mismatch)
                    sample = [(int(x),int(y)) for y,x in
                              zip(ys[:8],xs[:8])]
                    if wrong <= 4:
                        print(
                            f'BUCKET MINOR PIXEL VARIANCE: region '
                            f'{index}/{len(batch)}, color={rgb}, '
                            f'pixels={wrong}, sample_local_xy={sample}; '
                            'no Pencil/Undo/refill; final comparison will '
                            'measure remaining differences', flush=True)
                    else:
                        raise RuntimeError(
                            f'BUCKET REGION VALIDATION FAILED: region '
                            f'{index}/{len(batch)}, color={rgb}, '
                            f'missing_interior_pixels={wrong}, '
                            f'largest_mismatch_cluster={largest}, '
                            f'sample_local_xy={sample}. '
                            'Inspect contour closure, holes and seed in planner; '
                            'stopped without Undo, retry, or pixel repair.')
            completed += 1
            print(f'  REGION COMPLETE {index}/{len(batch)} '
                  f'color={rgb}',flush=True)
    print('DRAW STAGE 3/3: FINAL DETAILS AND VERIFICATION',flush=True)
    # No row-run fallback: nested areas are handled by layer ordering.
    # Exact-color fallback for the rejected regions, after safe bucket fills.
    # Record every late Pencil gesture and its source color so unexpected
    # eye-edge marks can be traced to the exact operation without guessing.
    final_pencil_log = []
    def log_final_pencil(stage, rgb, points):
        xs = [int(p[0]-left) for p in points]
        ys = [int(p[1]-top) for p in points]
        entry = {
            'stage': stage,
            'rgb': list(map(int,rgb)),
            'points': len(points),
            'bbox_local_xyxy': [min(xs),min(ys),max(xs),max(ys)],
        }
        final_pencil_log.append(entry)
        print('FINAL PENCIL ACTION: ' +
              json.dumps(entry,separators=(',',':')),flush=True)
        if logs:
            (output/'final-pencil-actions.json').write_text(
                json.dumps(final_pencil_log,indent=2),encoding='utf-8')
    fallback_completed = 0
    for rgb,y,x0,x1 in fallback_jobs:
        check_cancel()
        if rgb != previous_rgb:
            engine.colors.select_rgb(rgb)
            previous_rgb = rgb
            color_switches += 1
        engine.tools.select('pencil')
        engine.stroke_size.select(1)
        mapped = [(left+x0,top+y),(left+x1,top+y)]
        engine.paint.validate_canvas_points(mapped)
        log_final_pencil('fallback',rgb,mapped)
        if x0 == x1:
            engine.canvas.fill(*mapped[0])
        else:
            engine.paint.stroke(mapped,duration=0.008)
        fallback_completed += 1
        if fallback_completed % 200 == 0:
            print(f'BUCKET FALLBACK: {fallback_completed}/{len(fallback_jobs)}',flush=True)
    # Restore source-defined ink boundaries after color layers. Black Base
    # is not sufficient: subsequent color contours and Bucket fills can
    # overwrite one-pixel outlines. Trace connected ink contours as complete
    # Pencil gestures; never print the black mask as scanline/pixel runs.
    sx,sy = engine.paint.point(left,top)
    screenshot = pg.screenshot()
    if sx < 0 or sy < 0 or sx+w > screenshot.width or sy+h > screenshot.height:
        raise RuntimeError('Ink verification crop outside display')
    actual = np.asarray(screenshot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
    source_rgb = source[:,:,::-1]
    expected_dark = np.all(source_rgb <= 24,axis=2)
    actual_dark = np.all(actual <= 40,axis=2)
    # Capture ink overdraw before final restoration. This distinguishes
    # defects introduced by Black Base/color layers from late Pencil work.
    pre_restoration_extra_dark = int(np.count_nonzero(
        (~expected_dark) & actual_dark))
    pre_restoration_missing_dark = int(np.count_nonzero(
        expected_dark & (~actual_dark)))
    print(f'INK BEFORE RESTORATION: missing={pre_restoration_missing_dark}, '
          f'extra={pre_restoration_extra_dark}',flush=True)
    ink_unresolved = int(np.count_nonzero(expected_dark & ~actual_dark))
    black_planned = 0
    black_strokes = 0
    ink_restored = 0
    if ink_unresolved:
        print(f'FINAL INK RESTORATION: {ink_unresolved} missing dark pixels; '
              'draw only unresolved contour segments',flush=True)
        engine.colors.select_rgb(outline_rgb)
        previous_rgb = outline_rgb
        engine.tools.select('pencil')
        engine.stroke_size.select(1)
        ink_contours,_ = cv2.findContours(
            np.uint8(expected_dark)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        # Contours can share boundary pixels. A pixel is scheduled at most
        # once; after each contour we refresh the real Paint screenshot.
        scheduled = set()
        for contour in sorted(ink_contours,
                              key=lambda c:cv2.arcLength(c,True),
                              reverse=True):
            check_cancel()
            points = [tuple(map(int,p[0])) for p in contour]
            if not points:
                continue
            # RETR_LIST contours may include both sides of a thin ink ring.
            # Keep a cyclic contour only if its ends touch on the 8-grid.
            cyclic = (len(points)>2 and
                      max(abs(points[0][0]-points[-1][0]),
                          abs(points[0][1]-points[-1][1])) <= 1)
            missing = [(not actual_dark[y,x]) and (x,y) not in scheduled
                       for x,y in points]
            if not any(missing):
                continue
            if cyclic and all(missing):
                segments = [points+[points[0]]]
            else:
                # Split at already-correct pixels. Rotate a closed contour
                # to start after a correct point so wraparound fragments join.
                if cyclic and not all(missing):
                    cut = missing.index(False)
                    points = points[cut+1:]+points[:cut+1]
                    missing = missing[cut+1:]+missing[:cut+1]
                segments = []
                run = []
                for point,needs_ink in zip(points,missing):
                    if needs_ink:
                        run.append(point)
                    elif run:
                        segments.append(run)
                        run = []
                if run:
                    segments.append(run)
            for segment in segments:
                check_cancel()
                # One dot or one contiguous missing arc, not the whole ring.
                mapped = [(left+x,top+y) for x,y in segment]
                engine.paint.validate_canvas_points(mapped)
                if len(mapped)==1:
                    engine.canvas.fill(*mapped[0])
                else:
                    engine.paint.stroke(mapped,duration=0.012)
                scheduled.update(segment)
                black_planned += 1
                black_strokes += 1
            # A real screenshot, not an assumed in-memory pixel change,
            # determines whether overlapping contours still need work.
            shot = pg.screenshot()
            actual = np.asarray(shot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
            actual_dark = np.all(actual <= 40,axis=2)
        ink_restored = ink_unresolved - int(np.count_nonzero(expected_dark & ~actual_dark))
        print(f'FINAL INK VERIFY: restored={ink_restored}, '
              f'remaining={ink_unresolved-ink_restored}; '
              f'gestures={black_strokes}',flush=True)
    # Last nested layer: white glints inside black eyes. Keep them after
    # the dark ink pass; do not globally paint white before black.
    if white_after_ink:
        if previous_rgb != (255,255,255):
            engine.colors.select_rgb((255,255,255))
            previous_rgb = (255,255,255)
            color_switches += 1
        for index,region in enumerate(white_after_ink,1):
            check_cancel()
            # The black base and color layers can already leave this glint
            # correctly white. Avoid redundant late Pencil movement in that
            # case; preserve the original draw/fill path if it is missing.
            # Reuse the latest verified Paint capture. No canvas edits occur
            # between the ink check and the first highlight, or between
            # highlights that were already white.
            highlight_pixels = region['mask'] != 0
            already_white = bool(np.all(
                actual[highlight_pixels] == 255))
            if already_white:
                print(f'NESTED WHITE HIGHLIGHT ALREADY COMPLETE '
                      f'{index}/{len(white_after_ink)}; skipping Pencil/Bucket',
                      flush=True)
                continue
            engine.tools.select('pencil')
            engine.stroke_size.select(1)
            for path in region['paths']:
                mapped = [(left+x,top+y) for x,y in path]
                engine.paint.validate_canvas_points(mapped)
                log_final_pencil('white_eye_highlight',(255,255,255),mapped)
                engine.paint.stroke(mapped if len(mapped)>1 else mapped*2,
                                    duration=0.012)
            seed = region['seed']
            fill_point = (left+seed[0],top+seed[1])
            engine.paint.validate_canvas_points([fill_point])
            engine.tools.select('bucket')
            engine.canvas.fill(*fill_point)
            sx,sy = engine.paint.point(left,top)
            screenshot = pg.screenshot()
            actual = np.asarray(
                screenshot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
            core = cv2.erode(region['mask'],
                             np.ones((3,3),np.uint8)) != 0
            missing = int(np.count_nonzero(
                core & np.any(actual != 255,axis=2)))
            if missing:
                raise RuntimeError(
                    f'White highlight {index}/{len(white_after_ink)} '
                    f'incomplete: {missing} interior pixels; not advancing.')
            print(f'NESTED WHITE HIGHLIGHT COMPLETE '
                  f'{index}/{len(white_after_ink)}',flush=True)
    narrow_strokes = 0
    narrow_skipped = len(narrow_regions)
    narrow_budget = 0
    engine.stroke_size.select(1)
    # Fine details are never bucket-filled. Trace each contour once and
    # simplify mildly to avoid repeated jitter around individual pixels.
    for i,detail in enumerate(details,1):
        check_cancel()
        rgb = detail['rgb']
        if rgb != previous_rgb:
            engine.colors.select_rgb(rgb)
            previous_rgb = rgb
            color_switches += 1
        engine.tools.select('pencil')
        contour = detail['contour']
        points = [tuple(map(int,v[0])) for v in contour]
        if len(points) > 2:
            points.append(points[0])
        mapped = [(left+x,top+y) for x,y in points]
        engine.paint.validate_canvas_points(mapped)
        log_final_pencil('detail',rgb,mapped)
        if len(mapped) == 1:
            engine.canvas.fill(*mapped[0])
        elif len(mapped) > 1:
            engine.paint.stroke(mapped,duration=0.012)
        if i % 50 == 0:
            print(f'  Details: {i}/{len(details)}')
    finish_stats = {'enabled':bool(pixel_finish),'planned_strokes':0,
                    'attempted_strokes':0,'limited':False}
    if pixel_finish:
        import pyautogui as pg
        sx,sy = engine.paint.point(left,top)
        screenshot = pg.screenshot()
        if sx < 0 or sy < 0 or sx+w > screenshot.width or sy+h > screenshot.height:
            raise RuntimeError('Pixel finish crop outside display')
        actual = np.asarray(screenshot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
        target = cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
        different = np.any(actual != target,axis=2)
        errors = int(np.count_nonzero(different))
        finish_stats['initial_error_pixels'] = errors
        finish_stats['initial_exact_match_percent'] = round(100*(1-errors/(w*h)),4)
        jobs = []
        for rgb in sorted({tuple(int(v) for v in target[y,x])
                           for y,x in zip(*np.where(different))}):
            color_mask = different & np.all(target == np.asarray(rgb,np.uint8),axis=2)
            for y in range(h):
                xs = np.flatnonzero(color_mask[y])
                if len(xs):
                    for run in np.split(xs,np.flatnonzero(np.diff(xs)>1)+1):
                        jobs.append((rgb,int(y),int(run[0]),int(run[-1])))
        finish_stats['planned_strokes'] = len(jobs)
        finish_stats['limited'] = len(jobs) > finish_limit
        print(f'PIXEL FINISH: {errors} errors; {len(jobs)} planned strokes; limit={finish_limit}',flush=True)
        engine.stroke_size.select(1)
        for index,(rgb,y,x0,x1) in enumerate(jobs[:finish_limit],1):
            check_cancel()
            if rgb != previous_rgb:
                engine.colors.select_rgb(rgb)
                previous_rgb = rgb
                color_switches += 1
            engine.tools.select('pencil')
            pts = [(left+x0,top+y),(left+x1,top+y)]
            engine.paint.validate_canvas_points(pts)
            if x0 == x1:
                engine.canvas.fill(*pts[0])
            else:
                engine.paint.stroke(pts,duration=0.008)
            finish_stats['attempted_strokes'] = index
            if index % 100 == 0:
                print(f'PIXEL FINISH: {index}/{min(len(jobs),finish_limit)}',flush=True)
        print('PIXEL FINISH: verifying final image',flush=True)
    black_gap_stats = {'enabled':bool(black_gap_finish),'planned_strokes':0,
                       'attempted_strokes':0,'skipped_strokes':0}
    if black_gap_finish:
        print('BLACK GAP FINISH DISABLED: unsafe contour-following repair was removed; '
              'no finishing mouse actions will be performed.',flush=True)
        black_gap_stats['disabled_reason'] = 'Unsafe contour-following repair disabled'
    # Mandatory one-pixel seam repair: a white pixel between painted
    # neighbors is never delegated to Paint's bucket tool.
    # This is a small finishing pass, never the primary drawing method.
    import pyautogui as pg
    sx,sy = engine.paint.point(left,top)
    shot = pg.screenshot()
    if sx < 0 or sy < 0 or sx+w > shot.width or sy+h > shot.height:
        raise RuntimeError('Gap inspection crop outside display')
    actual = np.asarray(shot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
    target = cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
    white = np.all(actual >= 248,axis=2)
    expected_ink = np.any(target < 248,axis=2)
    painted = ~white
    # A seam can be an entire connected one-pixel-wide corridor, not
    # necessarily a single white cell flanked by colored pixels. Identify
    # narrow *missing* regions from the actual Paint screenshot. Do not
    # touch intentional source-white pixels or broad unfilled interiors.
    missing = np.uint8(white & expected_ink)
    missing_count,missing_labels,_,_ = cv2.connectedComponentsWithStats(
        missing,connectivity=8)
    gaps = np.zeros((h,w),bool)
    painted_u8 = np.uint8(painted)
    # Opposite-side painted neighbors prove that a missing corridor lies
    # between two already-drawn surfaces, rather than inside a blank area.
    opposite = np.zeros((h,w),bool)
    opposite[:,1:-1] |= painted[:,:-2] & painted[:,2:]
    opposite[1:-1,:] |= painted[:-2,:] & painted[2:,:]
    opposite[1:-1,1:-1] |= painted[:-2,:-2] & painted[2:,2:]
    opposite[1:-1,1:-1] |= painted[:-2,2:] & painted[2:,:-2]
    for component_id in range(1,missing_count):
        component = np.uint8(missing_labels == component_id)
        # One-pixel-wide corridors have no fully surrounded 3x3 block.
        # Restrict repairs to components touching opposite painted sides.
        if np.any(cv2.erode(component,np.ones((3,3),np.uint8))):
            continue
        if np.any((component != 0) & opposite):
            gaps |= component != 0
    print(f'ONE-PIXEL GAP DETECTION: {int(np.count_nonzero(missing))} '
          f'missing nonwhite pixels, {int(np.count_nonzero(gaps))} '
          'in narrow enclosed seams',flush=True)
    gap_pixels = int(np.count_nonzero(gaps))
    gap_jobs = []
    for rgb in sorted({tuple(int(v) for v in target[y,x])
                       for y,x in zip(*np.where(gaps))}):
        color_mask = gaps & np.all(target == np.asarray(rgb,np.uint8),axis=2)
        for y in range(h):
            xs = np.flatnonzero(color_mask[y])
            if len(xs):
                for run in np.split(xs,np.flatnonzero(np.diff(xs)>1)+1):
                    gap_jobs.append((rgb,int(y),int(run[0]),int(run[-1])))
    gap_limit = 1500
    print(f'ONE-PIXEL GAP CHECK: {gap_pixels} pixels, '
          f'{len(gap_jobs)} pencil strokes (limit={gap_limit})',flush=True)
    if len(gap_jobs) > gap_limit:
        raise RuntimeError(
            f'Unresolved white seams: {gap_pixels} pixels need '
            f'{len(gap_jobs)} strokes, exceeding safe limit {gap_limit}. '
            'Refusing to report drawing as complete.')
    for rgb,y,x0,x1 in gap_jobs:
        check_cancel()
        if rgb != previous_rgb:
            engine.colors.select_rgb(rgb)
            previous_rgb = rgb
            color_switches += 1
        engine.tools.select('pencil')
        engine.stroke_size.select(1)
        mapped = [(left+x0,top+y),(left+x1,top+y)]
        engine.paint.validate_canvas_points(mapped)
        # Never use the Bucket for a single missing pixel: it can flood
        # a larger unclosed white area instead of placing a pencil dot.
        engine.paint.stroke(mapped if x0 != x1 else mapped*2,duration=0.008)
    if gap_jobs:
        shot = pg.screenshot()
        actual = np.asarray(shot.crop((sx,sy,sx+w,sy+h)).convert('RGB'))
        remaining = int(np.count_nonzero(
            gaps & np.all(actual >= 248,axis=2)))
        print(f'ONE-PIXEL GAP VERIFY: {remaining} unresolved pixels',flush=True)
        if remaining:
            raise RuntimeError(
                f'One-pixel seam verification failed: {remaining} white '
                'pixels remain; drawing not accepted as complete.')
    repaired = gap_pixels
    repair_strokes = len(gap_jobs)
    report = compare_paint_canvas(engine.paint,
                                  output/'expected.png' if logs else cv2.cvtColor(source,cv2.COLOR_BGR2RGB),
                                  w,h,left,top,output=output,tolerance=0,save_logs=logs)
    # A completed drawing is not a passed 1:1 test unless every pixel agrees.
    # Persist the comparison report before rejecting the drawing.
    report['test'] = 'generic image: color boundaries separate from black source-pixel ink'
    report['region_count'] = len(regions)
    report['unsafe_bucket_regions'] = len(unsafe_regions)
    report['unsafe_bucket_fallback_strokes'] = fallback_completed
    report['unsafe_bucket_fallback_limit'] = fallback_limit
    report['completed_regions'] = completed
    report['thick_outline_regions'] = thick_outline_regions
    report['thick_outline_size_preset'] = 1
    report['black_centerline_components'] = 0
    report['black_ink_method'] = 'source ink contour restoration after color layers'
    report['ink_before_restoration'] = {
        'missing_dark_pixels': pre_restoration_missing_dark,
        'extra_dark_pixels': pre_restoration_extra_dark,
    }
    report['ink_after_restoration'] = {
        'missing_dark_pixels': int(np.count_nonzero(expected_dark & ~actual_dark)),
        'extra_dark_pixels': int(np.count_nonzero(~expected_dark & actual_dark)),
    }
    report['black_ink_planned_runs'] = black_planned
    report['black_ink_completed_runs'] = black_strokes
    report['black_ink_safety_limit'] = None  # No arbitrary stroke cap; verify actual Paint pixels
    report['thick_outline_min_area'] = None
    report['thin_regions'] = len(details)
    report['narrow_pencil_regions'] = len(narrow_regions)
    report['narrow_pencil_strokes'] = narrow_strokes
    report['narrow_pencil_stroke_budget'] = narrow_budget
    report['narrow_regions_skipped_for_speed'] = narrow_skipped
    report['omitted_tiny_pixels'] = omitted_pixels
    report['color_switches'] = color_switches
    report['white_gap_repair_attempted_pixels'] = repaired
    report['white_gap_repair_strokes'] = repair_strokes
    report['white_gap_skipped_pixels'] = skipped
    report['repair_method'] = 'mandatory screenshot-based one-pixel white seam detection, source-color pencil strokes, and verification'
    report['elapsed_seconds'] = round(time.perf_counter()-started,3)
    report['startup_timing'] = startup_timings
    report['pixel_finish'] = finish_stats
    report['black_gap_finish'] = black_gap_stats
    if logs:
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['error_pixels'] != 0:
        print(f'WARNING: Paint drawing completed, but exact 1:1 validation did not pass: '
              f'{report["error_pixels"]} mismatching pixels; '
              f'exact match={report["exact_pixel_match_percent"]}%. '
              + (f'Inspect {output / "error-overlay.png"} and {output / "report.json"}.'
                 if logs else 'Use --logs to save comparison diagnostics.'), flush=True)
    else:
        print('PASS: Paint drawing matches the source exactly (1:1).', flush=True)
    print(f'IMAGE DRAW DONE: exact match={report["exact_pixel_match_percent"]}%, '
          f'errors={report["error_pixels"]}' + (f'; output={output}' if logs else ''), flush=True)


def draw_source_image(source_path, speed, pixel_finish=False, finish_limit=1500,
                      black_gap_finish=False, black_gap_limit=120, logs=False, output_dir=None):
    """Convert a lossless <=16-color image to the generic contour/Bucket engine."""
    import cv2
    import numpy as np
    image = cv2.imread(str(source_path),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read source image: {source_path}')
    h,w = image.shape[:2]
    if not (1 <= w <= 500 and 1 <= h <= 547):
        raise ValueError('Source must be at most 500x547; no automatic resizing')
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    colors,counts = np.unique(rgb.reshape(-1,3),axis=0,return_counts=True)
    if len(colors) > 16:
        raise ValueError(f'Source has {len(colors)} colors (maximum 16). '
                         'Refusing lossy palette reduction.')
    groups = []
    for color,count in zip(colors,counts):
        color = tuple(int(v) for v in color)
        if color == (255,255,255):
            continue
        mask = np.all(rgb == np.asarray(color,np.uint8),axis=2)
        strokes = []
        for y,row in enumerate(mask):
            xs = np.flatnonzero(row)
            if not len(xs):
                continue
            for run in np.split(xs,np.flatnonzero(np.diff(xs)>1)+1):
                strokes.append([int(run[0]),y,int(run[-1]),y])
        groups.append({'rgb':'#%02X%02X%02X'%color,
                       'pixels':int(count),'strokes':strokes})
    plan = {
        'format':'drawbuddy-paint-stroke-plan-v1','width':w,'height':h,
        'coordinate_system':'image-local-zero-based','background':'white-prefilled',
        'groups':groups
    }
    if logs:
        output = output_dir or Path('output/pixel-map')
        output.mkdir(parents=True,exist_ok=True)
        plan_path = output/'pixel-map-paint-plan16.json'
        plan_path.write_text(json.dumps(plan,separators=(',',':')),encoding='utf-8')
    print(f'SOURCE IMAGE: {source_path}, {w}x{h}, {len(colors)} exact colors; '
          'using existing full contour/Bucket engine',flush=True)
    draw_image_bucket(plan_path if logs else plan,speed,pixel_finish,finish_limit,
                         black_gap_finish,black_gap_limit,logs=logs,output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description='Draw a source image in Microsoft Paint using Pencil contours and Bucket fills.')
    parser.add_argument('--source-image', type=Path, required=True,
                        help='Exact-color source image (up to 16 colors)')
    parser.add_argument('--speed', type=float, default=20,
                        help='Drawing speed multiplier (0..100)')
    parser.add_argument('--logs', action='store_true',
                        help='Save diagnostics in a unique output directory (off by default)')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 100:
        parser.error('--speed must be greater than 0 and at most 100')
    output_dir = None
    if args.logs:
        output_dir = Path('output') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f'LOG DIRECTORY: {output_dir}', flush=True)
    draw_source_image(args.source_image, args.speed, logs=args.logs,
                      output_dir=output_dir)


if __name__ == '__main__':
    main()
