import numpy as np
import yaml  # requires: pip install PyYAML
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import os
import cv2

# Optional imports for noise filtering (not needed for clean/exact PGM files)
try:
    from scipy import ndimage
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, noise filtering disabled")

try:
    from skimage import morphology, measure
    from skimage.feature import canny
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, advanced filtering disabled")


def load_map_data(pgm_file, yaml_file):
    """Load PGM image and YAML metadata"""
    # Load YAML metadata
    with open(yaml_file, 'r') as f:
        map_metadata = yaml.safe_load(f)

    # Load PGM image
    image = Image.open(pgm_file).convert('L')  # Convert to grayscale
    map_array = np.array(image)

    return map_array, map_metadata


def apply_noise_filtering(map_array, filter_config=None):
    """Apply noise filtering to clean up SLAM map artifacts"""
    # Check if required modules are available
    if not SCIPY_AVAILABLE:
        print("Noise filtering skipped: scipy not available")
        return map_array
    
    if filter_config is None:
        filter_config = {
            'remove_isolated_pixels': True,
            'fill_small_gaps': True,
            'smooth_edges': False,  # Made more conservative
            'remove_small_components': False,
            'min_component_size': 1,  # Much less aggressive - keep single pixels
            'opening_iterations': 1,  # Disable opening to preserve thin walls
            'closing_iterations': 0,  # Keep closing to fill small gaps
            'edge_smooth_kernel_size': 1
        }

    # Debug: Check initial map statistics
    print(f"Map array shape: {map_array.shape}")
    print(f"Map array dtype: {map_array.dtype}")
    print(f"Map array min/max: {map_array.min()}/{map_array.max()}")
    unique_values = np.unique(map_array)
    print(f"Unique pixel values: {unique_values}")

    # Work with binary image (walls = 1, free space = 0)
    # Convert to binary: occupied pixels (< 128) become walls (1)
    binary_map = (map_array < 128).astype(np.uint8)
    original_wall_count = np.sum(binary_map)

    print(f"Original wall pixels: {original_wall_count}")

    # Safety check: If no walls detected, return original
    if original_wall_count == 0:
        print("WARNING: No wall pixels detected! Check your threshold value.")
        print("Returning original map without filtering.")
        return map_array

    # Safety check: If too many walls (likely inverted), warn user
    total_pixels = map_array.shape[0] * map_array.shape[1]
    wall_percentage = (original_wall_count / total_pixels) * 100
    print(f"Wall coverage: {wall_percentage:.1f}% of map")

    if wall_percentage > 80:
        print("WARNING: Very high wall coverage detected!")
        print("Your map might be inverted (walls=white instead of black)")
        print("Consider adjusting the threshold parameter.")

    # 1. Remove isolated pixels (salt noise) - Morphological Opening
    if filter_config['remove_isolated_pixels'] and filter_config['opening_iterations'] > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary_map = binary_opening(binary_map, structure=kernel,
                                    iterations=filter_config['opening_iterations'])
        after_opening = np.sum(binary_map)
        print(f"After removing isolated pixels: {after_opening} ({original_wall_count - after_opening} removed)")
    else:
        after_opening = original_wall_count

    # 2. Fill small gaps in walls - Morphological Closing
    if filter_config['fill_small_gaps'] and filter_config['closing_iterations'] > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary_map = binary_closing(binary_map, structure=kernel,
                                    iterations=filter_config['closing_iterations'])
        after_closing = np.sum(binary_map)
        print(f"After filling gaps: {after_closing} ({after_closing - after_opening} added)")
    else:
        after_closing = after_opening

    # 3. Remove small disconnected components
    if filter_config['remove_small_components'] and SKIMAGE_AVAILABLE:
        # Label connected components
        labeled_map = measure.label(binary_map, connectivity=2)  # 8-connectivity
        component_props = measure.regionprops(labeled_map)

        print(f"Found {len(component_props)} connected components")

        # Remove components smaller than threshold
        min_size = filter_config['min_component_size']
        removed_components = 0
        for prop in component_props:
            if prop.area < min_size:
                binary_map[labeled_map == prop.label] = 0
                removed_components += 1

        after_component_removal = np.sum(binary_map)
        print(f"After removing {removed_components} small components (< {min_size} pixels): {after_component_removal}")
    else:
        after_component_removal = after_closing

    # 4. Smooth edges using morphological operations (disabled by default)
    if filter_config['smooth_edges'] and SKIMAGE_AVAILABLE:
        kernel_size = filter_config['edge_smooth_kernel_size']
        if kernel_size > 1:
            # Use diamond-shaped kernel for edge smoothing
            kernel = morphology.diamond(kernel_size // 2)

            # Light erosion followed by dilation to smooth edges
            smoothed = binary_erosion(binary_map, structure=kernel)
            smoothed = binary_dilation(smoothed, structure=kernel)
            binary_map = smoothed.astype(np.uint8)

            after_smoothing = np.sum(binary_map)
            print(f"After edge smoothing: {after_smoothing}")
        else:
            after_smoothing = after_component_removal
    else:
        after_smoothing = after_component_removal

    # Skip thin protrusion removal for now - it was too aggressive

    # Convert back to grayscale format (walls = 0, free space = 255)
    filtered_map = np.where(binary_map == 1, 0, 255).astype(np.uint8)

    final_wall_count = np.sum(binary_map)

    # Safety check: If we removed too many walls, warn user
    if final_wall_count == 0:
        print("ERROR: All walls were removed by filtering! Returning original map.")
        return map_array

    if final_wall_count < original_wall_count * 0.1:  # Lost more than 90%
        print("WARNING: Filtering removed more than 90% of walls!")
        print("Consider using --no-filter or adjusting filter parameters.")

    reduction_percent = ((original_wall_count - final_wall_count) / original_wall_count) * 100
    print(
        f"Noise filtering complete: {original_wall_count} → {final_wall_count} pixels ({reduction_percent:.1f}% reduction)")

    return filtered_map


def simple_line_detection(map_array, threshold=128, sensitivity=1.0):
    """Line detection with adjustable sensitivity for detecting curved/short walls"""
    print(f"Line detection starting (sensitivity: {sensitivity})...")
    
    # Create binary image: walls are black (< threshold)
    binary = (map_array < threshold).astype(np.uint8) * 255
    print(f"Binary image created: {binary.shape}")
    
    # Simple edge detection
    edges = cv2.Canny(binary, 50, 150)
    
    # Adjust Hough parameters based on sensitivity
    # Higher sensitivity = detect more/shorter/weaker lines
    base_hough_threshold = 15
    base_min_length = 5
    base_max_gap = 5
    
    hough_threshold = max(1, int(base_hough_threshold / sensitivity))
    min_line_length = max(1, int(base_min_length / sensitivity))  
    max_gap = int(base_max_gap * sensitivity)
    
    print(f"Hough parameters: threshold={hough_threshold}, minLength={min_line_length}, maxGap={max_gap}")
    
    # Hough line detection with sensitivity-adjusted parameters
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                    # Distance resolution in pixels
        theta=np.pi/180,          # Angle resolution in radians  
        threshold=hough_threshold, # Minimum number of votes (lower = more sensitive)
        minLineLength=min_line_length, # Minimum line length (lower = detect shorter lines)
        maxLineGap=max_gap        # Maximum gap between segments (higher = connect more)
    )
    
    # Convert to simple line format
    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
            })
    
    print(f"Found {len(line_segments)} line segments")
    return line_segments


def contour_based_detection(map_array, threshold=128, min_area=10, approx_epsilon=2.0):
    """
    Contour-based wall detection for clean/exact PGM files.
    Uses OpenCV contour detection to find wall boundaries, then creates
    bounding rectangles that preserve wall geometry better than Hough lines.
    
    Args:
        map_array: Grayscale numpy array of the map
        threshold: Pixel value threshold for wall detection
        min_area: Minimum contour area to consider (pixels)
        approx_epsilon: Contour approximation epsilon (lower = more detail)
    
    Returns:
        List of wall rectangles with position, size, and rotation
    """
    print(f"Contour-based detection starting...")
    
    # Create binary image: walls are black (< threshold)
    binary = (map_array < threshold).astype(np.uint8) * 255
    print(f"Binary image created: {binary.shape}")
    print(f"Wall pixels: {np.sum(binary > 0)}")
    
    # Find contours - use RETR_LIST to get all contours including holes
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    rectangles = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Get minimum area bounding rectangle (rotated)
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        
        # Skip if too small
        if size[0] < 1 or size[1] < 1:
            continue
        
        rectangles.append({
            'center_x': center[0],
            'center_y': center[1],
            'width': max(size[0], size[1]),  # Length along major axis
            'height': min(size[0], size[1]),  # Width (thickness)
            'angle': np.radians(angle) if size[0] > size[1] else np.radians(angle + 90),
            'area': area
        })
    
    print(f"Created {len(rectangles)} rectangle primitives from contours")
    return rectangles


def run_length_encoding_detection(map_array, threshold=128, min_length=1):
    """
    Run-length encoding based wall detection for clean/exact PGM files.
    Scans the image horizontally and vertically to find wall runs, then
    creates boxes for each run. This preserves the exact pixel-level geometry.
    
    Best for orthogonal mazes and grid-based maps.
    
    Args:
        map_array: Grayscale numpy array of the map
        threshold: Pixel value threshold for wall detection
        min_length: Minimum run length to consider
    
    Returns:
        List of wall boxes in pixel coordinates
    """
    print(f"Run-length encoding detection starting...")
    
    # Create binary image: walls = 1
    binary = (map_array < threshold).astype(np.uint8)
    wall_count = np.sum(binary)
    print(f"Wall pixels: {wall_count}")
    
    if wall_count == 0:
        print("No walls found!")
        return []
    
    height, width = binary.shape
    boxes = []
    used = np.zeros_like(binary, dtype=bool)
    
    # Scan horizontally for horizontal wall segments
    for y in range(height):
        x = 0
        while x < width:
            if binary[y, x] == 1 and not used[y, x]:
                # Start of a run
                start_x = x
                while x < width and binary[y, x] == 1:
                    x += 1
                run_length = x - start_x
                
                if run_length >= min_length:
                    # Try to extend vertically to find rectangle
                    max_height = 1
                    for dy in range(1, height - y):
                        # Check if entire row segment exists
                        if np.all(binary[y + dy, start_x:start_x + run_length] == 1):
                            max_height = dy + 1
                        else:
                            break
                    
                    # Only use if not already covered by larger rect
                    if not np.all(used[y:y + max_height, start_x:start_x + run_length]):
                        # Mark as used
                        used[y:y + max_height, start_x:start_x + run_length] = True
                        
                        boxes.append({
                            'x': start_x + run_length / 2.0,
                            'y': y + max_height / 2.0,
                            'width': float(run_length),
                            'height': float(max_height),
                            'angle': 0
                        })
            else:
                x += 1
    
    # Scan vertically for any remaining vertical wall segments
    for x in range(width):
        y = 0
        while y < height:
            if binary[y, x] == 1 and not used[y, x]:
                # Start of a run
                start_y = y
                while y < height and binary[y, x] == 1 and not used[y, x]:
                    y += 1
                run_length = y - start_y
                
                if run_length >= min_length:
                    # Try to extend horizontally
                    max_width = 1
                    for dx in range(1, width - x):
                        if np.all(binary[start_y:start_y + run_length, x + dx] == 1) and \
                           not np.all(used[start_y:start_y + run_length, x + dx]):
                            max_width = dx + 1
                        else:
                            break
                    
                    if not np.all(used[start_y:start_y + run_length, x:x + max_width]):
                        used[start_y:start_y + run_length, x:x + max_width] = True
                        
                        boxes.append({
                            'x': x + max_width / 2.0,
                            'y': start_y + run_length / 2.0,
                            'width': float(max_width),
                            'height': float(run_length),
                            'angle': 0
                        })
            else:
                y += 1
    
    # Report coverage
    covered = np.sum(used)
    print(f"Created {len(boxes)} wall boxes from run-length encoding")
    print(f"Coverage: {covered}/{wall_count} pixels ({100*covered/wall_count:.1f}%)")
    
    return boxes


def pixel_grid_detection(map_array, threshold=128, merge_distance=2):
    """
    Pixel grid-based wall detection for clean/exact PGM files.
    Converts wall pixels directly to small box primitives, with optional merging.
    Best for maps where you want exact pixel-level representation.
    
    Args:
        map_array: Grayscale numpy array of the map
        threshold: Pixel value threshold for wall detection
        merge_distance: Distance for merging adjacent pixels (0 = no merge)
    
    Returns:
        List of wall boxes in pixel coordinates
    """
    print(f"Pixel grid detection starting...")
    
    # Create binary image: walls = 1
    binary = (map_array < threshold).astype(np.uint8)
    wall_count = np.sum(binary)
    print(f"Wall pixels: {wall_count}")
    
    if wall_count == 0:
        print("No walls found!")
        return []
    
    # Find connected components for efficient processing
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"Found {num_labels - 1} connected components")
    
    boxes = []
    
    # Process each connected component
    for label in range(1, num_labels):  # Skip background (0)
        # Get bounding box of this component
        x, y, w, h, area = stats[label]
        
        if area < 1:
            continue
        
        # For small components, use single box
        if area < 50 or (w <= 3 and h <= 3):
            boxes.append({
                'x': x + w/2,
                'y': y + h/2,
                'width': w,
                'height': h,
                'angle': 0
            })
        else:
            # For larger components, extract the actual shape using contours
            component_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Use rotated rectangle for better fit
                if len(contour) >= 5:
                    rect = cv2.minAreaRect(contour)
                    center, size, angle = rect
                    boxes.append({
                        'x': center[0],
                        'y': center[1],
                        'width': max(size[0], size[1]),
                        'height': min(size[0], size[1]),
                        'angle': np.radians(angle) if size[0] > size[1] else np.radians(angle + 90)
                    })
                else:
                    # Fallback to bounding box
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    boxes.append({
                        'x': bx + bw/2,
                        'y': by + bh/2,
                        'width': bw,
                        'height': bh,
                        'angle': 0
                    })
    
    print(f"Created {len(boxes)} wall boxes from pixel grid")
    return boxes


def scale_rectangles_to_world(rectangles, resolution, origin, map_height):
    """Convert rectangle coordinates from image pixels to world coordinates"""
    world_rects = []
    
    for rect in rectangles:
        # Convert pixel coordinates to world coordinates
        # Fix Y-axis flip: PGM origin is top-left, world origin is bottom-left
        world_x = origin[0] + rect['x'] * resolution if 'x' in rect else origin[0] + rect['center_x'] * resolution
        world_y = origin[1] + (map_height - (rect['y'] if 'y' in rect else rect['center_y'])) * resolution
        
        world_rects.append({
            'x': world_x,
            'y': world_y,
            'width': rect['width'] * resolution,
            'height': rect['height'] * resolution,
            'angle': rect.get('angle', 0)
        })
    
    print(f"Scaled {len(world_rects)} rectangles to world coordinates")
    return world_rects


def create_walls_from_rectangles(world_rects, wall_height=2.0):
    """Create wall boxes from world coordinate rectangles"""
    walls = []
    
    for i, rect in enumerate(world_rects):
        wall = {
            'name': f'wall_{i}',
            'x': rect['x'],
            'y': rect['y'],
            'z': wall_height / 2.0,
            'length': rect['width'],     # Length of the wall
            'width': rect['height'],      # Thickness of the wall  
            'height': wall_height,        # Vertical height
            'yaw': rect.get('angle', 0)   # Rotation around Z-axis
        }
        walls.append(wall)
    
    print(f"Created {len(walls)} wall boxes from rectangles")
    return walls


def scale_lines_to_world(line_segments, resolution, origin, map_height):
    """Convert line coordinates from image pixels to world coordinates"""
    world_lines = []
    
    for line in line_segments:
        # Convert pixel coordinates to world coordinates
        # Fix Y-axis flip: PGM origin is top-left, world origin is bottom-left
        world_x1 = origin[0] + line['x1'] * resolution
        world_y1 = origin[1] + (map_height - line['y1']) * resolution
        world_x2 = origin[0] + line['x2'] * resolution  
        world_y2 = origin[1] + (map_height - line['y2']) * resolution
        
        world_lines.append({
            'x1': world_x1, 'y1': world_y1,
            'x2': world_x2, 'y2': world_y2,
            'length': np.sqrt((world_x2-world_x1)**2 + (world_y2-world_y1)**2)
        })
    
    print(f"Scaled {len(world_lines)} lines to world coordinates")
    return world_lines


def create_walls_from_lines(world_lines, wall_height=2.0, wall_thickness=0.1):
    """Create wall boxes from world coordinate lines"""
    walls = []
    
    for i, line in enumerate(world_lines):
        # Calculate center point of the line
        center_x = (line['x1'] + line['x2']) / 2.0
        center_y = (line['y1'] + line['y2']) / 2.0
        center_z = wall_height / 2.0
        
        # Calculate line length and angle
        length = line['length']
        angle = np.arctan2(line['y2'] - line['y1'], line['x2'] - line['x1'])
        
        # Create wall box
        wall = {
            'name': f'wall_{i}',
            'x': center_x,
            'y': center_y, 
            'z': center_z,
            'length': length,        # Along the line
            'width': wall_thickness, # Perpendicular to line
            'height': wall_height,   # Vertical
            'yaw': angle            # Rotation around Z-axis
        }
        walls.append(wall)
    
    print(f"Created {len(walls)} wall boxes")
    return walls


def create_simple_sdf(walls, world_name="slam_world"):
    """Create simple SDF world with walls"""
    
    # Create root SDF element
    sdf = ET.Element("sdf", version="1.6")
    world = ET.SubElement(sdf, "world", name=world_name)
    
    # Add basic lighting
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"
    
    # Add ground plane
    ground = ET.SubElement(world, "model", name="ground_plane")
    ET.SubElement(ground, "static").text = "true"
    ground_link = ET.SubElement(ground, "link", name="link")
    
    # Ground collision
    ground_coll = ET.SubElement(ground_link, "collision", name="collision")
    ground_geom = ET.SubElement(ground_coll, "geometry")
    ground_plane = ET.SubElement(ground_geom, "plane")
    ET.SubElement(ground_plane, "normal").text = "0 0 1"
    ET.SubElement(ground_plane, "size").text = "100 100"
    
    # Ground visual
    ground_vis = ET.SubElement(ground_link, "visual", name="visual")
    ground_vis_geom = ET.SubElement(ground_vis, "geometry") 
    ground_vis_plane = ET.SubElement(ground_vis_geom, "plane")
    ET.SubElement(ground_vis_plane, "normal").text = "0 0 1"
    ET.SubElement(ground_vis_plane, "size").text = "100 100"
    
    # Ground material
    ground_material = ET.SubElement(ground_vis, "material")
    ET.SubElement(ground_material, "ambient").text = "0.8 0.8 0.8 1"
    ET.SubElement(ground_material, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(ground_material, "specular").text = "0.8 0.8 0.8 1"
    
    # Add each wall
    for wall in walls:
        wall_model = ET.SubElement(world, "model", name=wall['name'])
        ET.SubElement(wall_model, "static").text = "true"
        ET.SubElement(wall_model, "pose").text = f"{wall['x']} {wall['y']} {wall['z']} 0 0 {wall['yaw']}"
        
        wall_link = ET.SubElement(wall_model, "link", name="link")
        
        # Collision
        wall_coll = ET.SubElement(wall_link, "collision", name="collision")
        coll_geom = ET.SubElement(wall_coll, "geometry")
        coll_box = ET.SubElement(coll_geom, "box")
        ET.SubElement(coll_box, "size").text = f"{wall['length']} {wall['width']} {wall['height']}"
        
        # Visual  
        wall_vis = ET.SubElement(wall_link, "visual", name="visual")
        vis_geom = ET.SubElement(wall_vis, "geometry")
        vis_box = ET.SubElement(vis_geom, "box")
        ET.SubElement(vis_box, "size").text = f"{wall['length']} {wall['width']} {wall['height']}"
        
        # Material
        material = ET.SubElement(wall_vis, "material")
        ET.SubElement(material, "ambient").text = "0.5 0.5 0.5 1"
        ET.SubElement(material, "diffuse").text = "0.7 0.7 0.7 1"
    
    return sdf


def merge_collinear_segments(segments, angle_tolerance=5, distance_tolerance=3):
    """Merge collinear and adjacent line segments into longer straight lines"""
    if not segments:
        return []
    
    # Convert angle tolerance to radians
    angle_tol_rad = np.radians(angle_tolerance)
    
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
            
        # Start with current segment
        current_seg = {
            'start': seg1['start'],
            'end': seg1['end'], 
            'length': seg1['length'],
            'angle': seg1['angle']
        }
        used.add(i)
        
        # Try to extend this segment by merging with others
        extended = True
        while extended:
            extended = False
            best_merge = None
            best_j = None
            
            for j, seg2 in enumerate(segments):
                if j in used:
                    continue
                    
                # Check if segments can be merged (similar angle and close/touching)
                angle_diff = abs(current_seg['angle'] - seg2['angle'])
                angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wraparound
                
                if angle_diff <= angle_tol_rad:
                    # Check if segments are close enough to merge
                    merge_info = can_merge_segments(current_seg, seg2, distance_tolerance)
                    if merge_info:
                        best_merge = merge_info
                        best_j = j
                        extended = True
                        break
            
            if extended and best_j is not None:
                # Merge the best candidate
                current_seg = best_merge
                used.add(best_j)
        
        merged.append(current_seg)
    
    return merged


def can_merge_segments(seg1, seg2, distance_tolerance):
    """Check if two segments can be merged and return merged segment if possible"""
    
    def distance_point_to_point(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Get all four endpoints
    points = [seg1['start'], seg1['end'], seg2['start'], seg2['end']]
    
    # Find the two points that are furthest apart (these will be the new endpoints)
    max_dist = 0
    best_endpoints = None
    
    for i in range(4):
        for j in range(i+1, 4):
            dist = distance_point_to_point(points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                best_endpoints = (points[i], points[j])
    
    if best_endpoints is None:
        return None
    
    # Check if segments are close enough (endpoints should be close to each other)
    min_endpoint_dist = float('inf')
    for p1 in [seg1['start'], seg1['end']]:
        for p2 in [seg2['start'], seg2['end']]:
            dist = distance_point_to_point(p1, p2)
            min_endpoint_dist = min(min_endpoint_dist, dist)
    
    if min_endpoint_dist > distance_tolerance:
        return None
    
    # Create merged segment
    start, end = best_endpoints
    length = distance_point_to_point(start, end)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    
    return {
        'start': start,
        'end': end,
        'length': length,
        'angle': angle
    }


def extract_walls(map_array, threshold=128, apply_filtering=True, filter_config=None):
    """Extract wall positions from the map array with optional noise filtering (legacy)"""

    # Apply noise filtering if requested
    if apply_filtering:
        print("Applying noise filtering...")
        map_array = apply_noise_filtering(map_array, filter_config)

    # Find all occupied pixels (walls)
    wall_pixels = np.where(map_array < threshold)
    wall_positions = list(zip(wall_pixels[1], wall_pixels[0]))  # (x, y) format

    return wall_positions


def find_rectangular_walls(wall_positions):
    """Find rectangular wall segments from individual wall pixels"""
    if not wall_positions:
        return []

    # Convert to set for O(1) lookup
    wall_set = set(wall_positions)
    processed = set()
    rectangles = []

    for x, y in wall_positions:
        if (x, y) in processed:
            continue

        # Try to grow a rectangle starting from this pixel
        # First, find the maximum width (horizontal extent)
        max_width = 1
        while (x + max_width, y) in wall_set:
            max_width += 1

        # For each possible width, find the maximum height
        best_rect = {'x': x, 'y': y, 'width': 1, 'height': 1}
        best_area = 1

        for width in range(1, max_width + 1):
            # Check if we can extend vertically
            height = 1
            while height <= 1000:  # reasonable limit
                # Check if all pixels in this row exist
                valid_row = True
                for dx in range(width):
                    if (x + dx, y + height) not in wall_set:
                        valid_row = False
                        break

                if not valid_row:
                    break
                height += 1

            area = width * height
            if area > best_area:
                best_area = area
                best_rect = {'x': x, 'y': y, 'width': width, 'height': height}

        # Mark all pixels in this rectangle as processed
        for dx in range(best_rect['width']):
            for dy in range(best_rect['height']):
                processed.add((x + dx, y + dy))

        rectangles.append(best_rect)

    return rectangles


def merge_adjacent_pixels(wall_positions):
    """Merge adjacent wall pixels into larger rectangular segments"""
    if not wall_positions:
        return []

    # Convert list to numpy array for easier processing
    wall_array = np.array(wall_positions)

    # Create a binary image for processing
    if len(wall_array) == 0:
        return []

    min_x, min_y = wall_array.min(axis=0)
    max_x, max_y = wall_array.max(axis=0)

    # Create binary image
    img_width = max_x - min_x + 3  # +3 for padding
    img_height = max_y - min_y + 3
    binary_img = np.zeros((img_height, img_width), dtype=np.uint8)

    # Fill in wall pixels (offset by 1 for padding)
    for x, y in wall_positions:
        binary_img[y - min_y + 1, x - min_x + 1] = 1

    # Find rectangles using a greedy approach
    rectangles = []
    processed = np.zeros_like(binary_img, dtype=bool)

    for y in range(img_height):
        for x in range(img_width):
            if binary_img[y, x] == 1 and not processed[y, x]:
                # Find the largest rectangle starting from this point
                rect = find_largest_rectangle(binary_img, processed, x, y)
                if rect:
                    # Convert back to original coordinates
                    rect['x'] += min_x - 1
                    rect['y'] += min_y - 1
                    rectangles.append(rect)

    return rectangles


def find_largest_rectangle(binary_img, processed, start_x, start_y):
    """Find the largest rectangle starting from a given point"""
    height, width = binary_img.shape

    if processed[start_y, start_x] or binary_img[start_y, start_x] == 0:
        return None

    # Find maximum width for the first row
    max_width = 0
    for x in range(start_x, width):
        if binary_img[start_y, x] == 1 and not processed[start_y, x]:
            max_width += 1
        else:
            break

    if max_width == 0:
        return None

    # Find the maximum height for each possible width
    best_rect = {'x': start_x, 'y': start_y, 'width': 1, 'height': 1}
    best_area = 1

    for w in range(1, max_width + 1):
        h = 1
        # Extend downward while possible
        while start_y + h < height:
            # Check if the entire row is available
            valid_row = True
            for x in range(start_x, start_x + w):
                if (binary_img[start_y + h, x] == 0 or
                        processed[start_y + h, x]):
                    valid_row = False
                    break

            if valid_row:
                h += 1
            else:
                break

        area = w * h
        if area > best_area:
            best_area = area
            best_rect = {'x': start_x, 'y': start_y, 'width': w, 'height': h}

    # Mark the rectangle as processed
    for y in range(best_rect['y'], best_rect['y'] + best_rect['height']):
        for x in range(best_rect['x'], best_rect['x'] + best_rect['width']):
            processed[y, x] = True

    return best_rect


def create_wall_boxes_from_segments(line_segments, resolution, origin, wall_height=2.0, wall_thickness=0.1, map_width=None, map_height=None):
    """Create wall boxes from line segments"""
    if not line_segments:
        return []

    walls = []
    
    for i, segment in enumerate(line_segments):
        x1, y1 = segment['start']
        x2, y2 = segment['end']
        
        # Calculate center point
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Convert to world coordinates (flip Y-axis to fix 180-degree rotation)
        if map_height is not None:
            # Flip Y coordinate to correct orientation (PGM origin top-left -> SDF origin bottom-left)
            world_x = origin[0] + center_x * resolution
            world_y = origin[1] + (map_height - center_y) * resolution
        else:
            # Fallback to original coordinates
            world_x = origin[0] + center_x * resolution
            world_y = origin[1] + center_y * resolution
        
        # Calculate length and angle
        length = segment['length'] * resolution
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        # Create wall box
        walls.append({
            'x': world_x,
            'y': world_y,
            'z': wall_height / 2,
            'width': length,
            'length': wall_thickness,  # Thickness of the wall
            'height': wall_height,
            'angle': angle,
            'segment_id': i
        })
    
    print(f"Created {len(walls)} wall boxes from line segments")
    return walls

def create_wall_boxes(wall_positions, resolution, origin, wall_height=2.0, merge_adjacent=True):
    """Create wall boxes from wall positions (legacy method)"""
    if not wall_positions:
        return []

    walls = []

    if merge_adjacent:
        print("Merging adjacent wall pixels...")
        rectangles = merge_adjacent_pixels(wall_positions)
        print(f"Merged {len(wall_positions)} pixels into {len(rectangles)} rectangles")

        for rect in rectangles:
            # Calculate center position and size in world coordinates
            center_x = rect['x'] + rect['width'] / 2.0
            center_y = rect['y'] + rect['height'] / 2.0

            world_x = origin[0] + center_x * resolution
            world_y = origin[1] + center_y * resolution

            walls.append({
                'x': world_x,
                'y': world_y,
                'z': wall_height / 2,
                'width': rect['width'] * resolution,
                'length': rect['height'] * resolution,
                'height': wall_height
            })
    else:
        # Create individual boxes for each wall pixel
        for x, y in wall_positions:
            world_x = origin[0] + x * resolution
            world_y = origin[1] + y * resolution

            walls.append({
                'x': world_x,
                'y': world_y,
                'z': wall_height / 2,
                'width': resolution,
                'length': resolution,
                'height': wall_height
            })

    return walls


def create_sdf_world(walls, world_name="slam_world"):
    """Create SDF XML structure"""
    # Create root SDF element
    sdf = ET.Element("sdf", version="1.6")

    # Create world element
    world = ET.SubElement(sdf, "world", name=world_name)

    # Add basic world properties
    # Physics
    physics = ET.SubElement(world, "physics", name="default_physics", default="0", type="ode")
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1"
    ET.SubElement(physics, "real_time_update_rate").text = "1000"

    # Gravity
    ET.SubElement(world, "gravity").text = "0 0 -9.8"

    # Scene
    scene = ET.SubElement(world, "scene")
    ET.SubElement(scene, "ambient").text = "0.4 0.4 0.4 1"
    ET.SubElement(scene, "background").text = "0.7 0.7 0.7 1"
    ET.SubElement(scene, "shadows").text = "true"

    # Add sun light
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "cast_shadows").text = "1"
    ET.SubElement(light, "pose").text = "0 0 10 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "specular").text = "0.2 0.2 0.2 1"
    ET.SubElement(light, "attenuation").text = "1 0.0 0.0"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"

    # Add ground plane
    ground_model = ET.SubElement(world, "model", name="ground_plane")
    ET.SubElement(ground_model, "static").text = "true"
    ground_link = ET.SubElement(ground_model, "link", name="link")
    ground_collision = ET.SubElement(ground_link, "collision", name="collision")
    ground_geom = ET.SubElement(ground_collision, "geometry")
    ground_plane = ET.SubElement(ground_geom, "plane")
    ET.SubElement(ground_plane, "normal").text = "0 0 1"
    ET.SubElement(ground_plane, "size").text = "100 100"

    ground_visual = ET.SubElement(ground_link, "visual", name="visual")
    ground_vis_geom = ET.SubElement(ground_visual, "geometry")
    ground_vis_plane = ET.SubElement(ground_vis_geom, "plane")
    ET.SubElement(ground_vis_plane, "normal").text = "0 0 1"
    ET.SubElement(ground_vis_plane, "size").text = "100 100"

    ground_material = ET.SubElement(ground_visual, "material")
    ET.SubElement(ground_material, "ambient").text = "0.8 0.8 0.8 1"
    ET.SubElement(ground_material, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(ground_material, "specular").text = "0.8 0.8 0.8 1"

    # Add wall models
    for i, wall in enumerate(walls):
        wall_model = ET.SubElement(world, "model", name=f"wall_{i}")
        ET.SubElement(wall_model, "static").text = "true"
        
        # Handle rotation if angle is specified
        if 'angle' in wall:
            pose_str = f"{wall['x']} {wall['y']} {wall['z']} 0 0 {wall['angle']}"
        else:
            pose_str = f"{wall['x']} {wall['y']} {wall['z']} 0 0 0"
        
        ET.SubElement(wall_model, "pose").text = pose_str

        wall_link = ET.SubElement(wall_model, "link", name="link")

        # Collision
        wall_collision = ET.SubElement(wall_link, "collision", name="collision")
        coll_geometry = ET.SubElement(wall_collision, "geometry")
        coll_box = ET.SubElement(coll_geometry, "box")
        ET.SubElement(coll_box, "size").text = f"{wall['width']} {wall['length']} {wall['height']}"

        # Visual
        wall_visual = ET.SubElement(wall_link, "visual", name="visual")
        vis_geometry = ET.SubElement(wall_visual, "geometry")
        vis_box = ET.SubElement(vis_geometry, "box")
        ET.SubElement(vis_box, "size").text = f"{wall['width']} {wall['length']} {wall['height']}"

        # Material
        wall_material = ET.SubElement(wall_visual, "material")
        ET.SubElement(wall_material, "ambient").text = "0.5 0.5 0.5 1"
        ET.SubElement(wall_material, "diffuse").text = "0.7 0.7 0.7 1"
        ET.SubElement(wall_material, "specular").text = "0.1 0.1 0.1 1"

    return sdf


def simple_pgm_to_sdf(pgm_file, yaml_file, output_sdf, wall_height=2.0, wall_thickness=0.1, 
                      threshold=128, sensitivity=1.0, method='hough', min_area=10, approx_epsilon=2.0):
    """
    Convert PGM/YAML to SDF with selectable detection method.
    
    Args:
        pgm_file: Path to PGM map file
        yaml_file: Path to YAML metadata file  
        output_sdf: Output SDF file path
        wall_height: Height of walls in meters
        wall_thickness: Thickness of walls (only used for 'hough' method)
        threshold: Pixel value threshold for wall detection
        sensitivity: Line detection sensitivity (only for 'hough' method)
        method: Detection method - 'hough', 'contour', 'pixel', or 'rle'
            - 'hough': Uses Hough line detection (good for noisy SLAM maps)
            - 'contour': Uses contour detection (good for clean maps with thick walls)
            - 'pixel': Uses connected components (for simple maps)
            - 'rle': Uses run-length encoding (BEST for orthogonal/grid mazes)
        min_area: Minimum contour area (for 'contour' and 'pixel' methods)
        approx_epsilon: Contour approximation (for 'contour' method)
    """
    print(f"Converting: {pgm_file} + {yaml_file} -> {output_sdf}")
    print(f"Method: {method}")
    
    # 1. Load map data
    map_array, map_metadata = load_map_data(pgm_file, yaml_file)
    print(f"Map: {map_array.shape}, Resolution: {map_metadata['resolution']}, Origin: {map_metadata['origin']}")
    
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    map_height = map_array.shape[0]
    
    # 2. Detect walls based on method
    if method == 'hough':
        # Original Hough line detection
        line_segments = simple_line_detection(map_array, threshold, sensitivity)
        world_lines = scale_lines_to_world(line_segments, resolution, origin, map_height)
        walls = create_walls_from_lines(world_lines, wall_height, wall_thickness)
        
    elif method == 'contour':
        # Contour-based detection (preserves wall thickness)
        rectangles = contour_based_detection(map_array, threshold, min_area, approx_epsilon)
        world_rects = scale_rectangles_to_world(rectangles, resolution, origin, map_height)
        walls = create_walls_from_rectangles(world_rects, wall_height)
        
    elif method == 'pixel':
        # Pixel grid-based detection (for simple maps)
        boxes = pixel_grid_detection(map_array, threshold)
        world_rects = scale_rectangles_to_world(boxes, resolution, origin, map_height)
        walls = create_walls_from_rectangles(world_rects, wall_height)
    
    elif method == 'rle':
        # Run-length encoding detection (BEST for orthogonal mazes)
        boxes = run_length_encoding_detection(map_array, threshold)
        world_rects = scale_rectangles_to_world(boxes, resolution, origin, map_height)
        walls = create_walls_from_rectangles(world_rects, wall_height)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hough', 'contour', 'pixel', or 'rle'")
    
    # 3. Generate SDF
    sdf_root = create_simple_sdf(walls)
    
    # 4. Save to file
    rough_string = ET.tostring(sdf_root, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    with open(output_sdf, 'w') as f:
        f.write(pretty_xml)
    
    print(f"Conversion complete: {output_sdf}")
    return output_sdf


def process_directory(input_dir, output_dir, args):
    """Process all maps in a directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Find all yaml files
    yaml_files = [f for f in os.listdir(input_dir) if f.endswith('.yaml')]
    yaml_files.sort()
    
    if not yaml_files:
        print(f"No YAML files found in {input_dir}")
        return

    print(f"Found {len(yaml_files)} maps in {input_dir}")
    
    success_count = 0
    
    for yaml_file in yaml_files:
        yaml_path = os.path.join(input_dir, yaml_file)
        
        try:
            # Parse YAML to find image
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                image_file = data.get('image', '')
                
            if not image_file:
                print(f"Skipping {yaml_file}: No image specified in YAML")
                continue
                
            # Handle relative paths in YAML
            pgm_path = os.path.join(input_dir, image_file)
            if not os.path.exists(pgm_path):
                print(f"Skipping {yaml_file}: Image file {pgm_path} not found")
                continue
                
            # Determine output filename
            base_name = os.path.splitext(yaml_file)[0]
            output_sdf = os.path.join(output_dir, f"{base_name}.sdf")
            
            print(f"Processing {base_name}...")
            simple_pgm_to_sdf(
                pgm_path,
                yaml_path, 
                output_sdf,
                wall_height=args.height,
                wall_thickness=args.thickness,
                threshold=args.threshold,
                sensitivity=args.sensitivity,
                method=args.method,
                min_area=args.min_area,
                approx_epsilon=args.approx_epsilon
            )
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            
    print(f"\nBatch processing complete: {success_count}/{len(yaml_files)} successful")


def main():
    parser = argparse.ArgumentParser(description='PGM/YAML to SDF converter with multiple detection methods')
    parser.add_argument('input', help='Path to PGM map file OR Directory containing maps')
    parser.add_argument('yaml_file', nargs='?', help='Path to YAML metadata file (required if input is a file)')
    parser.add_argument('-o', '--output', default=None, help='Output SDF file (for single file) or Directory (for batch)')
    parser.add_argument('--height', type=float, default=2.0, help='Wall height (default: 2.0m)')
    parser.add_argument('--thickness', type=float, default=0.1, help='Wall thickness (default: 0.1m, only for hough method)')
    parser.add_argument('--threshold', type=int, default=128, help='Wall threshold (default: 128)')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Line detection sensitivity (default: 1.0, only for hough method)')
    parser.add_argument('--method', choices=['hough', 'contour', 'pixel', 'rle'], default='rle',
                       help='Detection method: hough (SLAM maps), contour (clean maps), pixel (simple maps), rle (orthogonal mazes - RECOMMENDED)')
    parser.add_argument('--min-area', type=int, default=10, help='Minimum contour area (default: 10, for contour/pixel methods)')
    parser.add_argument('--approx-epsilon', type=float, default=2.0, help='Contour approximation epsilon (default: 2.0)')

    args = parser.parse_args()

    # Check if input is directory
    if os.path.isdir(args.input):
        output_dir = args.output if args.output else args.input
        process_directory(args.input, output_dir, args)
        return

    # Single file mode
    if not args.yaml_file:
        print("Error: yaml_file argument is required when input is a file")
        parser.print_help()
        return

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return
    if not os.path.exists(args.yaml_file):
        print(f"Error: {args.yaml_file} not found")
        return

    output_file = args.output if args.output else 'slam_world.sdf'

    try:
        # Conversion with selected method
        simple_pgm_to_sdf(
            args.input,
            args.yaml_file, 
            output_file,
            wall_height=args.height,
            wall_thickness=args.thickness,
            threshold=args.threshold,
            sensitivity=args.sensitivity,
            method=args.method,
            min_area=args.min_area,
            approx_epsilon=args.approx_epsilon
        )
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
