#!/usr/bin/env python3
"""
SDF2MAP CLI - Command-line SDF/World to Map Converter
Converts SDF/World files to ROS2 compatible PGM/YAML map files
"""
import sys
import os
import math
import yaml
import argparse
import numpy as np
from PIL import Image, ImageDraw

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*numpy.*')

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except:
    SCIPY_AVAILABLE = False

from sdf_parser import SDFParser

# =============== 2D PROJECTION MAP GENERATION ===============

def compute_world_bounds(shapes, margin=1.0):
    """Compute world bounds from shapes"""
    if not shapes:
        return -5.0, -5.0, 5.0, 5.0
        
    bounds = []
    for s in shapes:
        x, y, z = s['position']
        if s['type'] in ['box', 'mesh', 'plane', 'polyline']:
            sx, sy = s['size'][0], s['size'][1]
            bounds.extend([(x-sx/2, y-sy/2), (x+sx/2, y+sy/2)])
        elif s['type'] in ['cylinder', 'sphere']:
            r = s['size'][0]
            bounds.extend([(x-r, y-r), (x+r, y+r)])
    
    xs, ys = zip(*bounds)
    return min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin


def draw_mesh_projection(draw, shape, world_to_px, model_resolver, verbose=False):
    """
    Draw mesh using 2D projection with INTERNAL STRUCTURE
    Renders all triangles, not just the outer hull
    """
    if not model_resolver:
        return False
    
    mesh_uri = shape.get('resolved_uri') or shape.get('uri')
    if not mesh_uri:
        return False
    
    transform = shape['transform']
    mesh_scale = shape.get('mesh_scale', (1.0, 1.0, 1.0))
    
    # Get 2D projection
    projected_2d = model_resolver.get_mesh_2d_projection(mesh_uri, transform, mesh_scale)
    
    if projected_2d is None or len(projected_2d) < 3:
        if verbose:
            print(f"  Could not project mesh: {os.path.basename(mesh_uri)}")
        return False
    
    # Convert to pixel coordinates
    pixel_points = []
    for point_2d in projected_2d:
        px, py = world_to_px(point_2d[0], point_2d[1])
        pixel_points.append((px, py))
    
    # Draw ALL triangles (internal structure included) - NO OUTLINES
    if len(pixel_points) >= 3:
        triangles_drawn = 0
        
        # STL files store vertices in groups of 3 (triangles)
        for i in range(0, len(pixel_points) - 2, 3):
            triangle = pixel_points[i:i+3]
            if len(triangle) == 3:
                draw.polygon(triangle, fill=0, outline=None)
                triangles_drawn += 1
        
        if verbose:
            print(f"  Drew {triangles_drawn} triangles for {os.path.basename(mesh_uri)}")
        return True
    
    return False


def draw_single_shape(draw, shape, world_to_px, scale_factor, resolution, model_resolver=None, verbose=False):
    """Draw shape with 2D projection support for meshes"""
    shape_type = shape['type']
    x, y, z = shape['position']
    roll, pitch, yaw = shape['rotation']
    size = shape['size']
    
    if shape_type == 'box':
        sx, sy, sz = size
        half_x, half_y = sx/2, sy/2
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
        pixel_corners = []
        
        for cx, cy in corners:
            rx = cx * cos_yaw - cy * sin_yaw + x
            ry = cx * sin_yaw + cy * cos_yaw + y
            pixel_corners.append(world_to_px(rx, ry))
        
        if len(pixel_corners) >= 3:
            draw.polygon(pixel_corners, fill=0, outline=0)
            
    elif shape_type == 'polyline':
        if 'points' not in shape:
            return

        x_offset, y_offset, _ = shape['position']
        points = shape['points']
        height = shape.get('height', 1.0)

        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        transformed_points = []

        for px, py in points:
            rx = px * cos_yaw - py * sin_yaw + x_offset
            ry = px * sin_yaw + py * cos_yaw + y_offset
            transformed_points.append(world_to_px(rx, ry))

        if len(transformed_points) >= 2:
            thickness = max(1, int(height * scale_factor / resolution / 20))
            draw.line(transformed_points, fill=0, width=thickness)

    elif shape_type == 'cylinder':
        radius, length = size[0], size[1] if len(size) > 1 else 1.0
        tilt_factor = abs(math.sin(pitch)) + abs(math.sin(roll))
        
        if tilt_factor > 0.1:
            semi_major = max(radius, (length / 2) * tilt_factor)
            semi_minor = radius * (1 - tilt_factor * 0.3)
            points = []
            for angle in range(0, 360, 10):
                rad = math.radians(angle)
                ex = semi_major * math.cos(rad)
                ey = semi_minor * math.sin(rad)
                rotated_ex = ex * math.cos(yaw) - ey * math.sin(yaw) + x
                rotated_ey = ex * math.sin(yaw) + ey * math.cos(yaw) + y
                points.append(world_to_px(rotated_ex, rotated_ey))
            if len(points) > 2:
                draw.polygon(points, fill=0, outline=0)
        else:
            px, py = world_to_px(x, y)
            rp = max(2, int(math.ceil(radius * scale_factor / resolution)))
            draw.ellipse([px-rp, py-rp, px+rp, py+rp], fill=0, outline=0)
            
    elif shape_type == 'sphere':
        radius = size[0]
        px, py = world_to_px(x, y)
        rp = max(2, int(math.ceil(radius * scale_factor / resolution)))
        draw.ellipse([px-rp, py-rp, px+rp, py+rp], fill=0, outline=0)
        
    elif shape_type == 'mesh':
        # Try 2D projection first
        if shape.get('use_projection') and model_resolver:
            success = draw_mesh_projection(draw, shape, world_to_px, model_resolver, verbose)
            if success:
                return
        
    elif shape_type == 'plane':
        sx, sy, sz = size
        half_x, half_y = sx/2, sy/2
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
        pixel_corners = []
        
        for cx, cy in corners:
            rx = cx * cos_yaw - cy * sin_yaw + x
            ry = cx * sin_yaw + cy * cos_yaw + y
            pixel_corners.append(world_to_px(rx, ry))
        
        if len(pixel_corners) >= 3:
            draw.polygon(pixel_corners, fill=0, outline=0)


def shapes_to_occupancy_with_projection(shapes, resolution=0.05, margin=1.0, min_wall_thickness=0.1, 
                                       lidar_height=None, map_size_scale=1.0, super_sampling=4,
                                       model_resolver=None, verbose=False):
    """Generate occupancy map with 2D mesh projection support"""
    # Filter by LIDAR height
    if lidar_height is not None:
        lidar_height_m = lidar_height / 100.0
        filtered_shapes = []
        for shape in shapes:
            z_center = shape['position'][2]
            
            if shape['type'] in ['box', 'mesh', 'plane', 'polyline']:
                z_size = shape['size'][2] if len(shape['size']) > 2 else 1.0
                z_min, z_max = z_center - z_size/2, z_center + z_size/2
            elif shape['type'] == 'cylinder':
                z_size = shape['size'][1] if len(shape['size']) > 1 else 1.0
                z_min, z_max = z_center - z_size/2, z_center + z_size/2
            elif shape['type'] == 'sphere':
                radius = shape['size'][0]
                z_min, z_max = z_center - radius, z_center + radius
            else:
                z_min, z_max = z_center - 0.5, z_center + 0.5
            
            if z_min <= lidar_height_m <= z_max:
                filtered_shapes.append(shape)
        shapes = filtered_shapes
    
    # Calculate dimensions
    minx, miny, maxx, maxy = compute_world_bounds(shapes, margin)
    width_m, height_m = maxx - minx, maxy - miny
    
    effective_resolution = resolution / map_size_scale
    w = max(100, int(math.ceil(width_m / effective_resolution)))
    h = max(100, int(math.ceil(height_m / effective_resolution)))
    
    scale_factor = super_sampling
    high_res_w, high_res_h = w * scale_factor, h * scale_factor
    
    if verbose:
        print(f"Generating map: {w}x{h} pixels (High-res: {high_res_w}x{high_res_h})")
        print(f"World: {width_m:.2f}x{height_m:.2f}m, Resolution: {effective_resolution:.4f}m/px")
    
    # Create high-res image
    high_res_img = Image.new('L', (high_res_w, high_res_h), color=255)
    high_res_draw = ImageDraw.Draw(high_res_img)
    
    def world_to_px_high_res(world_x, world_y):
        px = int((world_x - minx) * scale_factor / effective_resolution)
        py = high_res_h - 1 - int((world_y - miny) * scale_factor / effective_resolution)
        return max(0, min(high_res_w-1, px)), max(0, min(high_res_h-1, py))
    
    # Sort by area
    sorted_shapes = sorted(shapes, key=lambda s: s['size'][0] * s['size'][1] if len(s['size']) >= 2 else 1.0, reverse=True)
    
    # Apply minimum thickness and draw
    for shape in sorted_shapes:
        shape_type = shape['type']
        
        if shape_type in ['box', 'mesh', 'plane']:
            sx, sy, sz = shape['size']
            
            if sx < min_wall_thickness or sy < min_wall_thickness:
                shape = shape.copy()
                new_sx = max(sx, min_wall_thickness)
                new_sy = max(sy, min_wall_thickness)
                shape['size'] = (new_sx, new_sy, sz)
                
                if shape_type == 'mesh' and (sx < 0.5 or sy < 0.5):
                    shape['size'] = (
                        max(new_sx, min_wall_thickness * 3),
                        max(new_sy, min_wall_thickness * 3),
                        sz
                    )
        
        draw_single_shape(high_res_draw, shape, world_to_px_high_res, scale_factor, 
                         effective_resolution, model_resolver, verbose)
    
    # Resize to final
    final_img = high_res_img.resize((w, h), Image.LANCZOS)
    
    # Enhance contrast
    img_array = np.array(final_img)
    enhanced = np.where(img_array > 200, 255, img_array)
    enhanced = np.where(enhanced < 100, 0, enhanced)
    
    mask = (enhanced >= 100) & (enhanced <= 200)
    enhanced[mask] = np.where(enhanced[mask] < 150, 0, 255)
    
    # Morphological operations
    if map_size_scale > 1.0 and SCIPY_AVAILABLE:
        try:
            enhanced = ndimage.binary_opening(enhanced < 128, structure=np.ones((2,2)))
            enhanced = np.where(enhanced, 0, 255).astype(np.uint8)
        except:
            pass
    
    return Image.fromarray(enhanced.astype(np.uint8)), (minx, miny, 0.0)


def convert_sdf_to_map(input_file, output_dir, resolution=0.05, lidar_height=20, 
                       margin=1.0, min_wall_thickness=0.1, map_size_scale=1.0,
                       super_sampling=4, occupied_thresh=0.65, free_thresh=0.25,
                       user_model_paths=None, skip_missing=False, verbose=False):
    """
    Convert SDF/World file to PGM and YAML map files
    
    Args:
        input_file: Path to SDF or World file
        output_dir: Directory to save output files
        resolution: Map resolution in meters/pixel (default: 0.05)
        lidar_height: LIDAR height in cm (default: 20)
        margin: World margin in meters (default: 1.0)
        min_wall_thickness: Minimum wall thickness in meters (default: 0.1)
        map_size_scale: Map size scale multiplier (default: 1.0)
        super_sampling: Super sampling factor for quality (default: 4)
        occupied_thresh: Occupied threshold for YAML (default: 0.65)
        free_thresh: Free threshold for YAML (default: 0.25)
        user_model_paths: Dict mapping model names to file paths
        skip_missing: Skip missing models instead of failing
        verbose: Print detailed progress
        
    Returns:
        Tuple of (pgm_file_path, yaml_file_path) on success
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Input: {input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Resolution: {resolution} m/px")
        print(f"LIDAR height: {lidar_height} cm")
        print(f"Map size scale: {map_size_scale}x")
        print(f"Super sampling: {super_sampling}x")
    
    # Parse SDF file
    if verbose:
        print("\nParsing SDF file...")
    
    def status_callback(msg):
        if verbose:
            print(f"  {msg}")
    
    sdf_parser = SDFParser()
    parsed_data = sdf_parser.parse_file(
        input_file,
        user_model_paths or {},
        skip_missing,
        status_callback if verbose else None
    )
    
    missing_models = sdf_parser.get_missing_models()
    if missing_models and not skip_missing:
        print(f"\nWarning: Missing models: {missing_models}")
        print("Use --skip-missing to continue without them, or provide paths with --model-path")
    
    shapes = sdf_parser.get_shapes()
    
    if not shapes:
        raise Exception("No shapes found in the file")
    
    stats = sdf_parser.get_statistics()
    if verbose:
        print(f"\nFound {stats['total_shapes']} shapes:")
        for shape_type, count in stats['shape_types'].items():
            print(f"  {shape_type}: {count}")
    
    # Generate map
    if verbose:
        print("\nGenerating map with 2D mesh projection...")
    
    img, origin = shapes_to_occupancy_with_projection(
        shapes,
        resolution=resolution,
        margin=margin,
        min_wall_thickness=min_wall_thickness,
        lidar_height=lidar_height,
        map_size_scale=map_size_scale,
        super_sampling=super_sampling,
        model_resolver=sdf_parser.model_resolver,
        verbose=verbose
    )
    
    # Save files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    pgm_file = os.path.join(output_dir, f"{base_name}.pgm")
    yaml_file = os.path.join(output_dir, f"{base_name}.yaml")
    
    if verbose:
        print(f"\nSaving files...")
    
    # Save PGM
    img_array = np.array(img)
    with open(pgm_file, 'wb') as f:
        f.write(b'P5\n')
        f.write(f'{img_array.shape[1]} {img_array.shape[0]}\n'.encode())
        f.write(b'255\n')
        f.write(img_array.tobytes())
    
    # Round numbers for cleaner output
    def round_num(x):
        return round(float(x), 5)
    
    yaml_content = {
        'image': os.path.basename(pgm_file),
        'mode': 'trinary',
        'resolution': round_num(resolution),
        'origin': [round_num(origin[0]), round_num(origin[1]), round_num(origin[2])],
        'negate': 0,
        'occupied_thresh': round_num(occupied_thresh),
        'free_thresh': round_num(free_thresh)
    }
    
    # Custom representer to print short lists inline
    def represent_list(dumper, data):
        if len(data) <= 3:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
    
    yaml.add_representer(list, represent_list)
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    if verbose:
        print(f"\nConversion complete!")
        print(f"  PGM: {pgm_file}")
        print(f"  YAML: {yaml_file}")
        print(f"  Map size: {img.size[0]}x{img.size[1]} pixels")
    
    return pgm_file, yaml_file


def main():
    parser = argparse.ArgumentParser(
        description='SDF2MAP CLI - Convert SDF/World files to ROS2 map files (PGM/YAML)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.sdf
  %(prog)s world.world -r 0.02 --lidar-height 30
  %(prog)s scene.sdf -o ./output --map-scale 1.5 --super-sampling 6
  %(prog)s model.sdf --model-path wall_mesh=/path/to/wall.stl
        """
    )
    
    # Required arguments
    parser.add_argument('input', help='Input SDF or World file')
    parser.add_argument('-o', '--output', default=None, 
                        help='Output directory (default: same as input file)')
    
    # Map parameters
    parser.add_argument('-r', '--resolution', type=float, default=0.05,
                        help='Map resolution in meters/pixel (default: 0.05)')
    parser.add_argument('--lidar-height', type=int, default=20,
                        help='LIDAR height in cm (default: 20)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='World margin in meters (default: 1.0)')
    parser.add_argument('--min-wall', type=float, default=0.1,
                        help='Minimum wall thickness in meters (default: 0.1)')
    parser.add_argument('--map-scale', type=float, default=1.0,
                        help='Map size scale multiplier (default: 1.0)')
    parser.add_argument('--super-sampling', type=int, default=4, choices=[1, 2, 4, 6, 8],
                        help='Super sampling factor for quality (default: 4)')
    
    # YAML thresholds
    parser.add_argument('--occupied-thresh', type=float, default=0.65,
                        help='Occupied threshold for YAML (default: 0.65)')
    parser.add_argument('--free-thresh', type=float, default=0.25,
                        help='Free threshold for YAML (default: 0.25)')
    
    # Model resolution
    parser.add_argument('--model-path', action='append', metavar='NAME=PATH',
                        help='Map model name to file path (can be used multiple times)')
    parser.add_argument('--skip-missing', action='store_true',
                        help='Skip missing models instead of failing')
    
    # Presets
    parser.add_argument('--preset', choices=['fast', 'standard', 'high'],
                        help='Use preset configuration (fast/standard/high)')
    
    # Output options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed progress')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress all output except errors')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset:
        presets = {
            'fast': {'resolution': 0.1, 'map_scale': 1.0, 'super_sampling': 2},
            'standard': {'resolution': 0.05, 'map_scale': 1.0, 'super_sampling': 4},
            'high': {'resolution': 0.02, 'map_scale': 1.5, 'super_sampling': 6},
        }
        preset = presets[args.preset]
        args.resolution = preset['resolution']
        args.map_scale = preset['map_scale']
        args.super_sampling = preset['super_sampling']
    
    # Parse model paths
    user_model_paths = {}
    if args.model_path:
        for mapping in args.model_path:
            if '=' in mapping:
                name, path = mapping.split('=', 1)
                user_model_paths[name.strip()] = path.strip()
            else:
                print(f"Warning: Invalid model path format: {mapping}")
                print("  Expected format: NAME=PATH")
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Default output directory to input file's directory
    output_dir = args.output if args.output else os.path.dirname(os.path.abspath(args.input))
    if not output_dir:
        output_dir = '.'
    
    try:
        pgm_file, yaml_file = convert_sdf_to_map(
            input_file=args.input,
            output_dir=output_dir,
            resolution=args.resolution,
            lidar_height=args.lidar_height,
            margin=args.margin,
            min_wall_thickness=args.min_wall,
            map_size_scale=args.map_scale,
            super_sampling=args.super_sampling,
            occupied_thresh=args.occupied_thresh,
            free_thresh=args.free_thresh,
            user_model_paths=user_model_paths,
            skip_missing=args.skip_missing,
            verbose=verbose
        )
        
        if not args.quiet:
            print(f"Generated: {pgm_file}")
            print(f"Generated: {yaml_file}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
