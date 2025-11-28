#!/usr/bin/env python3
"""
Maze Robot PGM Generator
------------------------
Generates a maze using mazegenerator.net (via requests) and converts it 
to a ROS 2 compatible PGM/YAML map.

The map resolution is calculated based on the robot's footprint to ensure the 
maze is solvable (corridors are wide enough).

Usage:
    python3 maze_robot_pgm_genrator.py --width 20 --height 20 --robot-size 0.5
"""

import os
import sys
import time
import argparse
import random
import yaml
import re
import numpy as np
import requests
import cairosvg
from PIL import Image
from io import BytesIO

def generate_maze_image(width, height, shape_idx=1, style_idx=1, inner_width=0, inner_height=0, starts_at=1, e_val=50, r_val=100):
    """
    Interact with mazegenerator.net to generate a maze using requests.
    Returns the PIL Image of the maze.
    """
    url = "https://www.mazegenerator.net/"
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    })
    
    print(f"Connecting to {url}...")
    r = s.get(url)
    
    # Extract hidden fields
    viewstate = re.search(r'id="__VIEWSTATE" value="([^"]+)"', r.text)
    viewstategen = re.search(r'id="__VIEWSTATEGENERATOR" value="([^"]+)"', r.text)
    eventvalidation = re.search(r'id="__EVENTVALIDATION" value="([^"]+)"', r.text)
    
    if not viewstate or not eventvalidation:
        raise Exception("Failed to parse form fields from website")

    data = {
        "__VIEWSTATE": viewstate.group(1),
        "__VIEWSTATEGENERATOR": viewstategen.group(1) if viewstategen else "",
        "__EVENTVALIDATION": eventvalidation.group(1),
        "ShapeDropDownList": str(shape_idx),
        "S1TesselationDropDownList": str(style_idx),
        "S1WidthTextBox": str(width),
        "S1HeightTextBox": str(height),
        "S1InnerWidthTextBox": str(inner_width),
        "S1InnerHeightTextBox": str(inner_height),
        "S1StartsAtDropDownList": str(starts_at),
        "AlgorithmParameter1TextBox": str(e_val),
        "AlgorithmParameter2TextBox": str(r_val),
        "GenerateButton": "Generate"
    }
    
    print(f"Generating maze ({width}x{height}, Shape:{shape_idx}, Style:{style_idx})...")
    r_post = s.post(url, data=data)
    
    # Find image URL
    # Look for ImageGenerator.ashx
    match = re.search(r'src="(ImageGenerator\.ashx[^"]+)"', r_post.text)
    
    if not match:
        # Try to find any image that looks like a maze if the ID changed
        # Or maybe the response is different
        raise Exception("Could not find maze image URL in response")
        
    img_rel_url = match.group(1).replace("&amp;", "&")
    img_url = url + img_rel_url
    
    print(f"Downloading maze from {img_url[:50]}...")
    r_img = s.get(img_url)
    
    if r_img.status_code == 200:
        content = r_img.content
        # Check if SVG
        if b"<svg" in content or b"<!DOCTYPE svg" in content:
            print("Detected SVG format. Converting to PNG...")
            try:
                png_data = cairosvg.svg2png(bytestring=content)
                image = Image.open(BytesIO(png_data))
                return image
            except Exception as e:
                print(f"SVG conversion failed: {e}")
                raise e
        else:
            try:
                image = Image.open(BytesIO(content))
                return image
            except Exception as e:
                print(f"Invalid image content. First 100 bytes: {content[:100]}")
                raise e
    else:
        raise Exception(f"Failed to download image: {r_img.status_code}")

def process_maze_image(image, robot_size, maze_width_cells):
    """
    Convert maze image to PGM and calculate resolution.
    
    Args:
        image: PIL Image object
        robot_size: Robot footprint diameter in meters
        maze_width_cells: Number of cells in the maze width
        
    Returns:
        pgm_image: PIL Image object (grayscale)
        resolution: meters per pixel
    """
    # Handle transparency: Create a white background
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Convert to RGBA to ensure we have an alpha channel
        image = image.convert('RGBA')
        # Create a white background image
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        # Paste the image on top of the background using the alpha channel as mask
        bg.alpha_composite(image)
        image = bg.convert('RGB')
    
    # Convert to grayscale
    gray_img = image.convert("L")
    
    # Threshold to binary (0=black/wall, 255=white/free)
    # The website usually produces black walls and white paths.
    # We want: Occupied (0) = Black, Free (255) = White.
    # Let's ensure high contrast.
    np_img = np.array(gray_img)
    
    # Debug: Print stats to help diagnose "all black" issues
    print(f"  Image Stats - Min: {np_img.min()}, Max: {np_img.max()}, Mean: {np_img.mean():.2f}")
    
    binary_img = np.where(np_img < 128, 0, 255).astype(np.uint8)
    pgm_img = Image.fromarray(binary_img)
    
    # Calculate Resolution
    # We want the robot to fit in the corridors.
    # A corridor is 1 cell wide.
    # Let's assume a safety factor.
    safety_factor = 1.5
    required_cell_size_m = robot_size * safety_factor
    
    # How many pixels is one cell?
    # Image width (px) / Maze width (cells) = Pixels per cell
    img_width_px = pgm_img.width
    pixels_per_cell = img_width_px / maze_width_cells
    
    # Resolution = Meters / Pixel
    # Resolution = Cell Size (m) / Cell Size (px)
    resolution = required_cell_size_m / pixels_per_cell
    
    print(f"Map Analysis:")
    print(f"  Image Width: {img_width_px} px")
    print(f"  Maze Width: {maze_width_cells} cells")
    print(f"  Pixels/Cell: {pixels_per_cell:.2f}")
    print(f"  Robot Size: {robot_size} m")
    print(f"  Target Cell Size: {required_cell_size_m:.2f} m")
    print(f"  Calculated Resolution: {resolution:.5f} m/px")
    
    return pgm_img, resolution

def save_map(image, resolution, output_dir, base_name):
    """Save PGM and YAML files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pgm_path = os.path.join(output_dir, f"{base_name}.pgm")
    yaml_path = os.path.join(output_dir, f"{base_name}.yaml")
    
    # Save PGM
    image.save(pgm_path)
    
    # Create YAML
    # Origin: [x, y, yaw]
    # Usually bottom-left.
    # We'll set origin so the map is centered or starts at (0,0).
    # Let's start at (0,0).
    
    map_data = {
        'image': f"{base_name}.pgm",
        'resolution': float(resolution),
        'origin': [0.0, 0.0, 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(map_data, f, default_flow_style=False)
        
    print(f"Saved map to:")
    print(f"  {pgm_path}")
    print(f"  {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ROS PGM map from mazegenerator.net")
    
    # General
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of maps to generate")
    parser.add_argument("--output-dir", default="generated_maps", help="Output directory")
    parser.add_argument("--name", default="maze", help="Base name for output files")
    
    # Robot Parameters
    parser.add_argument("--robot-size", type=float, default=0.5, help="Robot footprint diameter (meters)")
    
    # Maze Parameters
    parser.add_argument("--width", type=int, default=20, help="Maze width in cells")
    parser.add_argument("--height", type=int, default=20, help="Maze height in cells")
    parser.add_argument("--inner-width", type=int, default=0, help="Inner room width in cells")
    parser.add_argument("--inner-height", type=int, default=0, help="Inner room height in cells")
    
    parser.add_argument("--shape", type=int, choices=[1, 2, 3, 4], 
                        help="Shape (1=Rect, 2=Circ, 3=Tri, 4=Hex). Default: Random (Rect only for now)")
    parser.add_argument("--style", type=int, choices=[1, 2, 3], 
                        help="Style (1=Ortho, 2=Sigma, 3=Delta). Default: Random")
    
    parser.add_argument("--starts-at", type=int, choices=[1, 2], 
                        help="Starts At (1=Top, 2=Bottom/Inner). Default: Random")
    
    parser.add_argument("--e", type=int, help="Elitism (0-100). Default: Random")
    parser.add_argument("--r", type=int, help="River (0-100). Default: Random")
    
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Handle Randomness
    if args.seed:
        random.seed(args.seed)
    
    print("WARNING: Mazes from mazegenerator.net are for personal use only unless licensed.")
    print("See https://www.mazegenerator.net/CommercialUse.aspx for details.")
    
    for i in range(args.count):
        print(f"\n--- Generating Maze {i+1}/{args.count} ---")
        
        # Determine parameters for this iteration
        # Shape: Currently defaulting to 1 (Rect) because other shapes might use different form fields
        # If user explicitly sets shape, we try it. If not, we stick to 1 to be safe.
        shape = args.shape if args.shape is not None else 1 
        
        # Style: Random if not set
        # Valid styles depend on shape, but for Rect (1), all 3 are valid.
        style = args.style if args.style is not None else random.choice([1, 2, 3])
        
        # Starts At: Random if not set
        starts_at = args.starts_at if args.starts_at is not None else random.choice([1, 2])
        
        # E and R: Random if not set
        e_val = args.e if args.e is not None else random.randint(0, 100)
        r_val = args.r if args.r is not None else random.randint(0, 100)
        
        try:
            # Generate
            image = generate_maze_image(
                width=args.width, 
                height=args.height, 
                shape_idx=shape, 
                style_idx=style, 
                inner_width=args.inner_width,
                inner_height=args.inner_height,
                starts_at=starts_at,
                e_val=e_val, 
                r_val=r_val
            )
            
            # Process
            pgm_img, resolution = process_maze_image(image, args.robot_size, args.width)
            
            # Save
            suffix = f"_{i}" if args.count > 1 else ""
            save_map(pgm_img, resolution, args.output_dir, f"{args.name}{suffix}")
            
        except Exception as e:
            print(f"Failed to generate maze {i+1}: {e}")
            # Continue to next maze if one fails?
            # sys.exit(1) 
            pass

if __name__ == "__main__":
    main()
