#!/usr/bin/env python3
"""
Maze Robot PGM Generator
------------------------
Generates a maze using mazegenerator.net (via requests) and converts it 
to a ROS 2 compatible PGM/YAML map.

The map resolution is calculated based on the robot's footprint to ensure the 
maze is solvable (corridors are wide enough).

Supported Shapes and Parameters:
--------------------------------
1. Rectangular (shape=1):
   - Styles: 1=Orthogonal (square cells), 2=Sigma (hexagonal cells), 3=Delta (triangular cells)
   - Width: 2-200 cells
   - Height: 2-200 cells
   - Inner Width: 0 or 2 to (width-2) cells
   - Inner Height: 0 or 2 to (height-2) cells
   - Starts At: 1=Top, 2=Bottom

2. Circular/Theta (shape=2):
   - No style option (always theta maze)
   - Outer Diameter: 5-200 cells
   - Inner Diameter: 3 to (outer-2), difference must be even
   - Starts At: 1=Outer edge, 2=Center

3. Triangular (shape=3):
   - No style option (always delta/triangular cells)
   - Side Length: 3-200 cells
   - Inner Side Length: 0, or 3+ with (side - inner_side) divisible by 3
   - Starts At: 1=Top, 2=Bottom/Inner

4. Hexagonal (shape=4):
   - Styles: 1=Sigma (hexagonal cells), 2=Delta (triangular cells)
   - Side Length: 1-120 cells
   - Inner Side Length: 0 to (side-1)
   - Starts At: 1=Outer edge, 2=Center

Usage:
    # Rectangular maze (default)
    python3 maze_robot_pgm_genrator.py --width 20 --height 20 --robot-size 0.5
    
    # Circular/Theta maze
    python3 maze_robot_pgm_genrator.py --shape 2 --outer-diameter 20 --inner-diameter 5
    
    # Triangular maze
    python3 maze_robot_pgm_genrator.py --shape 3 --side-length 20
    
    # Hexagonal maze with sigma style
    python3 maze_robot_pgm_genrator.py --shape 4 --side-length 10 --style 1
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

# Shape constants
SHAPE_RECTANGULAR = 1
SHAPE_CIRCULAR = 2
SHAPE_TRIANGULAR = 3
SHAPE_HEXAGONAL = 4

# Style constants for rectangular mazes
STYLE_ORTHOGONAL = 1  # Square cells
STYLE_SIGMA = 2       # Hexagonal cells
STYLE_DELTA = 3       # Triangular cells

# Shape name mapping
SHAPE_NAMES = {
    SHAPE_RECTANGULAR: "Rectangular",
    SHAPE_CIRCULAR: "Circular/Theta",
    SHAPE_TRIANGULAR: "Triangular",
    SHAPE_HEXAGONAL: "Hexagonal"
}

# Style name mapping per shape
STYLE_NAMES = {
    SHAPE_RECTANGULAR: {1: "Orthogonal", 2: "Sigma", 3: "Delta"},
    SHAPE_CIRCULAR: {1: "Theta"},
    SHAPE_TRIANGULAR: {1: "Delta"},
    SHAPE_HEXAGONAL: {1: "Sigma", 2: "Delta"}
}


def get_valid_styles_for_shape(shape):
    """Return valid style indices for a given shape."""
    if shape == SHAPE_RECTANGULAR:
        return [1, 2, 3]  # Orthogonal, Sigma, Delta
    elif shape == SHAPE_CIRCULAR:
        return [1]  # Theta only (no style selection)
    elif shape == SHAPE_TRIANGULAR:
        return [1]  # Delta only (no style selection)
    elif shape == SHAPE_HEXAGONAL:
        return [1, 2]  # Sigma, Delta
    return [1]


def validate_circular_params(outer_diameter, inner_diameter):
    """Validate and adjust circular maze parameters."""
    # Outer diameter: 5-200
    outer_diameter = max(5, min(200, outer_diameter))
    
    # Inner diameter: 3 to outer-2, difference must be even
    if inner_diameter > 0:
        inner_diameter = max(3, min(outer_diameter - 2, inner_diameter))
        # Ensure difference is even
        diff = outer_diameter - inner_diameter
        if diff % 2 != 0:
            inner_diameter -= 1
        inner_diameter = max(3, inner_diameter)
    
    return outer_diameter, inner_diameter


def validate_triangular_params(side_length, inner_side_length):
    """Validate and adjust triangular maze parameters."""
    # Side length: 3-200
    side_length = max(3, min(200, side_length))
    
    # Inner side length: 0, or 3+ with (side - inner_side) divisible by 3
    if inner_side_length > 0:
        inner_side_length = max(3, inner_side_length)
        diff = side_length - inner_side_length
        if diff < 3:
            inner_side_length = 0
        elif diff % 3 != 0:
            # Adjust to make divisible by 3
            inner_side_length = side_length - ((diff // 3) * 3)
            if inner_side_length < 3:
                inner_side_length = 0
    
    return side_length, inner_side_length


def validate_hexagonal_params(side_length, inner_side_length):
    """Validate and adjust hexagonal maze parameters."""
    # Side length: 1-120
    side_length = max(1, min(120, side_length))
    
    # Inner side length: 0 to side-1
    if inner_side_length > 0:
        inner_side_length = max(1, min(side_length - 1, inner_side_length))
    
    return side_length, inner_side_length


def validate_rectangular_params(width, height, inner_width, inner_height):
    """Validate and adjust rectangular maze parameters."""
    # Width/Height: 2-200
    width = max(2, min(200, width))
    height = max(2, min(200, height))
    
    # Inner width: 0 or 2 to width-2
    if inner_width > 0:
        inner_width = max(2, min(width - 2, inner_width))
    
    # Inner height: 0 or 2 to height-2
    if inner_height > 0:
        inner_height = max(2, min(height - 2, inner_height))
    
    return width, height, inner_width, inner_height


def build_form_data(shape, style, params, starts_at, e_val, r_val, viewstate, viewstategen, eventvalidation):
    """
    Build the form data dictionary for the POST request based on shape.
    
    Different shapes use different form field names on mazegenerator.net.
    """
    data = {
        "__VIEWSTATE": viewstate,
        "__VIEWSTATEGENERATOR": viewstategen if viewstategen else "",
        "__EVENTVALIDATION": eventvalidation,
        "ShapeDropDownList": str(shape),
        "AlgorithmParameter1TextBox": str(e_val),
        "AlgorithmParameter2TextBox": str(r_val),
        "GenerateButton": "Generate"
    }
    
    if shape == SHAPE_RECTANGULAR:
        width, height, inner_width, inner_height = params
        data.update({
            "S1TesselationDropDownList": str(style),
            "S1WidthTextBox": str(width),
            "S1HeightTextBox": str(height),
            "S1InnerWidthTextBox": str(inner_width),
            "S1InnerHeightTextBox": str(inner_height),
            "S1StartsAtDropDownList": str(starts_at),
        })
    elif shape == SHAPE_CIRCULAR:
        outer_diameter, inner_diameter = params
        data.update({
            "S2OuterDiameterTextBox": str(outer_diameter),
            "S2InnerDiameterTextBox": str(inner_diameter),
            "S2StartsAtDropDownList": str(starts_at),
        })
    elif shape == SHAPE_TRIANGULAR:
        side_length, inner_side_length = params
        data.update({
            "S3SideLengthTextBox": str(side_length),
            "S3InnerSideLengthTextBox": str(inner_side_length),
            "S3StartsAtDropDownList": str(starts_at),
        })
    elif shape == SHAPE_HEXAGONAL:
        side_length, inner_side_length = params
        data.update({
            "S4TesselationDropDownList": str(style),
            "S4SideLengthTextBox": str(side_length),
            "S4InnerSideLengthTextBox": str(inner_side_length),
            "S4StartsAtDropDownList": str(starts_at),
        })
    
    return data


def generate_maze_image(shape=1, style=1, params=None, starts_at=1, e_val=50, r_val=100):
    """
    Interact with mazegenerator.net to generate a maze using requests.
    
    Args:
        shape: Shape index (1=Rectangular, 2=Circular, 3=Triangular, 4=Hexagonal)
        style: Style index (depends on shape)
        params: Tuple of size parameters depending on shape:
            - Rectangular: (width, height, inner_width, inner_height)
            - Circular: (outer_diameter, inner_diameter)
            - Triangular: (side_length, inner_side_length)
            - Hexagonal: (side_length, inner_side_length)
        starts_at: 1=Top/Outer, 2=Bottom/Center
        e_val: Elitism parameter (0-100)
        r_val: River parameter (0-100)
    
    Returns:
        PIL Image of the maze.
    """
    if params is None:
        params = (20, 20, 0, 0)  # Default rectangular params
    
    url = "https://www.mazegenerator.net/"
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    })
    
    print(f"Connecting to {url}...")
    r = s.get(url)
    
    # Extract hidden fields
    def extract_form_fields(html):
        viewstate = re.search(r'id="__VIEWSTATE" value="([^"]+)"', html)
        viewstategen = re.search(r'id="__VIEWSTATEGENERATOR" value="([^"]+)"', html)
        eventvalidation = re.search(r'id="__EVENTVALIDATION" value="([^"]+)"', html)
        return viewstate, viewstategen, eventvalidation
    
    viewstate, viewstategen, eventvalidation = extract_form_fields(r.text)
    
    if not viewstate or not eventvalidation:
        raise Exception("Failed to parse form fields from website")

    # For non-rectangular shapes, we need to first change the shape via postback
    # This updates the form to show the correct fields for that shape
    if shape != SHAPE_RECTANGULAR:
        # First POST: Change the shape (triggers ASP.NET postback)
        shape_change_data = {
            "__VIEWSTATE": viewstate.group(1),
            "__VIEWSTATEGENERATOR": viewstategen.group(1) if viewstategen else "",
            "__EVENTVALIDATION": eventvalidation.group(1),
            "__EVENTTARGET": "ShapeDropDownList",
            "__EVENTARGUMENT": "",
            "ShapeDropDownList": str(shape),
            "S1TesselationDropDownList": "1",
            "S1WidthTextBox": "20",
            "S1HeightTextBox": "20",
            "S1InnerWidthTextBox": "0",
            "S1InnerHeightTextBox": "0",
            "S1StartsAtDropDownList": "1",
            "AlgorithmParameter1TextBox": str(e_val),
            "AlgorithmParameter2TextBox": str(r_val),
        }
        
        r_shape = s.post(url, data=shape_change_data)
        
        # Extract the new form fields after shape change
        viewstate, viewstategen, eventvalidation = extract_form_fields(r_shape.text)
        
        if not viewstate or not eventvalidation:
            raise Exception("Failed to parse form fields after shape change")
    
    # Build form data based on shape
    data = build_form_data(
        shape=shape,
        style=style,
        params=params,
        starts_at=starts_at,
        e_val=e_val,
        r_val=r_val,
        viewstate=viewstate.group(1),
        viewstategen=viewstategen.group(1) if viewstategen else "",
        eventvalidation=eventvalidation.group(1)
    )
    
    # Create descriptive log message based on shape
    shape_name = SHAPE_NAMES.get(shape, f"Shape {shape}")
    style_name = STYLE_NAMES.get(shape, {}).get(style, f"Style {style}")
    
    if shape == SHAPE_RECTANGULAR:
        size_desc = f"{params[0]}x{params[1]}"
    elif shape == SHAPE_CIRCULAR:
        size_desc = f"outer={params[0]}, inner={params[1]}"
    else:
        size_desc = f"side={params[0]}, inner={params[1]}"
    
    print(f"Generating maze ({shape_name}, {style_name}, {size_desc})...")
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

def save_map(pgm_image, resolution, output_dir, base_name, original_image=None):
    """Save PGM, YAML, and optionally the original PNG files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pgm_path = os.path.join(output_dir, f"{base_name}.pgm")
    yaml_path = os.path.join(output_dir, f"{base_name}.yaml")
    png_path = os.path.join(output_dir, f"{base_name}.png")
    
    # Save original PNG if provided
    if original_image is not None:
        original_image.save(png_path)
    
    # Save PGM
    pgm_image.save(pgm_path)
    
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
    if original_image is not None:
        print(f"  {png_path}")
    print(f"  {pgm_path}")
    print(f"  {yaml_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate ROS PGM map from mazegenerator.net",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shapes and their parameters:
  1. Rectangular: --width, --height, --inner-width, --inner-height
                  Styles: 1=Orthogonal, 2=Sigma, 3=Delta
  2. Circular:    --outer-diameter, --inner-diameter
                  No style options (always Theta)
  3. Triangular:  --side-length, --inner-side-length
                  No style options (always Delta)
  4. Hexagonal:   --side-length, --inner-side-length
                  Styles: 1=Sigma, 2=Delta

Examples:
  # Rectangular orthogonal maze
  python3 maze_robot_pgm_genrator.py --shape 1 --width 20 --height 20 --style 1
  
  # Circular theta maze
  python3 maze_robot_pgm_genrator.py --shape 2 --outer-diameter 20 --inner-diameter 5
  
  # Triangular maze
  python3 maze_robot_pgm_genrator.py --shape 3 --side-length 20
  
  # Hexagonal sigma maze
  python3 maze_robot_pgm_genrator.py --shape 4 --side-length 10 --style 1
  
  # Random shape and style (default behavior)
  python3 maze_robot_pgm_genrator.py
  
  # Force rectangular shape only
  python3 maze_robot_pgm_genrator.py --no-random-shape
        """
    )
    
    # General
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of maps to generate")
    parser.add_argument("--output-dir", default="generated_maps", help="Output directory")
    parser.add_argument("--name", default="maze", help="Base name for output files")
    
    # Robot Parameters
    parser.add_argument("--robot-size", type=float, default=0.5, help="Robot footprint diameter (meters)")
    
    # Shape Selection
    parser.add_argument("--shape", type=int, choices=[1, 2, 3, 4], 
                        help="Shape: 1=Rectangular, 2=Circular, 3=Triangular, 4=Hexagonal. Default: Random")
    parser.add_argument("--no-random-shape", action="store_true",
                        help="Disable random shape selection (use Rectangular by default)")
    
    # Style (for Rectangular and Hexagonal)
    parser.add_argument("--style", type=int, choices=[1, 2, 3], 
                        help="Style (shape-dependent): Rect: 1=Ortho, 2=Sigma, 3=Delta; Hex: 1=Sigma, 2=Delta. Default: Random")
    
    # Rectangular Maze Parameters (Shape 1)
    parser.add_argument("--width", type=int, default=20, help="[Rectangular] Maze width in cells (2-200)")
    parser.add_argument("--height", type=int, default=20, help="[Rectangular] Maze height in cells (2-200)")
    parser.add_argument("--inner-width", type=int, default=0, help="[Rectangular] Inner room width (0 or 2 to width-2)")
    parser.add_argument("--inner-height", type=int, default=0, help="[Rectangular] Inner room height (0 or 2 to height-2)")
    
    # Circular Maze Parameters (Shape 2)
    parser.add_argument("--outer-diameter", type=int, default=20, help="[Circular] Outer diameter in cells (5-200)")
    parser.add_argument("--inner-diameter", type=int, default=0, help="[Circular] Inner diameter (0, or 3 to outer-2, diff must be even)")
    
    # Triangular Maze Parameters (Shape 3)
    parser.add_argument("--side-length", type=int, default=20, help="[Triangular/Hexagonal] Side length in cells")
    parser.add_argument("--inner-side-length", type=int, default=0, help="[Triangular/Hexagonal] Inner side length")
    
    # Common Parameters
    parser.add_argument("--starts-at", type=int, choices=[1, 2], 
                        help="Starts At: 1=Top/Outer, 2=Bottom/Center. Default: Random")
    
    # Algorithm Parameters
    parser.add_argument("--e", type=int, help="Elitism (0-100): Low=solution uses more cells. Default: Random")
    parser.add_argument("--r", type=int, help="River (0-100): High=fewer but longer dead ends. Default: Random")
    
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Handle Randomness
    if args.seed:
        random.seed(args.seed)
    
    print("WARNING: Mazes from mazegenerator.net are for personal use only unless licensed.")
    print("See https://www.mazegenerator.net/CommercialUse.aspx for details.")
    
    for i in range(args.count):
        print(f"\n--- Generating Maze {i+1}/{args.count} ---")
        
        # Determine shape (random by default if not specified)
        if args.shape is not None:
            shape = args.shape
        elif args.no_random_shape:
            shape = SHAPE_RECTANGULAR
        else:
            shape = random.choice([SHAPE_RECTANGULAR, SHAPE_CIRCULAR, SHAPE_TRIANGULAR, SHAPE_HEXAGONAL])
        
        # Determine style based on shape
        valid_styles = get_valid_styles_for_shape(shape)
        if args.style is not None and args.style in valid_styles:
            style = args.style
        else:
            style = random.choice(valid_styles)
        
        # Determine size parameters based on shape
        if shape == SHAPE_RECTANGULAR:
            width, height, inner_width, inner_height = validate_rectangular_params(
                args.width, args.height, args.inner_width, args.inner_height
            )
            params = (width, height, inner_width, inner_height)
            maze_size_for_resolution = width  # Use width for resolution calculation
            
        elif shape == SHAPE_CIRCULAR:
            outer_diameter, inner_diameter = validate_circular_params(
                args.outer_diameter, args.inner_diameter
            )
            params = (outer_diameter, inner_diameter)
            maze_size_for_resolution = outer_diameter
            
        elif shape == SHAPE_TRIANGULAR:
            side_length, inner_side_length = validate_triangular_params(
                args.side_length, args.inner_side_length
            )
            params = (side_length, inner_side_length)
            maze_size_for_resolution = side_length
            
        elif shape == SHAPE_HEXAGONAL:
            side_length, inner_side_length = validate_hexagonal_params(
                args.side_length, args.inner_side_length
            )
            params = (side_length, inner_side_length)
            maze_size_for_resolution = side_length
        
        # Starts At: Random if not set
        starts_at = args.starts_at if args.starts_at is not None else random.choice([1, 2])
        
        # E and R: Random if not set
        e_val = args.e if args.e is not None else random.randint(0, 100)
        r_val = args.r if args.r is not None else random.randint(0, 100)
        
        try:
            # Generate
            image = generate_maze_image(
                shape=shape,
                style=style,
                params=params,
                starts_at=starts_at,
                e_val=e_val, 
                r_val=r_val
            )
            
            # Process
            pgm_img, resolution = process_maze_image(image, args.robot_size, maze_size_for_resolution)
            
            # Save (including original PNG)
            suffix = f"_{i}" if args.count > 1 else ""
            save_map(pgm_img, resolution, args.output_dir, f"{args.name}{suffix}", original_image=image)
            
        except Exception as e:
            print(f"Failed to generate maze {i+1}: {e}")
            import traceback
            traceback.print_exc()
            pass

if __name__ == "__main__":
    main()
