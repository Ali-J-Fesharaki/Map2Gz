#!/usr/bin/env python3
"""
SDF to GIF Converter
--------------------
Iterates through SDF files in a directory, launches Ignition Gazebo,
orbits the camera around the world, captures frames, and creates a GIF.

Usage:
    python3 sdf_to_gif.py --input maps/generated --output gifs
"""

import os
import sys
import time
import subprocess
import argparse
import math
import glob
import shutil
import re
import numpy as np
import signal
from PIL import Image
import xml.etree.ElementTree as ET

def kill_gazebo_processes():
    """Kill all gazebo/ruby processes forcefully"""
    try:
        # Kill ruby processes (used by ign/gz)
        subprocess.run(['pkill', '-9', '-f', 'ruby.*ign'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'ruby.*gz'], stderr=subprocess.DEVNULL)
        # Kill gzserver and gzclient
        subprocess.run(['pkill', '-9', 'gzserver'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', 'gzclient'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'ign gazebo'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'gz sim'], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception as e:
        print(f"Warning: Could not kill processes: {e}")

def get_world_bounds_from_sdf(sdf_file):
    """Parse SDF file to get the bounding box of all models"""
    try:
        tree = ET.parse(sdf_file)
        root = tree.getroot()
        
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        # Find all pose elements
        for model in root.iter('model'):
            pose_elem = model.find('pose')
            if pose_elem is not None and pose_elem.text:
                parts = pose_elem.text.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        if min_x == float('inf'):
            return 0, 0, 15.0  # Default center and radius
            
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calculate appropriate radius based on world size
        world_size = max(max_x - min_x, max_y - min_y)
        radius = world_size * 0.8  # 80% of world size for good view
        radius = max(radius, 5.0)  # Minimum radius of 5m
        
        print(f"World bounds: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}]")
        print(f"World center: ({center_x:.1f}, {center_y:.1f}), Orbit radius: {radius:.1f}")
        
        return center_x, center_y, radius
        
    except Exception as e:
        print(f"Warning: Could not parse SDF bounds: {e}")
        return 0, 0, 15.0

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to Quaternion"""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return {'x': qx, 'y': qy, 'z': qz, 'w': qw}

def calculate_lookat_quaternion(eye_x, eye_y, eye_z, target_x, target_y, target_z):
    """
    Calculates the quaternion to point a camera at a target.
    Uses Gazebo convention where camera looks along a direction.
    
    Args:
        eye_x, eye_y, eye_z: Camera position
        target_x, target_y, target_z: Point to look at
    
    Returns:
        dict with quaternion components {x, y, z, w}
    """
    # Vector from eye to target (forward direction)
    dx = target_x - eye_x
    dy = target_y - eye_y
    dz = target_z - eye_z
    
    # Distance
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    
    if dist == 0:
        return {'x': 0, 'y': 0, 'z': 0, 'w': 1}
    
    # Normalize forward vector
    fx = dx / dist
    fy = dy / dist
    fz = dz / dist
    
    # Calculate pitch (rotation around Y-axis, looking up/down)
    # pitch = -asin(fz) because looking down means positive pitch
    pitch = -math.asin(np.clip(fz, -1.0, 1.0))
    
    # Calculate yaw (rotation around Z-axis, left/right)
    yaw = math.atan2(fy, fx)
    
    # Roll is 0 (no banking)
    roll = 0.0
    
    # Convert to quaternion
    return euler_to_quaternion(roll, pitch, yaw)

def take_screenshot(cmd='ign'):
    """Trigger internal screenshot service - saves to default location (~/.ignition/gui/pictures/)"""
    # Service: /gui/screenshot
    # Empty string -> saves to default location
    
    msg_pkg = "ignition.msgs" if "ign" in cmd else "gz.msgs"
    
    cmd_args = [
        cmd, 'service', '-s', '/gui/screenshot',
        '--reqtype', f'{msg_pkg}.StringMsg',
        '--reptype', f'{msg_pkg}.Boolean',
        '--timeout', '2000',
        '--req', 'data: ""'
    ]
    
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Screenshot service failed: {result.stderr}")
        return False
    elif "data: false" in result.stdout.lower():
        print(f"Screenshot service returned false.")
        return False
    return True

def find_latest_file(directories, min_mtime):
    """Find the most recently modified file in a list of directories"""
    latest_file = None
    latest_time = 0
    
    for d in directories:
        if not os.path.exists(d):
            continue
            
        for root, dirs, files in os.walk(d):
            for f in files:
                if not f.endswith('.png'):
                    continue
                    
                fp = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(fp)
                    if mtime >= min_mtime and mtime > latest_time:
                        latest_time = mtime
                        latest_file = fp
                except Exception:
                    pass
                    
    return latest_file

def get_default_screenshot_dir(cmd='ign'):
    """Get the default directory where Gazebo saves screenshots"""
    home = os.path.expanduser("~")
    # The actual location is ~/.ignition/gui/pictures/ (or ~/.gz/gui/pictures/)
    return [
        os.path.join(home, ".ignition", "gui", "pictures"),
        os.path.join(home, ".gz", "gui", "pictures")
    ]

def focus_on_model(model_name, cmd='ign'):
    """Focus camera on a named model using /gui/move_to service"""
    cmd_args = [
        cmd, 'service', '-s', '/gui/move_to',
        '--reqtype', 'ignition.msgs.StringMsg',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '3000',
        '--req', f'data: "{model_name}"'
    ]
    
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    return result.returncode == 0

def read_camera_pose(cmd='ign', timeout=5.0):
    """Read current camera pose from /gui/camera/pose topic"""
    import re
    
    cmd_args = [cmd, 'topic', '-e', '-t', '/gui/camera/pose', '-n', '1']
    
    try:
        result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=timeout)
        output = result.stdout
        
        # Extract position block
        pos_block = re.search(r'position\s*\{([^}]*)\}', output, re.DOTALL)
        orient_block = re.search(r'orientation\s*\{([^}]*)\}', output, re.DOTALL)
        
        if pos_block and orient_block:
            pos_text = pos_block.group(1)
            orient_text = orient_block.group(1)
            
            # Parse individual values (default to 0 if not present)
            def get_val(text, key):
                match = re.search(rf'{key}:\s*([-\d.e+]+)', text)
                return float(match.group(1)) if match else 0.0
            
            x = get_val(pos_text, 'x')
            y = get_val(pos_text, 'y')
            z = get_val(pos_text, 'z')
            
            qx = get_val(orient_text, 'x')
            qy = get_val(orient_text, 'y')
            qz = get_val(orient_text, 'z')
            qw = get_val(orient_text, 'w')
            
            return (x, y, z, qx, qy, qz, qw)
        else:
            print(f"  Could not find position/orientation blocks in output")
    except subprocess.TimeoutExpired:
        print(f"  Timeout reading camera pose")
    except Exception as e:
        print(f"  Failed to read camera pose: {e}")
    
    return None

def move_camera(x, y, z, roll, pitch, yaw, cmd='ign'):
    """Send service call to move camera to absolute pose (Euler angles)"""
    q = euler_to_quaternion(roll, pitch, yaw)
    return move_camera_quat(x, y, z, q['x'], q['y'], q['z'], q['w'], cmd)

def move_camera_quat(x, y, z, qx, qy, qz, qw, cmd='ign'):
    """Send service call to move camera using quaternion orientation directly"""
    # Construct Protobuf-like string for the service call
    req_str = (
        f'pose: {{ '
        f'position: {{ x: {x}, y: {y}, z: {z} }}, '
        f'orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }} '
        f'}}'
    )
    
    # Call the service (note: path is /gui/move_to/pose with slash, not underscore)
    cmd_args = [
        cmd, 'service', '-s', '/gui/move_to/pose',
        '--reqtype', 'ignition.msgs.GUICamera',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '2000',
        '--req', req_str
    ]
    
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Camera move failed: {result.stderr[:100] if result.stderr else 'unknown error'}")

def wait_for_service(service_name, timeout=30, cmd='ign'):
    """Wait for a service to become available"""
    print(f"Waiting for service {service_name}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # List services
            output = subprocess.check_output([cmd, 'service', '-l'], stderr=subprocess.DEVNULL).decode()
            if service_name in output:
                print(f"Service {service_name} is available.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"Timeout waiting for {service_name}. Is the GUI loaded?")
    return False

def capture_frames(sdf_file, output_gif, frames=60, radius=15.0, height=10.0, cmd='ign'):
    """Launch Gazebo, orbit camera, capture frames, create GIF"""
    
    print(f"\n{'='*60}")
    print(f"Processing {sdf_file}...")
    print(f"{'='*60}")
    
    # Kill any existing gazebo processes first
    print("Killing existing Gazebo processes...")
    kill_gazebo_processes()
    
    # Get world center and appropriate radius from SDF
    center_x, center_y, auto_radius = get_world_bounds_from_sdf(sdf_file)
    
    # Use auto-calculated radius if not specified
    if radius == 15.0:  # default value
        radius = auto_radius
    
    # Ensure we are using absolute paths for the SDF file
    sdf_abs_path = os.path.abspath(sdf_file)
    
    # Launch Gazebo
    print(f"Launching Gazebo with {sdf_abs_path}...")
    gz_process = subprocess.Popen([cmd, 'gazebo', sdf_abs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    try:
        # Wait for Gazebo to load and services to be ready
        if not wait_for_service('/gui/screenshot', timeout=60, cmd=cmd):
            print("Skipping capture: Screenshot service not found.")
            return

        # Create persistent dir for frames
        base_name = os.path.splitext(os.path.basename(sdf_file))[0]
        frames_dir = os.path.abspath(os.path.join("frames", base_name))
        
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)
        
        print(f"Capturing frames to {frames_dir}...")
        captured_images = []
        
        # Wait for scene to fully render
        print("Waiting for scene to render...")
        time.sleep(4)
        
        # STEP 1: Focus on ground_plane model to center the camera (ground mode)
        print("STEP 1: Focusing on ground_plane (centering camera)...")
        focus_on_model("ground_plane", cmd)
        time.sleep(2.0)  # Wait for focus animation to complete
        
        # STEP 2: Read the camera pose after focus (this is our starting position)
        print("STEP 2: Reading camera pose after focus...")
        pose = read_camera_pose(cmd)
        
        if pose:
            start_x, start_y, start_z, _, _, _, _ = pose  # Ignore orientation, we'll recalculate
            print(f"  Camera position after focus: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")
        else:
            # Fallback if we can't read pose
            print("  Warning: Could not read camera pose, using defaults")
            start_x = center_x + radius * 0.5
            start_y = center_y + radius * 0.5
            start_z = radius * 0.5
        
        # STEP 3: Setup Vector Unzoom
        # ---------------------------------------------------------
        # Target point (Where we are looking at - center of maze at ground level)
        focal_x, focal_y, focal_z = center_x, center_y, 0.0
        
        # Vector from Target TO Camera (The "Backwards" vector)
        vec_x = start_x - focal_x
        vec_y = start_y - focal_y
        vec_z = start_z - focal_z
        
        # Calculate current distance (magnitude)
        current_dist = math.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
        
        # Normalize the vector (Direction unit vector)
        if current_dist > 0:
            dir_x = vec_x / current_dist
            dir_y = vec_y / current_dist
            dir_z = vec_z / current_dist
        else:
            # Fallback if camera is exactly at the target (unlikely)
            dir_x, dir_y, dir_z = 1.0, 0.0, 1.0
            current_dist = radius
        
        # Define Zoom limits
        start_distance = current_dist
        end_distance = current_dist * 3.5  # Zoom out to 3.5x distance
        
        print(f"STEP 3: Vector Unzoom Setup")
        print(f"  Target (focal point): ({focal_x:.1f}, {focal_y:.1f}, {focal_z:.1f})")
        print(f"  Direction vector: <{dir_x:.3f}, {dir_y:.3f}, {dir_z:.3f}>")
        print(f"  Distance: {start_distance:.1f}m → {end_distance:.1f}m")
        
        # STEP 4: Animation Loop - Capture frames while unzooming
        print(f"STEP 4: Capturing {frames} frames...")
        for i in range(frames):
            # Linear Interpolation of the Distance
            t = i / max(frames - 1, 1)
            new_dist = start_distance + (end_distance - start_distance) * t
            
            # Calculate New Camera Position: Pos = Target + (Direction * NewDistance)
            cam_x = focal_x + (dir_x * new_dist)
            cam_y = focal_y + (dir_y * new_dist)
            cam_z = focal_z + (dir_z * new_dist)
            
            # Recalculate Orientation to keep target perfectly centered (LookAt)
            q = calculate_lookat_quaternion(cam_x, cam_y, cam_z, focal_x, focal_y, focal_z)
            
            print(f"  Frame {i+1}/{frames}: dist={new_dist:.1f}m → pos=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
            
            # Move Camera with calculated LookAt orientation
            move_camera_quat(cam_x, cam_y, cam_z, q['x'], q['y'], q['z'], q['w'], cmd)
            
            # Wait for camera to settle
            time.sleep(0.4)
            
            # Capture screen using internal service (saves to default location)
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            
            # Get file count in default directories before screenshot
            search_dirs = get_default_screenshot_dir(cmd)
            files_before = {}
            for d in search_dirs:
                if os.path.exists(d):
                    try:
                        files_before[d] = set(os.listdir(d))
                    except:
                        files_before[d] = set()
            
            # Take screenshot
            call_time = time.time()
            success = take_screenshot(cmd)
            
            if not success:
                print(f"Warning: Screenshot service call failed for frame {i}")
                continue
            
            # Wait and look for new file in default directories
            timeout = 5.0
            start_wait = time.time()
            found = False
            latest_file = None
            
            while (time.time() - start_wait) < timeout:
                # Check for new files
                for d in search_dirs:
                    if not os.path.exists(d):
                        continue
                    try:
                        files_now = set(os.listdir(d))
                        new_files = files_now - files_before.get(d, set())
                        
                        for nf in new_files:
                            if nf.endswith('.png'):
                                fp = os.path.join(d, nf)
                                latest_file = fp
                                found = True
                                break
                    except:
                        pass
                    
                    if found:
                        break
                
                if found:
                    break
                    
                time.sleep(0.2)
            
            if found and latest_file:
                # Move it to the expected location
                try:
                    shutil.move(latest_file, frame_path)
                    captured_images.append(frame_path)
                    print(f"Captured frame {i+1}/{frames}", end='\r')
                except Exception as e:
                    print(f"Error moving file: {e}")
                    found = False
            
            if not found:
                print(f"Warning: Frame {i} not found after {timeout}s")
            
        print("\nGenerating GIF...")
        
        if not captured_images:
            print(f"Error: No frames captured for {sdf_file}. Skipping GIF generation.")
            return

        # Create GIF
        images = [Image.open(f) for f in captured_images]
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=100, # ms per frame
            loop=0
        )
        
        print(f"Saved {output_gif}")
        
    finally:
        # Kill Gazebo and all related processes
        print("Closing Gazebo...")
        gz_process.terminate()
        try:
            gz_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            gz_process.kill()
        
        # Force kill all related processes
        kill_gazebo_processes()
        
        print("Done with this world.")

def main():
    parser = argparse.ArgumentParser(description="Generate GIFs from SDF files using Ignition Gazebo")
    parser.add_argument("input", nargs="?", default=None, help="Single SDF file or directory containing SDF files")
    parser.add_argument("-o", "--output", default=None, help="Output GIF file (for single SDF) or directory (for batch)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames per GIF")
    parser.add_argument("--radius", type=float, default=15.0, help="Orbit radius")
    parser.add_argument("--height", type=float, default=10.0, help="Camera height")
    parser.add_argument("--cmd", default="ign", help="Command prefix (ign or gz)")
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.input if args.input else "maps/generated"
    
    # Check if input is a single file or directory
    if os.path.isfile(input_path):
        # Single file mode
        if not input_path.endswith('.sdf'):
            print(f"Error: {input_path} is not an SDF file")
            return
        
        # Determine output path
        if args.output:
            output_gif = args.output
            # Ensure output directory exists
            output_dir = os.path.dirname(output_gif)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_gif = f"gifs/{base_name}.gif"
            if not os.path.exists("gifs"):
                os.makedirs("gifs")
        
        print(f"Processing single file: {input_path} -> {output_gif}")
        capture_frames(input_path, output_gif, args.frames, args.radius, args.height, args.cmd)
    
    elif os.path.isdir(input_path):
        # Directory/batch mode
        output_dir = args.output if args.output else "gifs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        sdf_files = glob.glob(os.path.join(input_path, "*.sdf"))
        sdf_files.sort()
        
        if not sdf_files:
            print(f"No SDF files found in {input_path}")
            return
            
        print(f"Found {len(sdf_files)} SDF files")
        
        for sdf_file in sdf_files:
            base_name = os.path.splitext(os.path.basename(sdf_file))[0]
            output_gif = os.path.join(output_dir, f"{base_name}.gif")
            
            if os.path.exists(output_gif):
                print(f"Skipping {base_name} (GIF exists)")
                continue
                
            capture_frames(sdf_file, output_gif, args.frames, args.radius, args.height, args.cmd)
    else:
        print(f"Error: {input_path} does not exist")

if __name__ == "__main__":
    main()
