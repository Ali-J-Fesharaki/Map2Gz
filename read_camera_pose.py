#!/usr/bin/env python3
"""
Read the current camera pose from Gazebo.
Launch Gazebo with an SDF, manually unzoom to your desired view,
then this script will read the camera position and orientation.
"""

import subprocess
import sys
import time
import os

def get_camera_pose():
    """Try to get camera pose from Gazebo GUI topic"""
    
    # Try to echo the camera pose topic (one message)
    try:
        result = subprocess.run(
            ['ign', 'topic', '-e', '-t', '/gui/camera/pose', '-n', '1'],
            capture_output=True, text=True, timeout=3
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    return None

def parse_pose(pose_str):
    """Parse pose string to extract position and orientation"""
    lines = pose_str.split('\n')
    data = {}
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line == 'position {':
            current_section = 'position'
            data['position'] = {}
        elif line == 'orientation {':
            current_section = 'orientation'
            data['orientation'] = {}
        elif line == '}':
            current_section = None
        elif current_section and ':' in line:
            key, value = line.split(':')
            key = key.strip()
            value = value.strip()
            try:
                data[current_section][key] = float(value)
            except:
                pass
    
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_camera_pose.py <sdf_file>")
        print("Example: python read_camera_pose.py maps/generated/maze_batch_1.sdf")
        sys.exit(1)
    
    sdf_file = sys.argv[1]
    if not os.path.exists(sdf_file):
        print(f"Error: SDF file not found: {sdf_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Camera Pose Reader - Unzoom Test")
    print("=" * 60)
    
    # Kill existing Gazebo
    print("\nKilling existing Gazebo processes...")
    subprocess.run(['pkill', '-9', 'ruby'], capture_output=True)
    subprocess.run(['pkill', '-9', 'gzserver'], capture_output=True)
    subprocess.run(['pkill', '-9', 'gzclient'], capture_output=True)
    subprocess.run(['pkill', '-9', 'ign'], capture_output=True)
    time.sleep(1)
    
    # Launch Gazebo
    print(f"\nLaunching Gazebo with: {sdf_file}")
    gz_process = subprocess.Popen(
        ['ign', 'gazebo', sdf_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    print("\nWaiting for Gazebo to start...")
    time.sleep(5)
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("1. Gazebo should now be open")
    print("2. Use SCROLL WHEEL to UNZOOM to your desired view")
    print("3. Adjust the view as needed (drag to rotate)")
    print("4. Press ENTER here when ready to capture the camera pose")
    print("=" * 60)
    
    # Loop to capture multiple poses as user unzooms
    pose_count = 0
    while True:
        input(f"\nPress ENTER to capture camera pose #{pose_count + 1} (or Ctrl+C to quit)...")
        
        pose = get_camera_pose()
        if pose:
            pose_count += 1
            print(f"\n=== Camera Pose #{pose_count} ===")
            print(pose)
            
            # Parse and show summary
            data = parse_pose(pose)
            if 'position' in data and 'orientation' in data:
                pos = data['position']
                ori = data['orientation']
                print(f"\n--- Summary ---")
                print(f"Position: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
                print(f"Orientation (quaternion): w={ori.get('w', 1):.4f}, x={ori.get('x', 0):.4f}, y={ori.get('y', 0):.4f}, z={ori.get('z', 0):.4f}")
        else:
            print("Could not read camera pose. Is Gazebo running?")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        subprocess.run(['pkill', '-9', 'ruby'], capture_output=True)
        subprocess.run(['pkill', '-9', 'ign'], capture_output=True)
