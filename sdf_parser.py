#!/usr/bin/env python3
"""
Enhanced SDF Parser with TRUE 2D Mesh Projection (Gazebo-like rendering)
Reads actual mesh vertices and projects them to 2D map
"""

import os
import math
import numpy as np
import xml.etree.ElementTree as ET
# import requests
# import tarfile
# import tempfile
# import shutil

# =============== Utility Functions ===============

def rotation_matrix_from_rpy(roll, pitch, yaw):
    """Create rotation matrix from RPY (ZYX order)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def transform_matrix_from_pose(pose):
    """Create 4x4 transformation matrix from pose"""
    x, y, z, roll, pitch, yaw = pose
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix_from_rpy(roll, pitch, yaw)
    transform[:3, 3] = [x, y, z]
    return transform

def parse_pose(pose_text):
    """Parse SDF pose string to (x,y,z,roll,pitch,yaw)"""
    if not pose_text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        vals = [float(p) for p in pose_text.strip().split()]
        while len(vals) < 6:
            vals.append(0.0)
        return tuple(vals[:6])
    except:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

def parse_scale(scale_text):
    """Parse scale from mesh or model"""
    if not scale_text:
        return (1.0, 1.0, 1.0)
    try:
        vals = [float(p) for p in scale_text.strip().split()]
        return (vals[0], vals[0], vals[0]) if len(vals) == 1 else tuple(vals[:3]) if len(vals) >= 3 else (1.0, 1.0, 1.0)
    except:
        return (1.0, 1.0, 1.0)

# =============== 2D PROJECTION: Core Mesh Reading ===============

class MeshProjector:
    """Reads mesh files and projects vertices to 2D (like Gazebo)"""
    
    def __init__(self, max_vertices=10000):
        self.max_vertices = max_vertices
        self.cache = {}
        
    def load_mesh_vertices(self, mesh_path):
        """Load 3D vertices from mesh file"""
        if mesh_path in self.cache:
            return self.cache[mesh_path]
            
        if not os.path.exists(mesh_path):
            return None
            
        file_ext = os.path.splitext(mesh_path.lower())[1]
        vertices = None
        
        try:
            if file_ext == '.stl' or file_ext == '.STL':
                vertices = self.load_stl_vertices(mesh_path)
            elif file_ext == '.obj' or file_ext == '.OBJ':
                vertices = self.load_obj_vertices(mesh_path)
            elif file_ext == '.dae' or file_ext == '.DAE':
                vertices = self.load_dae_vertices(mesh_path)
            else:
                return None
                
            if vertices is not None and len(vertices) > 0:
                self.cache[mesh_path] = vertices
                return vertices
                
        except Exception as e:
            print(f"✗ Error loading mesh {mesh_path}: {e}")
            
        return None
    
    def load_stl_vertices(self, file_path):
        """Load vertices from STL file (binary or ASCII)"""
        vertices = []
        
        try:
            # Try binary STL
            with open(file_path, 'rb') as f:
                header = f.read(80)
                triangle_count_bytes = f.read(4)
                if len(triangle_count_bytes) == 4:
                    triangle_count = int.from_bytes(triangle_count_bytes, byteorder='little')
                    triangles_to_read = min(triangle_count, self.max_vertices // 3)
                    
                    for i in range(triangles_to_read):
                        f.read(12)  # Normal
                        for _ in range(3):  # 3 vertices
                            x = np.frombuffer(f.read(4), dtype=np.float32)[0]
                            y = np.frombuffer(f.read(4), dtype=np.float32)[0]
                            z = np.frombuffer(f.read(4), dtype=np.float32)[0]
                            vertices.append([x, y, z])
                        f.read(2)  # Attribute
                        
        except:
            # Try ASCII STL
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if len(vertices) >= self.max_vertices:
                            break
                        if line.strip().startswith('vertex'):
                            coords = line.split()[1:4]
                            if len(coords) >= 3:
                                vertices.append([float(c) for c in coords])
            except:
                pass
                
        return np.array(vertices) if vertices else None
    
    def load_obj_vertices(self, file_path):
        """Load vertices from OBJ file"""
        vertices = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if len(vertices) >= self.max_vertices:
                        break
                    line = line.strip()
                    if line.startswith('v '):  # Only vertex positions, not normals/textures
                        parts = line.split()
                        if len(parts) >= 4:  # 'v' + x + y + z
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                vertices.append([x, y, z])
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error reading OBJ: {e}")
            
        return np.array(vertices) if vertices else None
    
    def load_dae_vertices(self, file_path):
        """Load vertices from DAE (COLLADA) file - FIXED with normalization"""
        vertices = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Get namespace from root
            ns_match = root.tag[root.tag.find('{'): root.tag.find('}')+1] if '{' in root.tag else ''
            
            # Try multiple strategies
            strategies = [
                # Strategy 1: Direct float_array search
                lambda: root.findall(f'.//{ns_match}float_array'),
                # Strategy 2: Through geometry->mesh->source
                lambda: [src.find(f'{ns_match}float_array') 
                        for geom in root.findall(f'.//{ns_match}geometry')
                        for mesh in [geom.find(f'{ns_match}mesh')] if mesh is not None
                        for src in mesh.findall(f'{ns_match}source')],
                # Strategy 3: Wildcard namespace
                lambda: root.findall('.//{*}float_array'),
            ]
            
            for strategy in strategies:
                try:
                    float_arrays = strategy()
                    for float_array in float_arrays:
                        if float_array is None or not float_array.text:
                            continue
                            
                        array_id = (float_array.get('id') or '').lower()
                        
                        # Check if this is position data
                        is_position = any(kw in array_id for kw in ['position', 'vertex', 'coord', 'location'])
                        
                        # Also check stride/count to verify it's 3D data
                        count = int(float_array.get('count', 0))
                        if count > 0 and count % 3 == 0:
                            is_position = True
                        
                        if is_position:
                            try:
                                # Parse values
                                text = float_array.text.strip()
                                # Handle different separators
                                text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ')
                                values = [float(v) for v in text.split() if v]
                                
                                # Extract vertices (every 3 values)
                                for i in range(0, min(len(values) - 2, self.max_vertices * 3), 3):
                                    vertices.append([values[i], values[i+1], values[i+2]])
                                    if len(vertices) >= self.max_vertices:
                                        break
                                
                                if vertices:
                                    break
                                    
                            except (ValueError, IndexError) as e:
                                continue
                    
                    if vertices:
                        break
                        
                except Exception as e:
                    continue
            
            # Normalize vertices to fit in reasonable bounds
            if len(vertices) > 0:
                vertices = np.array(vertices)
                # Get bounding box
                min_coords = vertices.min(axis=0)
                max_coords = vertices.max(axis=0)
                size = max_coords - min_coords
                max_size = size.max()
                
                # Scale to unit cube if too large
                if max_size > 100:
                    center = (min_coords + max_coords) / 2
                    vertices = (vertices - center) / max_size * 2
                    
        except Exception as e:
            print(f"Error reading DAE: {e}")
            
        return np.array(vertices) if len(vertices) > 0 else None
    
    def project_to_2d(self, vertices_3d, transform_matrix, lidar_height_m=None, slice_tolerance=0.1):
        """
        Project 3D mesh vertices to 2D with LIDAR height slicing support
        
        Args:
            vertices_3d: 3D vertices array
            transform_matrix: 4x4 transformation matrix
            lidar_height_m: LIDAR height in meters (if None, projects all vertices)
            slice_tolerance: Tolerance range for slicing (meters)
        """
        if vertices_3d is None or len(vertices_3d) == 0:
            return None
        
        # Convert to homogeneous coordinates
        n = len(vertices_3d)
        vertices_homog = np.ones((n, 4))
        vertices_homog[:, :3] = vertices_3d
        
        # Apply transformation
        transformed = (transform_matrix @ vertices_homog.T).T
        
        # If LIDAR height specified, filter vertices within height range
        if lidar_height_m is not None:
            z_coords = transformed[:, 2]
            
            # Keep vertices within tolerance of LIDAR height
            # This creates a "slice" through the mesh
            mask = np.abs(z_coords - lidar_height_m) <= slice_tolerance
            
            if not np.any(mask):
                # If no vertices in slice, try getting closest vertices
                distances = np.abs(z_coords - lidar_height_m)
                closest_threshold = np.percentile(distances, 20)  # Keep closest 20%
                mask = distances <= closest_threshold
            
            transformed = transformed[mask]
            
            if len(transformed) == 0:
                return None
        
        # Extract 2D projection (X, Y only - top-down view)
        projected_2d = transformed[:, :2]
        
        return projected_2d

# =============== Enhanced Model Resolution ===============

class ModelResolver:
    def __init__(self):
        self.missing_models = []
        self.user_model_paths = {}
        self.mesh_projector = MeshProjector()
        
    def set_user_model_paths(self, user_paths):
        """Store user provided model/mesh paths"""
        if isinstance(user_paths, dict):
            self.user_model_paths = user_paths.copy()
        
    def resolve_model_uri(self, uri, status_callback=None):
        """Resolve model URI with proper Fuel/Gazebo download support"""
        # Check user paths first
        if self.user_model_paths:
            if uri in self.user_model_paths:
                path = self.user_model_paths[uri]
                if os.path.exists(path):
                    if status_callback:
                        status_callback(f"Using user-provided path for {uri}")
                    return path
            
            model_name = os.path.basename(uri).replace('model://', '')
            if model_name in self.user_model_paths:
                path = self.user_model_paths[model_name]
                if os.path.exists(path):
                    if status_callback:
                        status_callback(f"Using user-provided mesh file for {model_name}")
                    return path
            
            base_name = os.path.splitext(model_name)[0]
            if base_name in self.user_model_paths:
                path = self.user_model_paths[base_name]
                if os.path.exists(path):
                    if status_callback:
                        status_callback(f"Using user-provided file for {base_name}")
                    return path
        
        # Handle HTTP/HTTPS URIs
        # if uri.startswith(("http://", "https://")):
        #     try:
        #         base_download = os.path.expanduser("~/Downloads/models")
        #         os.makedirs(base_download, exist_ok=True)
                
        #         parts = uri.strip("/").split("/")
        #         if "models" not in parts:
        #             return None
                
        #         # Extract organization and model name
        #         org_index = parts.index("1.0") + 1 if "1.0" in parts else parts.index("models") - 1
        #         org = parts[org_index]
        #         model_name = parts[parts.index("models") + 1]
        #         model_dir = os.path.join(base_download, org, model_name)
                
        #         # Check if already exists
        #         if os.path.exists(model_dir):
        #             sdf_files = []
        #             for root, dirs, files in os.walk(model_dir):
        #                 sdf_files.extend([os.path.join(root, f) for f in files if f.endswith('.sdf')])
        #             if sdf_files:
        #                 if status_callback:
        #                     status_callback(f"Using cached model: {model_name}")
        #                 return model_dir
                
        #         if status_callback:
        #             status_callback(f"Downloading model: {model_name}...")
                
        #         # Try direct .tar.gz download (old Fuel format)
        #         download_url = uri + ".tar.gz"
        #         tmp_file = tempfile.mktemp(suffix=".tar.gz")
                
        #         r = requests.get(download_url, stream=True, timeout=30)
        #         if r.status_code == 200:
        #             with open(tmp_file, "wb") as f:
        #                 shutil.copyfileobj(r.raw, f)
                    
        #             os.makedirs(model_dir, exist_ok=True)
        #             with tarfile.open(tmp_file, "r:gz") as tar:
        #                 tar.extractall(model_dir)
        #             os.remove(tmp_file)
                    
        #             if status_callback:
        #                 status_callback(f"Successfully downloaded model: {model_name}")
        #             return model_dir
        #         else:
        #             # Try Fuel API format
        #             api_urls = [
        #                 f"https://fuel.gazebosim.org/1.0/{org}/models/{model_name}/tip/files/model.tar.gz",
        #                 f"https://fuel.ignitionrobotics.org/1.0/{org}/models/{model_name}/tip/files/model.tar.gz",
        #             ]
                    
        #             for api_url in api_urls:
        #                 try:
        #                     if status_callback:
        #                         status_callback(f"Trying: {api_url}")
        #                     r = requests.get(api_url, stream=True, timeout=30)
        #                     if r.status_code == 200:
        #                         with open(tmp_file, "wb") as f:
        #                             shutil.copyfileobj(r.raw, f)
                                
        #                         os.makedirs(model_dir, exist_ok=True)
        #                         with tarfile.open(tmp_file, "r:gz") as tar:
        #                             tar.extractall(model_dir)
        #                         os.remove(tmp_file)
                                
        #                         if status_callback:
        #                             status_callback(f"Successfully downloaded: {model_name}")
        #                         return model_dir
        #                 except:
        #                     continue
                    
        #             if status_callback:
        #                 status_callback(f"Failed to download model: {model_name} (HTTP {r.status_code})")
        #             return None
                
        #     except Exception as e:
        #         if status_callback:
        #             status_callback(f"Error downloading model: {str(e)}")
        #         return None
        
        if uri.startswith("model://"):
            model_path = uri.replace("model://", "")
            full_path = os.path.join(os.path.expanduser("~/Downloads/models"), model_path)
            return full_path if os.path.exists(full_path) else None
        
        return uri
    
    def get_mesh_2d_projection(self, mesh_path, transform_matrix, mesh_scale=(1.0, 1.0, 1.0), lidar_height_m=None):
        """
        Get 2D projection of mesh with enhanced path resolution
        """
        # Resolve actual path
        actual_path = mesh_path
        
        # If it's a URL or model:// URI, try to resolve it
        if mesh_path.startswith(("http://", "https://", "model://")):
            # Extract model name and mesh file name
            if "meshes/" in mesh_path:
                mesh_filename = os.path.basename(mesh_path)
                
                # Try to find it in downloaded models
                base_download = os.path.expanduser("~/Downloads/models")
                for root, dirs, files in os.walk(base_download):
                    if mesh_filename in files:
                        actual_path = os.path.join(root, mesh_filename)
                        break
            
            # Try direct resolution
            if not os.path.exists(actual_path):
                resolved = self.resolve_model_uri(mesh_path)
                if resolved and os.path.exists(resolved):
                    if os.path.isdir(resolved):
                        # Look for mesh file in directory
                        mesh_filename = os.path.basename(mesh_path)
                        for root, dirs, files in os.walk(resolved):
                            if mesh_filename in files:
                                actual_path = os.path.join(root, mesh_filename)
                                break
                    else:
                        actual_path = resolved
        
        # Try user paths if still not found
        if not os.path.exists(actual_path):
            base_name = os.path.basename(mesh_path)
            name_no_ext = os.path.splitext(base_name)[0]
            
            for key, path in self.user_model_paths.items():
                if (key == base_name or key == name_no_ext or 
                    key == mesh_path or os.path.basename(path) == base_name):
                    if os.path.exists(path):
                        actual_path = path
                        break
        
        if not os.path.exists(actual_path):
            return None
        
        # Load 3D vertices
        vertices_3d = self.mesh_projector.load_mesh_vertices(actual_path)
        if vertices_3d is None:
            return None
        
        # Apply mesh scale
        scale_matrix = np.diag([mesh_scale[0], mesh_scale[1], mesh_scale[2], 1.0])
        full_transform = transform_matrix @ scale_matrix
        
        # Project to 2D
        projected_2d = self.mesh_projector.project_to_2d(vertices_3d, full_transform, lidar_height_m)
        
        return projected_2d
    
    def estimate_mesh_bounds(self, mesh_uri, mesh_scale=(1.0, 1.0, 1.0)):
        """Fallback: estimate bounds from filename"""
        filename = os.path.basename(mesh_uri or "").lower()
        
        size_map = {
            'wall': (2.0, 0.2, 2.5),
            'door': (1.0, 0.1, 2.1),
            'window': (1.5, 0.1, 1.2),
            'table': (1.2, 0.8, 0.75),
            'chair': (0.5, 0.5, 0.9),
            'house': (10.0, 10.0, 3.0),
            'car': (4.5, 2.0, 1.5),
            'tree': (2.0, 2.0, 5.0),
        }
        
        for keyword, size in size_map.items():
            if keyword in filename:
                return tuple(s * ms for s, ms in zip(size, mesh_scale))
        
        return (1.0, 1.0, 1.0)

# =============== Geometry Processing ===============

def parse_polyline_points(polyline_elem):
    """Parse polyline points"""
    points = []
    height = 1.0
    
    for point_elem in polyline_elem.findall('point'):
        try:
            coords = [float(x) for x in point_elem.text.split()]
            if len(coords) >= 2:
                points.append((coords[0], coords[1]))
        except:
            continue
            
    height_elem = polyline_elem.find('height')
    if height_elem is not None:
        try:
            height = float(height_elem.text)
        except:
            pass
            
    return points, height

def process_includes(world_root, model_resolver, user_model_paths=None, skip_missing=False, status_callback=None):
    """Process include statements"""
    processed = []
    seen_uris = set()
    
    if user_model_paths:
        model_resolver.set_user_model_paths(user_model_paths)
    
    for include in world_root.findall('.//include'):
        uri_elem = include.find('uri')
        if uri_elem is None:
            continue
            
        uri = uri_elem.text.strip()
        model_name = include.findtext('name', os.path.basename(uri))
        model_pose = parse_pose(include.findtext('pose'))
        
        model_key = (uri, model_name, tuple(model_pose))
        if model_key in seen_uris:
            continue
        seen_uris.add(model_key)
        
        included_model = None
        
        # Try user paths
        if user_model_paths and isinstance(user_model_paths, dict):
            model_path = uri.replace('model://', '')
            path_keys = [uri, model_path, model_path.split('/')[0], 
                        os.path.basename(model_path), model_name]
            
            for key in path_keys:
                if key in user_model_paths:
                    user_file = user_model_paths[key]
                    if os.path.exists(user_file):
                        try:
                            if user_file.lower().endswith('.sdf'):
                                included_model = ET.parse(user_file).getroot()
                                break
                            elif user_file.lower().endswith(('.stl', '.dae', '.obj')):
                                # Create synthetic model
                                included_model = ET.Element('model')
                                included_model.set('name', model_name)
                                
                                link = ET.SubElement(included_model, 'link')
                                link.set('name', 'link')
                                
                                visual = ET.SubElement(link, 'visual')
                                visual.set('name', 'visual')
                                
                                geometry = ET.SubElement(visual, 'geometry')
                                mesh = ET.SubElement(geometry, 'mesh')
                                
                                uri_elem = ET.SubElement(mesh, 'uri')
                                uri_elem.text = user_file
                                
                                break
                        except Exception as e:
                            continue
        
        # Try resolver
        if included_model is None:
            resolved_path = model_resolver.resolve_model_uri(uri, status_callback)
            if resolved_path and os.path.exists(resolved_path):
                try:
                    if resolved_path.lower().endswith('.sdf'):
                        included_model = ET.parse(resolved_path).getroot()
                    elif os.path.isdir(resolved_path):
                        sdf_files = [f for f in os.listdir(resolved_path) if f.endswith('.sdf')]
                        if sdf_files:
                            sdf_path = os.path.join(resolved_path, sdf_files[0])
                            included_model = ET.parse(sdf_path).getroot()
                except:
                    pass
        
        if included_model is not None:
            wrapper = ET.Element('model')
            wrapper.set('name', model_name)
            pose_elem = ET.SubElement(wrapper, 'pose')
            pose_elem.text = ' '.join(map(str, model_pose))
            
            source = included_model if included_model.tag == 'model' else included_model.find('.//model')
            if source is not None:
                for child in source:
                    if child.tag != 'pose':
                        wrapper.append(child)
                        
            processed.append(wrapper)
        elif not skip_missing:
            model_id = uri.replace('model://', '')
            if model_id not in model_resolver.missing_models:
                model_resolver.missing_models.append(model_id)
    
    for model in processed:
        world_root.append(model)
    return world_root

def get_geometry_enhanced(world_root, model_resolver=None):
    """
    Extract geometry with 2D PROJECTION support for meshes
    """
    shapes = []
    
    for model in world_root.findall('.//model'):
        model_name = model.get('name', 'unnamed')
        model_pose = parse_pose(model.findtext('pose'))
        model_scale = parse_scale(model.findtext('scale'))
        model_transform = transform_matrix_from_pose(model_pose)
        
        scale_matrix = np.eye(4)
        np.fill_diagonal(scale_matrix[:3, :3], model_scale)
        model_transform = model_transform @ scale_matrix
        
        for link in model.findall('link'):
            link_name = link.get('name', 'unnamed')
            link_pose = parse_pose(link.findtext('pose'))
            link_transform = transform_matrix_from_pose(link_pose)
            combined_transform = model_transform @ link_transform
            
            geoms = [(g, g.find('geometry')) for g in link.findall('collision') + link.findall('visual') 
                     if g.find('geometry') is not None]
            
            for geom_elem, geometry in geoms:
                geom_pose = parse_pose(geom_elem.findtext('pose'))
                geom_transform = transform_matrix_from_pose(geom_pose)
                final_transform = combined_transform @ geom_transform
                
                position = final_transform[:3, 3]
                rotation_matrix = final_transform[:3, :3]
                
                yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
                
                if sy > 1e-6:
                    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    pitch = math.atan2(-rotation_matrix[2, 0], sy)
                else:
                    roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    pitch = math.atan2(-rotation_matrix[2, 0], sy)
                
                for geom_type in ['box', 'cylinder', 'sphere', 'mesh', 'plane', 'polyline']:
                    geom_elem = geometry.find(geom_type)
                    if geom_elem is None:
                        continue
                        
                    shape_info = {
                        'type': geom_type,
                        'position': position,
                        'rotation': (roll, pitch, yaw),
                        'model_name': model_name,
                        'link_name': link_name,
                        'transform': final_transform,
                        'model_scale': model_scale
                    }
                    
                    if geom_type == 'box':
                        size = [float(v) * s for v, s in zip(geom_elem.findtext('size', '1 1 1').split(), model_scale)]
                        shape_info['size'] = tuple(size)
                        
                    elif geom_type == 'cylinder':
                        radius = float(geom_elem.findtext('radius', '0.5')) * (model_scale[0] + model_scale[1]) / 2.0
                        length = float(geom_elem.findtext('length', '1.0')) * model_scale[2]
                        shape_info['size'] = (radius, length)
                        
                    elif geom_type == 'sphere':
                        radius = float(geom_elem.findtext('radius', '0.5')) * max(model_scale)
                        shape_info['size'] = (radius,)
                        
                    elif geom_type == 'plane':
                        size = [float(v) for v in geom_elem.findtext('size', '1 1').split()]
                        shape_info['size'] = (size[0] * model_scale[0], size[1] * model_scale[1], 0.01)
                        
                    elif geom_type == 'polyline':
                        points, height = parse_polyline_points(geom_elem)
                        if points:
                            xs, ys = zip(*points)
                            bounds_x, bounds_y = max(xs) - min(xs), max(ys) - min(ys)
                            shape_info['size'] = (bounds_x * model_scale[0], bounds_y * model_scale[1], height * model_scale[2])
                            shape_info['points'] = points
                            shape_info['height'] = height
                        else:
                            continue
                            
                    elif geom_type == 'mesh':
                        mesh_uri = geom_elem.findtext('uri', '')
                        scale_elem = geom_elem.find('scale')
                        mesh_scale = parse_scale(scale_elem.text if scale_elem is not None else None)
                        
                        # Resolve path
                        resolved_uri = mesh_uri
                        if model_resolver:
                            if mesh_uri.startswith('model://') or not os.path.isabs(mesh_uri):
                                resolved_uri = model_resolver.resolve_model_uri(mesh_uri)
                        
                        # Store for projection
                        shape_info['uri'] = mesh_uri
                        shape_info['resolved_uri'] = resolved_uri
                        shape_info['mesh_scale'] = mesh_scale
                        shape_info['use_projection'] = True  # Flag for 2D projection
                        
                        # Fallback bounds
                        if model_resolver:
                            bounds = model_resolver.estimate_mesh_bounds(resolved_uri or mesh_uri, mesh_scale)
                            final_bounds = tuple(b * ms for b, ms in zip(bounds, model_scale))
                            shape_info['size'] = final_bounds
                        else:
                            shape_info['size'] = (1.0, 1.0, 1.0)
                        
                        
                        # Track missing
                        if model_resolver and mesh_uri:
                            actual_path = resolved_uri or mesh_uri
                            if not (actual_path and os.path.exists(actual_path)):
                                mesh_name = os.path.splitext(os.path.basename(mesh_uri))[0]
                                if mesh_name and mesh_name not in model_resolver.missing_models:
                                    model_resolver.missing_models.append(mesh_name)
                    
                    shapes.append(shape_info)
                    break
                    
    return shapes

# =============== Main Parser ===============

class SDFParser:
    """Enhanced SDF Parser with 2D projection support"""
    
    def __init__(self):
        self.model_resolver = ModelResolver()
        self.parsed_data = {}
        
    def parse_file(self, file_path, user_model_paths=None, skip_missing=False, status_callback=None):
        """Parse SDF/World file"""
        try:
            if status_callback:
                status_callback(f"Loading: {os.path.basename(file_path)}")
                
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            if file_path.endswith('.sdf'):
                world = root.find('world') if root.tag == 'sdf' else (root if root.tag == 'world' else None)
                if world is None and root.tag in ['sdf', 'model']:
                    world = ET.Element('world')
                    world.set('name', 'temp_world')
                    model = root.find('model') if root.tag == 'sdf' else root
                    if model is not None:
                        world.append(model)
            else:
                world = root if root.tag == 'world' else root.find('world')
                
            if world is None:
                raise Exception("No world element found")
            
            if user_model_paths:
                self.model_resolver.set_user_model_paths(user_model_paths)
                if status_callback:
                    status_callback(f"Loaded {len(user_model_paths)} user paths")
            
            world = process_includes(world, self.model_resolver, user_model_paths, skip_missing, status_callback)
            shapes = get_geometry_enhanced(world, self.model_resolver)
            
            self.parsed_data = {
                'shapes': shapes,
                'world': world,
                'missing_models': self.model_resolver.missing_models.copy(),
                'file_path': file_path
            }
            
            if status_callback:
                shape_types = {}
                for shape in shapes:
                    stype = shape['type']
                    shape_types[stype] = shape_types.get(stype, 0) + 1
                
                status_callback(f"Parsed {len(shapes)} shapes")
                for stype, count in shape_types.items():
                    status_callback(f"  {stype}: {count}")
            
            return self.parsed_data
            
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            if status_callback:
                status_callback(error_msg)
            raise Exception(error_msg)
    
    def get_shapes(self):
        return self.parsed_data.get('shapes', [])
    
    def get_missing_models(self):
        return self.parsed_data.get('missing_models', [])
    
    def get_statistics(self):
        shapes = self.get_shapes()
        stats = {
            'total_shapes': len(shapes),
            'shape_types': {},
            'models_count': len(set(s['model_name'] for s in shapes)),
            'missing_models_count': len(self.get_missing_models())
        }
        
        for shape in shapes:
            shape_type = shape['type']
            stats['shape_types'][shape_type] = stats['shape_types'].get(shape_type, 0) + 1
            
        return stats