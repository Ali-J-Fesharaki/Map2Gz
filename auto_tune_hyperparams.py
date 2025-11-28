#!/usr/bin/env python3
"""
Automatic Hyperparameter Tuning for PGM to SDF Conversion

This script automatically finds optimal hyperparameters for pgm_to_sdf.py
by using a round-trip comparison:
    Original PGM → SDF → Reconstructed PGM → Compare with Original

The comparison metrics help identify which hyperparameters produce
the most accurate SDF representation of the original map.
"""

import os
import sys
import argparse
import tempfile
import shutil
import numpy as np
from PIL import Image
import yaml
from itertools import product
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import cv2

# Import the conversion modules
from pgm_to_sdf import simple_pgm_to_sdf, load_map_data
from sdf2map_cli import convert_sdf_to_map


class MapComparator:
    """Compare two PGM maps and compute similarity metrics"""
    
    def __init__(self, threshold: int = 128):
        self.threshold = threshold
    
    def load_pgm(self, pgm_path: str) -> np.ndarray:
        """Load PGM file as binary image"""
        img = Image.open(pgm_path).convert('L')
        return np.array(img)
    
    def load_yaml(self, yaml_path: str) -> dict:
        """Load YAML metadata"""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def to_binary(self, img: np.ndarray) -> np.ndarray:
        """Convert grayscale to binary (walls=1, free=0)"""
        return (img < self.threshold).astype(np.uint8)
    
    def align_maps(self, img1: np.ndarray, meta1: dict, img2: np.ndarray, meta2: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two maps based on their world coordinates (origin and resolution).
        Returns aligned binary maps of the same size.
        """
        bin1 = self.to_binary(img1)
        bin2 = self.to_binary(img2)
        
        res1 = meta1['resolution']
        res2 = meta2['resolution']
        origin1 = meta1['origin']
        origin2 = meta2['origin']
        
        # If same resolution and origin, just resize if needed
        if abs(res1 - res2) < 0.001 and \
           abs(origin1[0] - origin2[0]) < 0.01 and \
           abs(origin1[1] - origin2[1]) < 0.01:
            if bin1.shape != bin2.shape:
                bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            return bin1, bin2
        
        # Calculate world bounds for both maps
        h1, w1 = bin1.shape
        h2, w2 = bin2.shape
        
        # World coordinates of corners (note: Y-axis is flipped in PGM)
        world1_min_x = origin1[0]
        world1_max_x = origin1[0] + w1 * res1
        world1_min_y = origin1[1]
        world1_max_y = origin1[1] + h1 * res1
        
        world2_min_x = origin2[0]
        world2_max_x = origin2[0] + w2 * res2
        world2_min_y = origin2[1]
        world2_max_y = origin2[1] + h2 * res2
        
        # Find common world bounds
        common_min_x = max(world1_min_x, world2_min_x)
        common_max_x = min(world1_max_x, world2_max_x)
        common_min_y = max(world1_min_y, world2_min_y)
        common_max_y = min(world1_max_y, world2_max_y)
        
        if common_max_x <= common_min_x or common_max_y <= common_min_y:
            # No overlap - fallback to simple resize
            if bin1.shape != bin2.shape:
                bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            return bin1, bin2
        
        # Use resolution of first map for output
        out_res = res1
        out_w = int((common_max_x - common_min_x) / out_res)
        out_h = int((common_max_y - common_min_y) / out_res)
        
        if out_w < 10 or out_h < 10:
            # Too small - fallback
            if bin1.shape != bin2.shape:
                bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            return bin1, bin2
        
        # Extract aligned regions from both maps
        def extract_region(binary, origin, resolution, common_bounds):
            min_x, max_x, min_y, max_y = common_bounds
            h, w = binary.shape
            
            # Calculate pixel coordinates for the common region
            px_min_x = int((min_x - origin[0]) / resolution)
            px_max_x = int((max_x - origin[0]) / resolution)
            # Y is flipped in image coordinates
            px_min_y = h - int((max_y - origin[1]) / resolution)
            px_max_y = h - int((min_y - origin[1]) / resolution)
            
            # Clamp to valid range
            px_min_x = max(0, px_min_x)
            px_max_x = min(w, px_max_x)
            px_min_y = max(0, px_min_y)
            px_max_y = min(h, px_max_y)
            
            return binary[px_min_y:px_max_y, px_min_x:px_max_x]
        
        common_bounds = (common_min_x, common_max_x, common_min_y, common_max_y)
        aligned1 = extract_region(bin1, origin1, res1, common_bounds)
        aligned2 = extract_region(bin2, origin2, res2, common_bounds)
        
        # Resize aligned2 to match aligned1 if needed
        if aligned1.shape != aligned2.shape and aligned1.size > 0 and aligned2.size > 0:
            aligned2 = cv2.resize(aligned2, (aligned1.shape[1], aligned1.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        return aligned1, aligned2
    
    def compute_iou(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) for wall pixels
        Higher is better (1.0 = perfect match)
        """
        bin1 = self.to_binary(img1)
        bin2 = self.to_binary(img2)
        
        # Resize if needed
        if bin1.shape != bin2.shape:
            bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        intersection = np.sum(bin1 & bin2)
        union = np.sum(bin1 | bin2)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_dice(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Dice coefficient (F1 score for segmentation)
        Higher is better (1.0 = perfect match)
        """
        bin1 = self.to_binary(img1)
        bin2 = self.to_binary(img2)
        
        if bin1.shape != bin2.shape:
            bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        intersection = np.sum(bin1 & bin2)
        total = np.sum(bin1) + np.sum(bin2)
        
        if total == 0:
            return 1.0
        
        return 2 * intersection / total
    
    def compute_pixel_accuracy(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute pixel-wise accuracy
        Higher is better (1.0 = perfect match)
        """
        bin1 = self.to_binary(img1)
        bin2 = self.to_binary(img2)
        
        if bin1.shape != bin2.shape:
            bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        correct = np.sum(bin1 == bin2)
        total = bin1.size
        
        return correct / total
    
    def compute_wall_coverage(self, original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float]:
        """
        Compute how much of the original walls are captured (recall)
        and how accurate the reconstructed walls are (precision)
        """
        orig_walls = self.to_binary(original)
        recon_walls = self.to_binary(reconstructed)
        
        if orig_walls.shape != recon_walls.shape:
            recon_walls = cv2.resize(recon_walls, (orig_walls.shape[1], orig_walls.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        
        orig_wall_count = np.sum(orig_walls)
        recon_wall_count = np.sum(recon_walls)
        overlap = np.sum(orig_walls & recon_walls)
        
        # Recall: how much of original walls are in reconstructed
        recall = overlap / orig_wall_count if orig_wall_count > 0 else 1.0
        
        # Precision: how much of reconstructed walls are correct
        precision = overlap / recon_wall_count if recon_wall_count > 0 else 1.0
        
        return precision, recall
    
    def compute_hausdorff_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute approximate Hausdorff distance between wall contours
        Lower is better (0 = perfect match)
        Normalized by image diagonal
        """
        bin1 = self.to_binary(img1)
        bin2 = self.to_binary(img2)
        
        if bin1.shape != bin2.shape:
            bin2 = cv2.resize(bin2, (bin1.shape[1], bin1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        # Get wall pixel coordinates
        pts1 = np.column_stack(np.where(bin1 > 0))
        pts2 = np.column_stack(np.where(bin2 > 0))
        
        if len(pts1) == 0 or len(pts2) == 0:
            # One image has no walls
            diagonal = np.sqrt(bin1.shape[0]**2 + bin1.shape[1]**2)
            return diagonal if len(pts1) != len(pts2) else 0.0
        
        # Subsample for efficiency
        max_points = 1000
        if len(pts1) > max_points:
            indices = np.random.choice(len(pts1), max_points, replace=False)
            pts1 = pts1[indices]
        if len(pts2) > max_points:
            indices = np.random.choice(len(pts2), max_points, replace=False)
            pts2 = pts2[indices]
        
        # Compute directed Hausdorff distances
        def directed_hausdorff(a, b):
            """Distance from each point in a to nearest point in b"""
            dists = np.sqrt(((a[:, np.newaxis] - b[np.newaxis, :]) ** 2).sum(axis=2))
            return np.max(np.min(dists, axis=1))
        
        d1 = directed_hausdorff(pts1, pts2)
        d2 = directed_hausdorff(pts2, pts1)
        
        # Normalize by diagonal
        diagonal = np.sqrt(bin1.shape[0]**2 + bin1.shape[1]**2)
        
        return max(d1, d2) / diagonal
    
    # Aligned versions (take pre-aligned binary maps)
    def compute_iou_aligned(self, bin1: np.ndarray, bin2: np.ndarray) -> float:
        """Compute IoU on pre-aligned binary maps"""
        intersection = np.sum(bin1 & bin2)
        union = np.sum(bin1 | bin2)
        return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
    
    def compute_dice_aligned(self, bin1: np.ndarray, bin2: np.ndarray) -> float:
        """Compute Dice on pre-aligned binary maps"""
        intersection = np.sum(bin1 & bin2)
        total = np.sum(bin1) + np.sum(bin2)
        return 2 * intersection / total if total > 0 else 1.0
    
    def compute_pixel_accuracy_aligned(self, bin1: np.ndarray, bin2: np.ndarray) -> float:
        """Compute pixel accuracy on pre-aligned binary maps"""
        return np.sum(bin1 == bin2) / bin1.size
    
    def compute_wall_coverage_aligned(self, orig_walls: np.ndarray, recon_walls: np.ndarray) -> Tuple[float, float]:
        """Compute precision/recall on pre-aligned binary maps"""
        orig_count = np.sum(orig_walls)
        recon_count = np.sum(recon_walls)
        overlap = np.sum(orig_walls & recon_walls)
        
        recall = overlap / orig_count if orig_count > 0 else 1.0
        precision = overlap / recon_count if recon_count > 0 else 1.0
        return precision, recall
    
    def compute_hausdorff_distance_aligned(self, bin1: np.ndarray, bin2: np.ndarray) -> float:
        """Compute Hausdorff distance on pre-aligned binary maps"""
        pts1 = np.column_stack(np.where(bin1 > 0))
        pts2 = np.column_stack(np.where(bin2 > 0))
        
        if len(pts1) == 0 or len(pts2) == 0:
            diagonal = np.sqrt(bin1.shape[0]**2 + bin1.shape[1]**2)
            return diagonal if len(pts1) != len(pts2) else 0.0
        
        max_points = 1000
        if len(pts1) > max_points:
            pts1 = pts1[np.random.choice(len(pts1), max_points, replace=False)]
        if len(pts2) > max_points:
            pts2 = pts2[np.random.choice(len(pts2), max_points, replace=False)]
        
        def directed_hausdorff(a, b):
            dists = np.sqrt(((a[:, np.newaxis] - b[np.newaxis, :]) ** 2).sum(axis=2))
            return np.max(np.min(dists, axis=1))
        
        d1 = directed_hausdorff(pts1, pts2)
        d2 = directed_hausdorff(pts2, pts1)
        diagonal = np.sqrt(bin1.shape[0]**2 + bin1.shape[1]**2)
        return max(d1, d2) / diagonal
    
    def compute_all_metrics(self, original_pgm: str, reconstructed_pgm: str) -> Dict[str, float]:
        """Compute all comparison metrics with proper world-coordinate alignment"""
        orig = self.load_pgm(original_pgm)
        recon = self.load_pgm(reconstructed_pgm)
        
        # Try to load YAML metadata for alignment
        orig_yaml_path = original_pgm.replace('.pgm', '.yaml')
        recon_yaml_path = reconstructed_pgm.replace('.pgm', '.yaml')
        
        try:
            orig_meta = self.load_yaml(orig_yaml_path)
            recon_meta = self.load_yaml(recon_yaml_path)
            
            # Align maps based on world coordinates
            aligned_orig, aligned_recon = self.align_maps(orig, orig_meta, recon, recon_meta)
        except:
            # Fallback to simple resize if YAML not available
            aligned_orig = self.to_binary(orig)
            aligned_recon = self.to_binary(recon)
            if aligned_orig.shape != aligned_recon.shape:
                aligned_recon = cv2.resize(aligned_recon, (aligned_orig.shape[1], aligned_orig.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        
        # Use aligned maps for all metrics
        precision, recall = self.compute_wall_coverage_aligned(aligned_orig, aligned_recon)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'iou': self.compute_iou_aligned(aligned_orig, aligned_recon),
            'dice': self.compute_dice_aligned(aligned_orig, aligned_recon),
            'pixel_accuracy': self.compute_pixel_accuracy_aligned(aligned_orig, aligned_recon),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hausdorff_normalized': self.compute_hausdorff_distance_aligned(aligned_orig, aligned_recon)
        }
        
        # Compute combined score (weighted average)
        # Prioritize IoU and F1, penalize Hausdorff distance
        metrics['combined_score'] = (
            0.3 * metrics['iou'] +
            0.3 * metrics['f1'] +
            0.2 * metrics['dice'] +
            0.2 * (1.0 - min(metrics['hausdorff_normalized'], 1.0))
        )
        
        return metrics


class HyperparameterTuner:
    """Automatic hyperparameter tuning for PGM to SDF conversion"""
    
    def __init__(self, original_pgm: str, original_yaml: str, 
                 output_dir: str = None, verbose: bool = True):
        self.original_pgm = original_pgm
        self.original_yaml = original_yaml
        self.verbose = verbose
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(original_pgm), 'tuning_results')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.comparator = MapComparator()
        self.results = []
        
        # Load original map info
        self.original_map, self.original_metadata = load_map_data(original_pgm, original_yaml)
        self.resolution = self.original_metadata['resolution']
        
    def log(self, msg: str):
        """Print message if verbose"""
        if self.verbose:
            print(msg)
    
    def get_default_param_grid(self, method: str = 'hough') -> Dict[str, List]:
        """
        Get default hyperparameter grid for tuning based on detection method
        
        Args:
            method: Detection method - 'hough', 'contour', 'pixel', or 'rle'
        """
        if method == 'hough':
            return {
                'method': ['hough'],
                'wall_height': [2.0],
                'wall_thickness': [0.05, 0.1, 0.15, 0.2],
                'threshold': [100, 128, 150, 180],
                'sensitivity': [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                'min_area': [10],
                'approx_epsilon': [2.0],
            }
        elif method == 'contour':
            return {
                'method': ['contour'],
                'wall_height': [2.0],
                'wall_thickness': [0.1],  # Not used but needed for API
                'threshold': [100, 128, 150, 180],
                'sensitivity': [1.0],  # Not used but needed for API
                'min_area': [5, 10, 20, 50],
                'approx_epsilon': [1.0, 2.0, 3.0, 5.0],
            }
        elif method == 'pixel':
            return {
                'method': ['pixel'],
                'wall_height': [2.0],
                'wall_thickness': [0.1],  # Not used but needed for API
                'threshold': [100, 128, 150, 180],
                'sensitivity': [1.0],  # Not used but needed for API
                'min_area': [1, 5, 10, 20],
                'approx_epsilon': [2.0],  # Not used but needed for API
            }
        elif method == 'rle':
            # Run-length encoding - best for orthogonal mazes
            return {
                'method': ['rle'],
                'wall_height': [2.0],
                'wall_thickness': [0.1],  # Not used but needed for API
                'threshold': [100, 128, 150, 180],
                'sensitivity': [1.0],  # Not used but needed for API
                'min_area': [10],  # Not used but needed for API
                'approx_epsilon': [2.0],  # Not used but needed for API
            }
        elif method == 'all':
            # Test all methods
            return {
                'method': ['hough', 'contour', 'pixel', 'rle'],
                'wall_height': [2.0],
                'wall_thickness': [0.1, 0.15],
                'threshold': [128, 150],
                'sensitivity': [1.0, 1.5],
                'min_area': [10, 20],
                'approx_epsilon': [2.0],
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_fine_param_grid(self, best_params: Dict) -> Dict[str, List]:
        """
        Get fine-tuning parameter grid around best parameters
        """
        def expand_around(value, step, count=3):
            """Create values around a center value"""
            return [value + step * i for i in range(-count, count + 1)]
        
        method = best_params.get('method', 'hough')
        
        base_grid = {
            'method': [method],
            'wall_height': [best_params.get('wall_height', 2.0)],
            'threshold': [int(t) for t in expand_around(best_params.get('threshold', 128), 10, 2)],
        }
        
        if method == 'hough':
            base_grid['wall_thickness'] = expand_around(best_params.get('wall_thickness', 0.1), 0.02, 2)
            base_grid['sensitivity'] = expand_around(best_params.get('sensitivity', 1.0), 0.25, 2)
            base_grid['min_area'] = [10]
            base_grid['approx_epsilon'] = [2.0]
        elif method == 'contour':
            base_grid['wall_thickness'] = [0.1]
            base_grid['sensitivity'] = [1.0]
            base_grid['min_area'] = [int(a) for a in expand_around(best_params.get('min_area', 10), 5, 2)]
            base_grid['approx_epsilon'] = expand_around(best_params.get('approx_epsilon', 2.0), 0.5, 2)
        elif method == 'pixel':
            base_grid['wall_thickness'] = [0.1]
            base_grid['sensitivity'] = [1.0]
            base_grid['min_area'] = [int(a) for a in expand_around(best_params.get('min_area', 10), 3, 2)]
            base_grid['approx_epsilon'] = [2.0]
        elif method == 'rle':
            # RLE has minimal parameters - just threshold matters
            base_grid['wall_thickness'] = [0.1]
            base_grid['sensitivity'] = [1.0]
            base_grid['min_area'] = [10]
            base_grid['approx_epsilon'] = [2.0]
        
        return base_grid
    
    def run_conversion_pipeline(self, params: Dict, temp_dir: str) -> Optional[str]:
        """
        Run the full conversion pipeline:
        Original PGM → SDF → Reconstructed PGM
        
        Returns path to reconstructed PGM or None if failed
        """
        try:
            # Step 1: Convert PGM to SDF
            sdf_path = os.path.join(temp_dir, 'temp_world.sdf')
            
            simple_pgm_to_sdf(
                pgm_file=self.original_pgm,
                yaml_file=self.original_yaml,
                output_sdf=sdf_path,
                wall_height=params['wall_height'],
                wall_thickness=params['wall_thickness'],
                threshold=params['threshold'],
                sensitivity=params['sensitivity'],
                method=params.get('method', 'hough'),
                min_area=params.get('min_area', 10),
                approx_epsilon=params.get('approx_epsilon', 2.0)
            )
            
            if not os.path.exists(sdf_path):
                return None
            
            # Step 2: Convert SDF back to PGM
            pgm_path, yaml_path = convert_sdf_to_map(
                input_file=sdf_path,
                output_dir=temp_dir,
                resolution=self.resolution,
                lidar_height=20,
                margin=1.0,
                min_wall_thickness=params.get('wall_thickness', 0.1),
                verbose=False
            )
            
            return pgm_path
            
        except Exception as e:
            if self.verbose:
                print(f"  Conversion failed: {e}")
            return None
    
    def evaluate_params(self, params: Dict) -> Optional[Dict]:
        """
        Evaluate a single parameter combination
        Returns metrics dict or None if failed
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            reconstructed_pgm = self.run_conversion_pipeline(params, temp_dir)
            
            if reconstructed_pgm is None:
                return None
            
            try:
                metrics = self.comparator.compute_all_metrics(
                    self.original_pgm, reconstructed_pgm
                )
                return metrics
            except Exception as e:
                if self.verbose:
                    print(f"  Evaluation failed: {e}")
                return None
    
    def tune(self, param_grid: Dict[str, List] = None, 
             max_iterations: int = None) -> Dict:
        """
        Run hyperparameter tuning
        
        Args:
            param_grid: Parameter grid to search (uses default if None)
            max_iterations: Maximum number of parameter combinations to try
            
        Returns:
            Dict with best parameters and all results
        """
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        if max_iterations:
            all_combinations = all_combinations[:max_iterations]
        
        total = len(all_combinations)
        self.log(f"\nTuning {total} parameter combinations...")
        self.log(f"Parameters: {param_names}")
        
        best_score = -1
        best_params = None
        best_metrics = None
        
        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            
            self.log(f"\n[{i+1}/{total}] Testing: {params}")
            
            metrics = self.evaluate_params(params)
            
            if metrics is None:
                self.log("  → Failed")
                continue
            
            score = metrics['combined_score']
            self.log(f"  → Score: {score:.4f} (IoU: {metrics['iou']:.4f}, "
                    f"F1: {metrics['f1']:.4f})")
            
            result = {
                'params': params,
                'metrics': metrics,
                'score': score
            }
            self.results.append(result)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_metrics = metrics.copy()
                self.log(f"  ★ New best!")
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_score': best_score,
            'all_results': self.results
        }
    
    def fine_tune(self, coarse_results: Dict, iterations: int = 1) -> Dict:
        """
        Fine-tune around the best parameters from coarse search
        """
        best_params = coarse_results['best_params']
        
        for i in range(iterations):
            self.log(f"\n=== Fine-tuning iteration {i+1} ===")
            
            fine_grid = self.get_fine_param_grid(best_params)
            results = self.tune(fine_grid)
            
            if results['best_score'] > coarse_results['best_score']:
                coarse_results = results
                best_params = results['best_params']
            else:
                self.log("No improvement, stopping fine-tuning")
                break
        
        return coarse_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save tuning results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tuning_results_{timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        self.log(f"\nResults saved to: {filepath}")
        return filepath
    
    def generate_best_sdf(self, best_params: Dict, output_name: str = None):
        """Generate SDF file with best parameters"""
        if output_name is None:
            base_name = os.path.splitext(os.path.basename(self.original_pgm))[0]
            method = best_params.get('method', 'hough')
            output_name = f'{base_name}_optimized_{method}.sdf'
        
        output_path = os.path.join(self.output_dir, output_name)
        
        simple_pgm_to_sdf(
            pgm_file=self.original_pgm,
            yaml_file=self.original_yaml,
            output_sdf=output_path,
            wall_height=best_params['wall_height'],
            wall_thickness=best_params.get('wall_thickness', 0.1),
            threshold=best_params['threshold'],
            sensitivity=best_params.get('sensitivity', 1.0),
            method=best_params.get('method', 'hough'),
            min_area=best_params.get('min_area', 10),
            approx_epsilon=best_params.get('approx_epsilon', 2.0)
        )
        
        self.log(f"Optimized SDF saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Automatic Hyperparameter Tuning for PGM to SDF Conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s map.pgm map.yaml
  %(prog)s map.pgm map.yaml --method pixel  # Best for clean/exact maps
  %(prog)s map.pgm map.yaml --method contour  # Good for thick-walled maps
  %(prog)s map.pgm map.yaml --method all  # Test all methods
  %(prog)s map.pgm map.yaml -o ./results --fine-tune
  %(prog)s map.pgm map.yaml --quick
        """
    )
    
    # Required arguments
    parser.add_argument('pgm_file', help='Original PGM map file')
    parser.add_argument('yaml_file', help='Original YAML metadata file')
    
    # Output options
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory for results')
    
    # Detection method
    parser.add_argument('--method', choices=['hough', 'contour', 'pixel', 'rle', 'all'], default='all',
                       help='Detection method to tune: hough (SLAM maps), contour (clean maps), pixel (simple maps), rle (orthogonal mazes), all (test all)')
    
    # Tuning options
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: test fewer parameter combinations')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Perform additional fine-tuning around best parameters')
    parser.add_argument('--max-iter', type=int, default=None,
                       help='Maximum number of iterations')
    
    # Parameter ranges (for hough method)
    parser.add_argument('--sensitivity-range', type=float, nargs=2, 
                       metavar=('MIN', 'MAX'), default=[0.5, 3.0],
                       help='Sensitivity range for hough method (default: 0.5 3.0)')
    parser.add_argument('--thickness-range', type=float, nargs=2,
                       metavar=('MIN', 'MAX'), default=[0.05, 0.2],
                       help='Wall thickness range for hough method (default: 0.05 0.2)')
    parser.add_argument('--threshold-range', type=int, nargs=2,
                       metavar=('MIN', 'MAX'), default=[100, 180],
                       help='Threshold range (default: 100 180)')
    
    # Output control
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                       help='Verbose output (default: True)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress output')
    parser.add_argument('--generate-sdf', action='store_true',
                       help='Generate optimized SDF with best parameters')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pgm_file):
        print(f"Error: PGM file not found: {args.pgm_file}")
        sys.exit(1)
    if not os.path.exists(args.yaml_file):
        print(f"Error: YAML file not found: {args.yaml_file}")
        sys.exit(1)
    
    verbose = args.verbose and not args.quiet
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        original_pgm=args.pgm_file,
        original_yaml=args.yaml_file,
        output_dir=args.output,
        verbose=verbose
    )
    
    # Build parameter grid based on method
    if args.quick:
        # Quick mode - test only a few combinations
        if args.method == 'all':
            param_grid = {
                'method': ['hough', 'contour', 'pixel'],
                'wall_height': [2.0],
                'wall_thickness': [0.1],
                'threshold': [128],
                'sensitivity': [1.0],
                'min_area': [10],
                'approx_epsilon': [2.0],
            }
        else:
            param_grid = tuner.get_default_param_grid(args.method)
            # Reduce grid for quick mode
            for key in param_grid:
                if len(param_grid[key]) > 3 and key != 'method':
                    param_grid[key] = param_grid[key][::2]  # Take every other value
    else:
        param_grid = tuner.get_default_param_grid(args.method)
    
    if verbose:
        print("=" * 60)
        print("Automatic Hyperparameter Tuning for PGM to SDF Conversion")
        print("=" * 60)
        print(f"Original PGM: {args.pgm_file}")
        print(f"Original YAML: {args.yaml_file}")
        print(f"Output directory: {tuner.output_dir}")
        print(f"Resolution: {tuner.resolution} m/pixel")
        print(f"Original map size: {tuner.original_map.shape}")
        print(f"Detection method(s): {args.method}")
    
    # Run coarse tuning
    results = tuner.tune(param_grid, max_iterations=args.max_iter)
    
    # Fine-tune if requested
    if args.fine_tune and results['best_params'] is not None:
        results = tuner.fine_tune(results)
    
    # Save results
    tuner.save_results(results)
    
    # Generate optimized SDF if requested
    if args.generate_sdf and results['best_params'] is not None:
        tuner.generate_best_sdf(results['best_params'])
    
    # Print summary
    if verbose and results['best_params'] is not None:
        print("\n" + "=" * 60)
        print("TUNING COMPLETE")
        print("=" * 60)
        print(f"\nBest Parameters:")
        for k, v in results['best_params'].items():
            print(f"  {k}: {v}")
        print(f"\nBest Metrics:")
        for k, v in results['best_metrics'].items():
            print(f"  {k}: {v:.4f}")
        print(f"\nBest Combined Score: {results['best_score']:.4f}")
        
        # Print command to use these parameters
        bp = results['best_params']
        method = bp.get('method', 'hough')
        print(f"\nTo convert with optimal parameters, run:")
        
        if method == 'hough':
            print(f"  python pgm_to_sdf.py {args.pgm_file} {args.yaml_file} "
                  f"--method {method} "
                  f"--thickness {bp.get('wall_thickness', 0.1):.3f} "
                  f"--threshold {bp['threshold']} "
                  f"--sensitivity {bp.get('sensitivity', 1.0):.2f}")
        elif method == 'contour':
            print(f"  python pgm_to_sdf.py {args.pgm_file} {args.yaml_file} "
                  f"--method {method} "
                  f"--threshold {bp['threshold']} "
                  f"--min-area {bp.get('min_area', 10)} "
                  f"--approx-epsilon {bp.get('approx_epsilon', 2.0):.1f}")
        elif method == 'pixel':
            print(f"  python pgm_to_sdf.py {args.pgm_file} {args.yaml_file} "
                  f"--method {method} "
                  f"--threshold {bp['threshold']} "
                  f"--min-area {bp.get('min_area', 10)}")
        elif method == 'rle':
            print(f"  python pgm_to_sdf.py {args.pgm_file} {args.yaml_file} "
                  f"--method {method} "
                  f"--threshold {bp['threshold']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
