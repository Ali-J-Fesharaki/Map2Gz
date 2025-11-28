#!/usr/bin/env python3
"""
Visualization utility for comparing original and reconstructed PGM maps.
Creates side-by-side comparison images and difference maps.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def load_pgm(pgm_path: str) -> np.ndarray:
    """Load PGM file as numpy array"""
    img = Image.open(pgm_path).convert('L')
    return np.array(img)


def to_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert grayscale to binary (walls=1, free=0)"""
    return (img < threshold).astype(np.uint8)


def create_difference_visualization(original: np.ndarray, 
                                   reconstructed: np.ndarray,
                                   threshold: int = 128) -> np.ndarray:
    """
    Create a color-coded difference visualization
    
    Colors:
    - White: Free space in both
    - Black/Gray: Walls in both (true positive)
    - Green: Walls only in original (false negative - missing walls)
    - Red: Walls only in reconstructed (false positive - extra walls)
    """
    orig_binary = to_binary(original, threshold)
    recon_binary = to_binary(reconstructed, threshold)
    
    # Resize if needed
    if orig_binary.shape != recon_binary.shape:
        recon_binary = cv2.resize(recon_binary, 
                                  (orig_binary.shape[1], orig_binary.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    
    # Create RGB image
    h, w = orig_binary.shape
    diff_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
    
    # True positives (walls in both) - Dark gray
    true_pos = (orig_binary == 1) & (recon_binary == 1)
    diff_img[true_pos] = [80, 80, 80]
    
    # False negatives (walls only in original) - Green
    false_neg = (orig_binary == 1) & (recon_binary == 0)
    diff_img[false_neg] = [0, 200, 0]
    
    # False positives (walls only in reconstructed) - Red
    false_pos = (orig_binary == 0) & (recon_binary == 1)
    diff_img[false_pos] = [200, 0, 0]
    
    return diff_img


def create_overlay_visualization(original: np.ndarray,
                                 reconstructed: np.ndarray,
                                 threshold: int = 128,
                                 alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay visualization with original and reconstructed maps
    
    Original walls shown in blue, reconstructed in red, overlap in purple
    """
    orig_binary = to_binary(original, threshold)
    recon_binary = to_binary(reconstructed, threshold)
    
    if orig_binary.shape != recon_binary.shape:
        recon_binary = cv2.resize(recon_binary,
                                  (orig_binary.shape[1], orig_binary.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    
    h, w = orig_binary.shape
    overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Original walls in blue channel
    overlay[:, :, 0] = np.where(orig_binary == 1, 100, 255)  # R
    overlay[:, :, 2] = np.where(orig_binary == 1, 255, 255)  # B
    
    # Reconstructed walls in red channel (blend with existing)
    overlay[:, :, 0] = np.where(recon_binary == 1, 
                                np.minimum(overlay[:, :, 0], 255).astype(np.uint8) * int(alpha * 255) // 255 + 
                                int((1-alpha) * 255),
                                overlay[:, :, 0])
    overlay[:, :, 1] = np.where(recon_binary == 1, 100, overlay[:, :, 1])
    
    # Overlap in purple
    overlap = (orig_binary == 1) & (recon_binary == 1)
    overlay[overlap] = [128, 0, 128]
    
    return overlay


def create_comparison_figure(original_path: str,
                            reconstructed_path: str,
                            output_path: Optional[str] = None,
                            threshold: int = 128,
                            show: bool = True) -> None:
    """
    Create a comprehensive comparison figure with multiple views
    """
    original = load_pgm(original_path)
    reconstructed = load_pgm(reconstructed_path)
    
    # Resize reconstructed if needed
    if original.shape != reconstructed.shape:
        reconstructed = cv2.resize(reconstructed,
                                   (original.shape[1], original.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    
    # Create visualizations
    diff_vis = create_difference_visualization(original, reconstructed, threshold)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original map
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original PGM', fontsize=12)
    axes[0, 0].axis('off')
    
    # Reconstructed map
    axes[0, 1].imshow(reconstructed, cmap='gray')
    axes[0, 1].set_title('Reconstructed PGM (from SDF)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Difference visualization
    axes[1, 0].imshow(diff_vis)
    axes[1, 0].set_title('Difference Map\n(Gray=Both, Green=Missing, Red=Extra)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Compute metrics for text display
    orig_binary = to_binary(original, threshold)
    recon_binary = to_binary(reconstructed, threshold)
    
    intersection = np.sum(orig_binary & recon_binary)
    union = np.sum(orig_binary | recon_binary)
    iou = intersection / union if union > 0 else 0
    
    orig_walls = np.sum(orig_binary)
    recon_walls = np.sum(recon_binary)
    
    precision = intersection / recon_walls if recon_walls > 0 else 0
    recall = intersection / orig_walls if orig_walls > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Metrics text
    metrics_text = f"""
    Comparison Metrics:
    ─────────────────────
    IoU (Jaccard): {iou:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    F1 Score: {f1:.4f}
    
    Wall Pixel Counts:
    ─────────────────────
    Original: {orig_walls:,}
    Reconstructed: {recon_walls:,}
    Overlap: {intersection:,}
    
    Legend:
    ─────────────────────
    Gray: Walls in both maps
    Green: Missing walls (in original only)
    Red: Extra walls (in reconstructed only)
    """
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Metrics', fontsize=12)
    
    plt.suptitle(f'PGM Comparison: {os.path.basename(original_path)} vs Reconstructed',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_parameter_sweep_visualization(results_json: str,
                                        output_path: Optional[str] = None,
                                        show: bool = True) -> None:
    """
    Visualize hyperparameter tuning results from JSON file
    """
    import json
    
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    all_results = results.get('all_results', [])
    if not all_results:
        print("No results found in JSON file")
        return
    
    # Extract data for plotting
    sensitivities = []
    thicknesses = []
    thresholds = []
    scores = []
    ious = []
    
    for r in all_results:
        params = r['params']
        metrics = r['metrics']
        
        sensitivities.append(params.get('sensitivity', 1.0))
        thicknesses.append(params.get('wall_thickness', 0.1))
        thresholds.append(params.get('threshold', 128))
        scores.append(r.get('score', 0))
        ious.append(metrics.get('iou', 0))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score vs Sensitivity
    scatter1 = axes[0, 0].scatter(sensitivities, scores, c=thresholds, 
                                   cmap='viridis', s=50, alpha=0.7)
    axes[0, 0].set_xlabel('Sensitivity')
    axes[0, 0].set_ylabel('Combined Score')
    axes[0, 0].set_title('Score vs Sensitivity (color=threshold)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Threshold')
    
    # Score vs Thickness
    scatter2 = axes[0, 1].scatter(thicknesses, scores, c=sensitivities,
                                   cmap='plasma', s=50, alpha=0.7)
    axes[0, 1].set_xlabel('Wall Thickness')
    axes[0, 1].set_ylabel('Combined Score')
    axes[0, 1].set_title('Score vs Wall Thickness (color=sensitivity)')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Sensitivity')
    
    # IoU vs Sensitivity
    scatter3 = axes[1, 0].scatter(sensitivities, ious, c=thicknesses,
                                   cmap='coolwarm', s=50, alpha=0.7)
    axes[1, 0].set_xlabel('Sensitivity')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('IoU vs Sensitivity (color=thickness)')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Thickness')
    
    # Histogram of scores
    axes[1, 1].hist(scores, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(max(scores), color='red', linestyle='--', 
                       label=f'Best: {max(scores):.4f}')
    axes[1, 1].set_xlabel('Combined Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    
    # Add best parameters info
    best_params = results.get('best_params', {})
    best_score = results.get('best_score', 0)
    
    fig.suptitle(f'Hyperparameter Tuning Results\n'
                 f'Best: sensitivity={best_params.get("sensitivity", "N/A")}, '
                 f'thickness={best_params.get("wall_thickness", "N/A")}, '
                 f'threshold={best_params.get("threshold", "N/A")} '
                 f'(score={best_score:.4f})',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Parameter sweep visualization saved to: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize comparison between original and reconstructed PGM maps',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two PGM files')
    compare_parser.add_argument('original', help='Original PGM file')
    compare_parser.add_argument('reconstructed', help='Reconstructed PGM file')
    compare_parser.add_argument('-o', '--output', help='Output image path')
    compare_parser.add_argument('--threshold', type=int, default=128,
                               help='Wall detection threshold (default: 128)')
    compare_parser.add_argument('--no-show', action='store_true',
                               help='Do not display the figure')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='Visualize tuning results')
    results_parser.add_argument('json_file', help='Tuning results JSON file')
    results_parser.add_argument('-o', '--output', help='Output image path')
    results_parser.add_argument('--no-show', action='store_true',
                               help='Do not display the figure')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        if not os.path.exists(args.original):
            print(f"Error: Original file not found: {args.original}")
            sys.exit(1)
        if not os.path.exists(args.reconstructed):
            print(f"Error: Reconstructed file not found: {args.reconstructed}")
            sys.exit(1)
            
        create_comparison_figure(
            args.original,
            args.reconstructed,
            args.output,
            args.threshold,
            show=not args.no_show
        )
        
    elif args.command == 'results':
        if not os.path.exists(args.json_file):
            print(f"Error: JSON file not found: {args.json_file}")
            sys.exit(1)
            
        create_parameter_sweep_visualization(
            args.json_file,
            args.output,
            show=not args.no_show
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
