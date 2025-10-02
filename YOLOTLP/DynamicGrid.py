"""
Dynamic Sampling Grid Analyzer for YOLOv12
Automatically determines optimal detection scales based on dataset object sizes
"""

import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json


class DynamicGridAnalyzer:
    """Analyzes dataset and determines optimal sampling grids for YOLOv12"""
    
    def __init__(self, data_yaml: str, img_size: int = 640):
        """
        Args:
            data_yaml: Path to dataset YAML file
            img_size: Training image size
        """
        self.data_yaml = data_yaml
        self.img_size = img_size
        self.bbox_sizes = []
        self.bbox_areas = []
        
    def load_dataset(self, split: str = 'train') -> List[str]:
        """Load dataset paths from YAML"""
        with open(self.data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"📂 Loading {split} dataset...")
        
        # Get train/val path from YAML
        dataset_path = Path(data[split])
        print(f"   Image path: {dataset_path}")
        
        # Convert images path to labels path
        # VisDrone/VisDrone2019-DET-train/images -> VisDrone/VisDrone2019-DET-train/labels
        label_path = Path(str(dataset_path).replace('/images', '/labels'))
        
        # Also handle case without /images in path
        if not label_path.exists():
            label_path = dataset_path.parent / 'labels'
        
        print(f"   Label path: {label_path}")
        
        if not label_path.exists():
            print(f"❌ ERROR: Label directory not found: {label_path}")
            print(f"   Please check your dataset structure:")
            print(f"   Expected: VisDrone/VisDrone2019-DET-train/labels/")
            return []
        
        label_files = list(label_path.glob('*.txt'))
        print(f"   Found {len(label_files)} label files")
        
        if len(label_files) == 0:
            print(f"⚠️  WARNING: No .txt files found in {label_path}")
            print(f"   Make sure labels are in YOLO format (.txt files)")
        
        return label_files
    
    def analyze_bbox_sizes(self, label_files: List[str]) -> Dict:
        """Analyze bounding box sizes in the dataset"""
        bbox_widths = []
        bbox_heights = []
        bbox_areas = []
        
        print(f"🔍 Analyzing bounding boxes...")
        
        for i, label_file in enumerate(label_files):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(label_files)} files...")
            
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class x_center y_center width height (normalized)
                            w, h = float(parts[3]), float(parts[4])
                            
                            # Convert to pixel dimensions
                            w_px = w * self.img_size
                            h_px = h * self.img_size
                            area = w_px * h_px
                            
                            bbox_widths.append(w_px)
                            bbox_heights.append(h_px)
                            bbox_areas.append(area)
            except Exception as e:
                print(f"⚠️  Error reading {label_file}: {e}")
                continue
        
        self.bbox_sizes = list(zip(bbox_widths, bbox_heights))
        self.bbox_areas = bbox_areas
        
        print(f"   Total objects found: {len(bbox_areas)}")
        
        if len(bbox_areas) == 0:
            print("❌ ERROR: No bounding boxes found!")
            print("   Please check:")
            print("   1. Label files exist in the labels directory")
            print("   2. Labels are in YOLO format: class x_center y_center width height")
            print("   3. Files are not empty")
            return {
                'total_objects': 0,
                'width_stats': {},
                'height_stats': {},
                'area_stats': {},
                'percentiles': {}
            }
        
        # Calculate statistics
        stats = {
            'total_objects': len(bbox_widths),
            'width_stats': self._get_stats(bbox_widths),
            'height_stats': self._get_stats(bbox_heights),
            'area_stats': self._get_stats(bbox_areas),
            'percentiles': self._calculate_percentiles(bbox_areas)
        }
        
        return stats
    
    def _get_stats(self, data: List[float]) -> Dict:
        """Calculate statistics for data"""
        if not data:
            return {}
        
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data))
        }
    
    def _calculate_percentiles(self, areas: List[float]) -> Dict:
        """Calculate area percentiles"""
        if not areas:
            return {}
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        return {
            f'p{p}': float(np.percentile(areas, p)) for p in percentiles
        }
    
    def determine_optimal_strides(self, 
                                   min_stride: int = 4,
                                   max_stride: int = 64,
                                   num_scales: int = 3) -> List[int]:
        """
        Determine optimal detection strides based on object size distribution
        
        Strategy:
        - Small objects (< 32x32): Need fine grid (stride 4-8)
        - Medium objects (32x32 to 96x96): Need medium grid (stride 16)
        - Large objects (> 96x96): Need coarse grid (stride 32-64)
        """
        if not self.bbox_areas:
            print("⚠️  Warning: No bbox data analyzed. Using default strides.")
            return [8, 16, 32]
        
        areas = np.array(self.bbox_areas)
        
        # Define object size categories based on area
        small_threshold = 32 * 32  # 1024 pixels
        medium_threshold = 96 * 96  # 9216 pixels
        
        # Count objects in each category
        small_pct = np.sum(areas < small_threshold) / len(areas) * 100
        medium_pct = np.sum((areas >= small_threshold) & (areas < medium_threshold)) / len(areas) * 100
        large_pct = np.sum(areas >= medium_threshold) / len(areas) * 100
        
        print(f"\n📊 Object Size Distribution:")
        print(f"   Small objects (<32x32):  {small_pct:.1f}%")
        print(f"   Medium objects (32-96):  {medium_pct:.1f}%")
        print(f"   Large objects (>96x96):  {large_pct:.1f}%")
        
        # Determine optimal strides based on distribution
        strides = []
        
        # Strategy 1: High percentage of small objects (>50%)
        if small_pct > 50:
            print("\n🎯 Dataset has many SMALL objects - Using fine-grained grids")
            if num_scales == 2:
                strides = [4, 8]
            elif num_scales == 3:
                strides = [4, 8, 16]
            else:
                strides = [4, 8, 16, 32]
        
        # Strategy 2: Balanced distribution
        elif small_pct > 20 and medium_pct > 30:
            print("\n🎯 Dataset has BALANCED object sizes - Using standard grids")
            if num_scales == 2:
                strides = [8, 16]
            elif num_scales == 3:
                strides = [8, 16, 32]
            else:
                strides = [8, 16, 32, 64]
        
        # Strategy 3: Mostly large objects
        elif large_pct > 60:
            print("\n🎯 Dataset has many LARGE objects - Using coarse grids")
            if num_scales == 2:
                strides = [16, 32]
            elif num_scales == 3:
                strides = [16, 32, 64]
            else:
                strides = [8, 16, 32, 64]
        
        # Strategy 4: Mixed with emphasis on medium
        else:
            print("\n🎯 Dataset has MIXED sizes - Using adaptive grids")
            if num_scales == 2:
                strides = [8, 16]
            elif num_scales == 3:
                strides = [8, 16, 32]
            else:
                strides = [4, 8, 16, 32]
        
        return strides
    
    def calculate_grid_sizes(self, strides: List[int]) -> List[int]:
        """Calculate grid sizes from strides"""
        return [self.img_size // s for s in strides]
    
    def generate_dynamic_config(self, 
                                base_config: str,
                                output_config: str,
                                num_scales: int = 3):
        """
        Generate YOLOv12 config with dynamic grid sizes
        
        Args:
            base_config: Path to base YOLOv12 config
            output_config: Path to save modified config
            num_scales: Number of detection scales (2, 3, or 4)
        """
        # Analyze dataset
        print("\n" + "="*70)
        print("🔍 ANALYZING DATASET")
        print("="*70)
        
        label_files = self.load_dataset()
        
        if len(label_files) == 0:
            print("\n❌ Cannot proceed: No label files found!")
            return None, None, None
        
        stats = self.analyze_bbox_sizes(label_files)
        
        if stats['total_objects'] == 0:
            print("\n❌ Cannot proceed: No objects found in labels!")
            return None, None, None
        
        print("\n" + "="*70)
        print("📈 DATASET STATISTICS")
        print("="*70)
        print(f"Total objects: {stats['total_objects']:,}")
        print(f"\nWidth stats:")
        print(f"   Min:    {stats['width_stats']['min']:.2f} px")
        print(f"   Max:    {stats['width_stats']['max']:.2f} px")
        print(f"   Mean:   {stats['width_stats']['mean']:.2f} px")
        print(f"   Median: {stats['width_stats']['median']:.2f} px")
        print(f"\nHeight stats:")
        print(f"   Min:    {stats['height_stats']['min']:.2f} px")
        print(f"   Max:    {stats['height_stats']['max']:.2f} px")
        print(f"   Mean:   {stats['height_stats']['mean']:.2f} px")
        print(f"   Median: {stats['height_stats']['median']:.2f} px")
        print(f"\nArea stats:")
        print(f"   Min:    {stats['area_stats']['min']:.2f} px²")
        print(f"   Max:    {stats['area_stats']['max']:.2f} px²")
        print(f"   Mean:   {stats['area_stats']['mean']:.2f} px²")
        print(f"   Median: {stats['area_stats']['median']:.2f} px²")
        
        # Determine optimal strides
        optimal_strides = self.determine_optimal_strides(num_scales=num_scales)
        grid_sizes = self.calculate_grid_sizes(optimal_strides)
        
        print("\n" + "="*70)
        print("✅ OPTIMAL CONFIGURATION")
        print("="*70)
        print(f"Strides:    {optimal_strides}")
        print(f"Grid sizes: {grid_sizes} (at {self.img_size}x{self.img_size})")
        
        # Load base config
        print(f"\n📄 Loading base config: {base_config}")
        with open(base_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add dynamic info as comments (YAML doesn't support custom keys well)
        config['_dynamic_info'] = {
            'strides': optimal_strides,
            'grid_sizes': grid_sizes,
            'img_size': self.img_size,
            'optimized_for': 'VisDrone' if stats['total_objects'] > 0 else 'custom',
            'dataset_stats': {
                'total_objects': stats['total_objects'],
                'mean_width': stats['width_stats']['mean'],
                'mean_height': stats['height_stats']['mean'],
                'mean_area': stats['area_stats']['mean']
            }
        }
        
        # Update nc (number of classes) if available in data yaml
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            if 'nc' in data_config:
                config['nc'] = data_config['nc']
                print(f"   Updated nc (number of classes) to: {config['nc']}")
        
        # Modify head architecture based on number of scales
        config['head'] = self._generate_head_config(num_scales, optimal_strides)
        
        print(f"   Modified head for {num_scales} detection scales")
        
        # Save modified config
        with open(output_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n💾 Saved dynamic config to: {output_config}")
        print("="*70 + "\n")
        
        return optimal_strides, grid_sizes, stats
    
    def _generate_head_config(self, num_scales: int, strides: List[int]) -> List:
        """Generate head configuration based on number of scales for YOLOv12"""
        
        if num_scales == 2:
            # 2-scale detection head (P3, P4) - strides [8, 16] or [4, 8]
            head = [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 2, 'A2C2f', [512, False, -1]],  # 11 - P4
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 2, 'A2C2f', [256, False, -1]],  # 14 - P3
                
                [[14, 11], 1, 'Detect', ['nc']]  # Detect(P3, P4)
            ]
        
        elif num_scales == 3:
            # 3-scale detection head (P3, P4, P5) - strides [8, 16, 32] or [4, 8, 16]
            head = [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 2, 'A2C2f', [512, False, -1]],  # 11 - P4
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 2, 'A2C2f', [256, False, -1]],  # 14 - P3
                
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 11], 1, 'Concat', [1]],  # cat head P4
                [-1, 2, 'A2C2f', [512, False, -1]],  # 17 - P4
                
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 8], 1, 'Concat', [1]],  # cat head P5
                [-1, 2, 'C3k2', [1024, True]],  # 20 - P5
                
                [[14, 17, 20], 1, 'Detect', ['nc']]  # Detect(P3, P4, P5)
            ]
        
        elif num_scales == 4:
            # 4-scale detection head (P2, P3, P4, P5) - strides [4, 8, 16, 32]
            # For VisDrone and very small objects
            head = [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                [-1, 2, 'A2C2f', [512, False, -1]],  # 11
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                [-1, 2, 'A2C2f', [256, False, -1]],  # 14
                
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 2], 1, 'Concat', [1]],  # cat backbone P2
                [-1, 2, 'A2C2f', [128, False, -1]],  # 17 - P2 (finest, stride 4)
                
                [-1, 1, 'Conv', [128, 3, 2]],
                [[-1, 14], 1, 'Concat', [1]],  # cat with P3
                [-1, 2, 'A2C2f', [256, False, -1]],  # 20 - P3 (stride 8)
                
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 11], 1, 'Concat', [1]],  # cat with P4
                [-1, 2, 'A2C2f', [512, False, -1]],  # 23 - P4 (stride 16)
                
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 8], 1, 'Concat', [1]],  # cat with P5
                [-1, 2, 'C3k2', [1024, True]],  # 26 - P5 (stride 32)
                
                [[17, 20, 23, 26], 1, 'Detect', ['nc']]  # Detect(P2, P3, P4, P5)
            ]
        
        return head
    
    def visualize_distribution(self, save_path: str = None):
        """Visualize object size distribution"""
        if not self.bbox_areas:
            print("❌ No data to visualize. Run analyze_bbox_sizes first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Area histogram
        axes[0, 0].hist(self.bbox_areas, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Area (pixels²)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Object Area Distribution')
        axes[0, 0].axvline(32*32, color='r', linestyle='--', label='Small/Medium (32x32)')
        axes[0, 0].axvline(96*96, color='g', linestyle='--', label='Medium/Large (96x96)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Width/Height scatter
        widths, heights = zip(*self.bbox_sizes)
        axes[0, 1].scatter(widths, heights, alpha=0.3, s=1)
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Height (pixels)')
        axes[0, 1].set_title('Object Dimensions')
        axes[0, 1].plot([0, max(widths)], [0, max(widths)], 'r--', alpha=0.5, label='Square')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Width histogram
        axes[1, 0].hist(widths, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Object Width Distribution')
        axes[1, 0].axvline(32, color='r', linestyle='--', alpha=0.5, label='32px')
        axes[1, 0].axvline(96, color='g', linestyle='--', alpha=0.5, label='96px')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Height histogram
        axes[1, 1].hist(heights, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Height (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Object Height Distribution')
        axes[1, 1].axvline(32, color='r', linestyle='--', alpha=0.5, label='32px')
        axes[1, 1].axvline(96, color='g', linestyle='--', alpha=0.5, label='96px')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.show()


# Example Usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 YOLOV12 DYNAMIC GRID ANALYZER")
    print("="*70)
    print("Analyzing VisDrone Dataset for Optimal Detection Grids")
    print("="*70 + "\n")
    
    analyzer = DynamicGridAnalyzer(
        data_yaml='visdrone.yaml',
        img_size=640
    )
    
    # Generate dynamic config
    strides, grids, stats = analyzer.generate_dynamic_config(
        base_config='ultralytics/cfg/models/v12/yolov12.yaml',
        output_config='yolov12n-visdrone-dynamic.yaml',
        num_scales=4  # Use 4 scales for VisDrone's small objects
    )
    
    if strides is not None:
        # Visualize
        print("\n📊 Generating visualization...")
        analyzer.visualize_distribution('visdrone_distribution.png')
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"Generated files:")
        print(f"   1. yolov12n-visdrone-dynamic.yaml - Model configuration")
        print(f"   2. visdrone_distribution.png - Object size visualization")
        print(f"\nNext steps:")
        print(f"   1. Review the configuration and visualization")
        print(f"   2. Train with: yolo train model=yolov12n-visdrone-dynamic.yaml data=visdrone.yaml")
        print("="*70 + "\n")
