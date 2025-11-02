#!/usr/bin/env python3
"""
Batch Processing Example - OCT Classification

Processes multiple images and generates a CSV report.
"""

import sys
import csv
from pathlib import Path
from oct_predictor import OCTPredictor

def process_directory(input_dir, output_csv='results.csv'):
    """
    Process all OCT images in a directory.

    Args:
        input_dir (str): Directory containing OCT images
        output_csv (str): Output CSV file path
    """
    print("=" * 60)
    print("OCT Classification - Batch Processing")
    print("=" * 60)

    # Initialize predictor
    print("\nLoading model...")
    predictor = OCTPredictor('models/best_oct_model.h5')

    # Find all images
    print(f"\nScanning directory: {input_dir}")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(input_dir).rglob(ext))

    print(f"Found {len(image_files)} images\n")

    if len(image_files) == 0:
        print("❌ No images found!")
        return

    # Process each image
    results = []
    for i, img_path in enumerate(image_files, 1):
        try:
            result = predictor.predict(str(img_path))

            results.append({
                'filename': img_path.name,
                'path': str(img_path),
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'full_name': result['full_name'],
                'inference_time_ms': result['inference_time_ms']
            })

            status = "✓"
            info = f"{result['predicted_class']} ({result['confidence']:.1f}%)"
            print(f"[{i:3d}/{len(image_files)}] {status} {img_path.name:40s} {info}")

        except Exception as e:
            print(f"[{i:3d}/{len(image_files)}] ✗ {img_path.name:40s} ERROR: {str(e)}")

    # Save to CSV
    if results:
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['filename', 'path', 'predicted_class', 'confidence',
                         'full_name', 'inference_time_ms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Results saved to: {output_csv}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total processed: {len(results)}")

        # Count by class
        class_counts = {}
        for r in results:
            cls = r['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\nClass Distribution:")
        for cls, count in sorted(class_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {cls:8s}: {count:3d} ({percentage:.1f}%)")

        # Average confidence
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print(f"\nAverage confidence: {avg_conf:.2f}%")

        # Average inference time
        avg_time = sum(r['inference_time_ms'] for r in results) / len(results)
        print(f"Average inference time: {avg_time:.2f}ms")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_processing.py <input_directory> [output_csv]")
        print("\nExample:")
        print("  python batch_processing.py data/kermany2018/OCT2017/test/")
        print("  python batch_processing.py scans/ results.csv")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'results.csv'

    process_directory(input_dir, output_csv)

if __name__ == "__main__":
    main()
