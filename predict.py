#!/usr/bin/env python3
"""
OCT Image Classification - Inference Script

Command-line interface for predicting OCT image classifications.
Outputs results in JSON format for easy integration with applications.

Usage:
    python predict.py <image_path> [options]

Examples:
    python predict.py scan.jpg
    python predict.py scans/patient_001.jpg --model models/final_oct_model.h5
    python predict.py test.jpg --pretty
    python predict.py scan.jpg --save results.json

Author: Generated for AIOCT Project
Date: 2025
"""

import sys
import os
import json
import argparse
from oct_predictor import OCTPredictor, validate_image


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='OCT Image Classification - Predict retinal conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan.jpg
  %(prog)s patient_001.jpg --pretty
  %(prog)s test.jpg --model models/final_oct_model.h5
  %(prog)s scan.jpg --save results.json

Classes:
  CNV     - Choroidal Neovascularization
  DME     - Diabetic Macular Edema
  DRUSEN  - Drusen deposits
  NORMAL  - Healthy retina
        """
    )

    parser.add_argument(
        'image',
        type=str,
        help='Path to OCT image file (JPG, PNG, etc.)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/best_oct_model.h5',
        help='Path to trained model file (default: models/best_oct_model.h5)'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output with indentation'
    )

    parser.add_argument(
        '--save',
        type=str,
        metavar='FILE',
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show additional information during prediction'
    )

    parser.add_argument(
        '--no-probabilities',
        action='store_true',
        help='Exclude probability distribution from output'
    )

    return parser.parse_args()


def print_error(message):
    """Print error message to stderr."""
    print(json.dumps({
        'success': False,
        'error': message
    }), file=sys.stderr)


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_arguments()

    try:
        # Validate image file
        if args.verbose:
            print(f"Validating image: {args.image}", file=sys.stderr)

        is_valid, error_msg = validate_image(args.image)
        if not is_valid:
            print_error(error_msg)
            sys.exit(1)

        # Initialize predictor
        if args.verbose:
            print(f"Loading model: {args.model}", file=sys.stderr)

        predictor = OCTPredictor(
            model_path=args.model,
            img_size=224
        )

        if args.verbose:
            print(f"Making prediction...", file=sys.stderr)

        # Make prediction
        result = predictor.predict(
            image_path=args.image,
            return_probabilities=not args.no_probabilities
        )

        # Add metadata
        result['success'] = True
        result['image_path'] = os.path.abspath(args.image)
        result['model_path'] = os.path.abspath(args.model)

        # Format output
        if args.pretty:
            output = json.dumps(result, indent=2)
        else:
            output = json.dumps(result)

        # Print to stdout
        print(output)

        # Save to file if requested
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(result, f, indent=2)
            if args.verbose:
                print(f"✓ Results saved to: {args.save}", file=sys.stderr)

        # Exit with success
        sys.exit(0)

    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
