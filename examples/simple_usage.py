#!/usr/bin/env python3
"""
Simple Usage Example - OCT Classification

Demonstrates basic usage of the OCT predictor module.
"""

from oct_predictor import OCTPredictor

def main():
    print("=" * 60)
    print("OCT Classification - Simple Usage Example")
    print("=" * 60)

    # Initialize predictor
    print("\n1. Loading model...")
    predictor = OCTPredictor('models/best_oct_model.h5')

    # Make prediction
    print("\n2. Making prediction...")
    image_path = 'data/kermany2018/OCT2017/test/NORMAL/NORMAL-1000-1.jpeg'

    try:
        result = predictor.predict(image_path)

        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Image:      {image_path}")
        print(f"Class:      {result['predicted_class']}")
        print(f"Condition:  {result['full_name']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Time:       {result['inference_time_ms']:.2f}ms")

        print("\nProbability Distribution:")
        for class_name, prob in result['probabilities'].items():
            bar = '█' * int(prob / 2)  # Scale to 50 chars max
            print(f"  {class_name:8s} {prob:5.1f}% {bar}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
