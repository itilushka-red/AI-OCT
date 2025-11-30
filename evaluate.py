#!/usr/bin/env python3
"""
Model Evaluation Script
Generates comprehensive metrics and visualizations for trained OCT model
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime

print("="*70)
print("OCT MODEL EVALUATION")
print("="*70)
print(f"TensorFlow version: {tf.__version__}\n")

# Configuration
MODEL_PATH = 'models/simple_oct_final_balanced.h5'
TEST_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/test'
VAL_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/val'
IMG_SIZE = 224
BATCH_SIZE = 32  # Can use larger batch for inference
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

print("Configuration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Test directory: {TEST_DIR}")
print(f"  Validation directory: {VAL_DIR}")
print(f"  Classes: {CLASS_NAMES}\n")

# Check if directories exist - prefer TEST over VAL
use_test = os.path.exists(TEST_DIR)
use_val = os.path.exists(VAL_DIR)

if not use_test and not use_val:
    print("ERROR: Neither test nor validation directory found!")
    print(f"Looked for:\n  {TEST_DIR}\n  {VAL_DIR}")
    exit(1)

# Choose which dataset to use (PREFER TEST)
EVAL_DIR = TEST_DIR if use_test else VAL_DIR
dataset_name = "Test" if use_test else "Validation"
print(f"Using {dataset_name} dataset: {EVAL_DIR}\n")

# Load model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# Load dataset
print("Loading dataset...")
try:
    eval_ds = tf.keras.utils.image_dataset_from_directory(
        EVAL_DIR,
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False  # Important for consistent ordering
    )
    print(f"✓ Dataset loaded: {len(eval_ds)} batches\n")
except Exception as e:
    print(f"❌ ERROR loading dataset: {e}")
    exit(1)

# Preprocess (same as training)
normalization_layer = keras.layers.Rescaling(1./255)
eval_ds = eval_ds.map(lambda x, y: (normalization_layer(x), y))

# Make predictions
print("Making predictions...")
print("This may take a few minutes...\n")

y_true = []
y_pred = []

for batch_idx, (images, labels) in enumerate(eval_ds):
    predictions = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

    if (batch_idx + 1) % 10 == 0:
        print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} images...")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✓ Predictions complete: {len(y_true)} images evaluated\n")

# Calculate metrics
print("="*70)
print("EVALUATION METRICS")
print("="*70)

accuracy = accuracy_score(y_true, y_pred)
print(f"\n{'Overall Accuracy:':<25} {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Detailed classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    output_dict=True
)

print("Per-Class Metrics:")
print("-" * 70)
print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print("-" * 70)

for class_name in CLASS_NAMES:
    metrics = report[class_name]
    print(f"{class_name:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
          f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}")

print("-" * 70)
print(f"{'Macro Avg':<12} {report['macro avg']['precision']:<12.4f} "
      f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
print(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:<12.4f} "
      f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
print("="*70 + "\n")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print("-" * 70)
print(f"{'Predicted>':<12}", end="")
for name in CLASS_NAMES:
    print(f"{name:<12}", end="")
print()
print(f"{'Actual v':<12}")
print("-" * 70)

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name:<12}", end="")
    for j in range(len(CLASS_NAMES)):
        print(f"{cm[i][j]:<12}", end="")
    print()
print("="*70 + "\n")

# Create visualizations
print("Generating visualizations...")

# 1. Confusion Matrix Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_title(f'Confusion Matrix - {dataset_name} Set\nAccuracy: {accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

# Normalized confusion matrix (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax2, cbar_kws={'label': 'Percentage'})
ax2.set_title('Normalized Confusion Matrix (by True Label)',
              fontsize=14, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved: models/confusion_matrix.png")

# 2. Per-class metrics bar chart
fig, ax = plt.subplots(figsize=(12, 6))

metrics_data = {
    'Precision': [report[cls]['precision'] for cls in CLASS_NAMES],
    'Recall': [report[cls]['recall'] for cls in CLASS_NAMES],
    'F1-Score': [report[cls]['f1-score'] for cls in CLASS_NAMES]
}

x = np.arange(len(CLASS_NAMES))
width = 0.25

bars1 = ax.bar(x - width, metrics_data['Precision'], width, label='Precision', color='#3498db')
bars2 = ax.bar(x, metrics_data['Recall'], width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', color='#e74c3c')

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Per-Class Performance Metrics - {dataset_name} Set',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('models/per_class_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Per-class metrics saved: models/per_class_metrics.png")

# 3. Class distribution
class_counts = [np.sum(y_true == i) for i in range(len(CLASS_NAMES))]
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(CLASS_NAMES, class_counts, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title(f'{dataset_name} Set - Class Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('models/class_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Class distribution saved: models/class_distribution.png\n")

# Save detailed report to JSON
report_data = {
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': dataset_name,
    'dataset_path': EVAL_DIR,
    'model_path': MODEL_PATH,
    'total_images': int(len(y_true)),
    'overall_accuracy': float(accuracy),
    'per_class_metrics': {},
    'confusion_matrix': cm.tolist(),
    'class_names': CLASS_NAMES
}

for class_name in CLASS_NAMES:
    report_data['per_class_metrics'][class_name] = {
        'precision': float(report[class_name]['precision']),
        'recall': float(report[class_name]['recall']),
        'f1_score': float(report[class_name]['f1-score']),
        'support': int(report[class_name]['support'])
    }

report_data['macro_avg'] = {
    'precision': float(report['macro avg']['precision']),
    'recall': float(report['macro avg']['recall']),
    'f1_score': float(report['macro avg']['f1-score'])
}

report_data['weighted_avg'] = {
    'precision': float(report['weighted avg']['precision']),
    'recall': float(report['weighted avg']['recall']),
    'f1_score': float(report['weighted avg']['f1-score'])
}

with open('models/evaluation_report.json', 'w') as f:
    json.dump(report_data, f, indent=2)

print("✓ Detailed report saved: models/evaluation_report.json\n")

# Save text report
with open('models/evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("OCT MODEL EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total Images: {len(y_true)}\n\n")

    f.write("="*70 + "\n")
    f.write("OVERALL METRICS\n")
    f.write("="*70 + "\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

    f.write("="*70 + "\n")
    f.write("PER-CLASS METRICS\n")
    f.write("="*70 + "\n")
    f.write(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
    f.write("-" * 70 + "\n")

    for class_name in CLASS_NAMES:
        metrics = report[class_name]
        f.write(f"{class_name:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<12}\n")

    f.write("-" * 70 + "\n")
    f.write(f"{'Macro Avg':<12} {report['macro avg']['precision']:<12.4f} "
            f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}\n")
    f.write(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:<12.4f} "
            f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}\n")

    f.write("\n" + "="*70 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("="*70 + "\n")
    f.write(f"{'Predicted>':<12}")
    for name in CLASS_NAMES:
        f.write(f"{name:<12}")
    f.write("\n" + f"{'Actual v':<12}\n")
    f.write("-" * 70 + "\n")

    for i, class_name in enumerate(CLASS_NAMES):
        f.write(f"{class_name:<12}")
        for j in range(len(CLASS_NAMES)):
            f.write(f"{cm[i][j]:<12}")
        f.write("\n")

    f.write("\n" + "="*70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*70 + "\n\n")

    # Find best and worst performing classes
    f1_scores = [(cls, report[cls]['f1-score']) for cls in CLASS_NAMES]
    f1_scores.sort(key=lambda x: x[1], reverse=True)

    f.write(f"Best Performing Class: {f1_scores[0][0]} (F1: {f1_scores[0][1]:.4f})\n")
    f.write(f"Worst Performing Class: {f1_scores[-1][0]} (F1: {f1_scores[-1][1]:.4f})\n\n")

    if accuracy >= 0.85:
        f.write("Model Performance: EXCELLENT ✓\n")
        f.write("The model achieves strong classification accuracy across all classes.\n")
    elif accuracy >= 0.80:
        f.write("Model Performance: GOOD ✓\n")
        f.write("The model performs well with room for minor improvements.\n")
    elif accuracy >= 0.70:
        f.write("Model Performance: ACCEPTABLE\n")
        f.write("The model shows reasonable performance but could benefit from further training.\n")
    else:
        f.write("Model Performance: NEEDS IMPROVEMENT\n")
        f.write("Consider retraining with different hyperparameters or architecture.\n")

print("✓ Text report saved: models/evaluation_report.txt\n")

print("="*70)
print("EVALUATION COMPLETE! ✅")
print("="*70)
print("\nGenerated files:")
print("  1. models/evaluation_report.txt - Detailed text report")
print("  2. models/evaluation_report.json - Machine-readable report")
print("  3. models/confusion_matrix.png - Confusion matrix visualization")
print("  4. models/per_class_metrics.png - Per-class performance chart")
print("  5. models/class_distribution.png - Dataset distribution chart")
print("="*70)
print("\n📊 Summary:")
print(f"   Overall Accuracy: {accuracy*100:.2f}%")
print(f"   Total Images: {len(y_true)}")
print(f"   Classes: {', '.join(CLASS_NAMES)}")
print("\n✓ Ready for your class report!")
print("="*70)
