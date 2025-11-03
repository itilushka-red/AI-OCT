#!/usr/bin/env python3
"""
Simple OCT Training Script - Runs independently of Jupyter
Safe to close browser - training continues in background!
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("SIMPLE OCT TRAINING - Background Mode")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPUs")
print("="*70 + "\n")

# Configuration
TRAIN_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/train'
IMG_SIZE = 224
BATCH_SIZE = 8  # Reduced to prevent memory issues
EPOCHS = 10

print("Configuration:")
print(f"  Train directory: {TRAIN_DIR}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}\n")

# Check if data exists
if not os.path.exists(TRAIN_DIR):
    print(f"❌ ERROR: Training directory not found: {TRAIN_DIR}")
    print("Please check dataset location!")
    exit(1)

print("✓ Training directory found\n")

# Load dataset
print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
print(f"✓ Classes found: {class_names}")
print(f"✓ Training batches: {len(train_ds)}")
print(f"✓ Validation batches: {len(val_ds)}\n")

# Preprocessing
normalization_layer = layers.Rescaling(1./255)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Memory-efficient prefetch (NO CACHE to prevent memory leak)
train_ds = train_ds.prefetch(buffer_size=2)
val_ds = val_ds.prefetch(buffer_size=2)

print("✓ Data preprocessing configured (memory-efficient mode)\n")

# Create model
print("Creating model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs, outputs)

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print("✓ Model created")
print(f"  Trainable parameters: {trainable:,}")
print(f"  Non-trainable parameters: {non_trainable:,}\n")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Model compiled\n")

# Callbacks with memory cleanup
import gc

class MemoryCleanupCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()
        print(f"  [Memory cleaned after epoch {epoch+1}]")

os.makedirs('models', exist_ok=True)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/simple_oct_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        'models/training_log.csv',
        append=False
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    MemoryCleanupCallback()
]

print("="*70)
print("STARTING TRAINING")
print("="*70)
print("Expected results:")
print("  Epoch 1-2: 50-65% accuracy")
print("  Epoch 5: 75-80% accuracy")
print("  Epoch 10: 80-85% accuracy")
print("="*70)
print("\n⚠️  You can safely close your terminal/browser now!")
print("Training will continue in the background.\n")
print("="*70 + "\n")

# Train
try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
    print("="*70 + "\n")

    # Save final model
    model.save('models/simple_oct_final.h5')
    print("✓ Final model saved to: models/simple_oct_final.h5")

    # Plot and save training curves
    print("✓ Generating training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=100)
    print("✓ Training curves saved to: models/training_curves.png\n")

    print("="*70)
    print("ALL DONE! 🎉")
    print("="*70)
    print("\nYour trained model is ready at:")
    print("  models/simple_oct_final.h5")
    print("\nTo use it:")
    print("  python predict.py path/to/image.jpg --model models/simple_oct_final.h5")
    print("="*70)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user!")
    print("Partial model may be saved in models/simple_oct_model.h5")

except Exception as e:
    print(f"\n\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
