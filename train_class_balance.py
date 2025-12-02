#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import AUC, Precision, Recall

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("OCT TRAINING")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU'))} GPUs")
print("="*70 + "\n")

# Configuration
TRAIN_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/train'
TEST_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/test'
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
MAX_SAMPLES_PER_CLASS = 8000 # Using 8000 samples per class
VAL_SPLIT = 0.2

print("Configuration:")
print(f"  Train directory: {TRAIN_DIR}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Max samples per class: {MAX_SAMPLES_PER_CLASS}\n")

# Check if data exists
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Training directory not found: {TRAIN_DIR}")
    exit(1)

print("Training directory found\n")

# Data loading function
def load_balanced_dataset(data_dir, max_samples, val_split, seed=42):
    """Loads and balances dataset, then splits into train and validation sets."""
    all_fpaths = []
    all_labels = []
    
    # Map class name to an integer label
    class_names = sorted(os.listdir(data_dir))
    class_to_label = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    print(f"Found {num_classes} classes: {class_names}")

    # 1. Collect and balance file paths
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        fpaths = glob(os.path.join(class_path, '*'))
        
        # Randomly sample up to max_samples
        if len(fpaths) > max_samples:
            print(f"  Limiting class '{class_name}' from {len(fpaths)} to {max_samples} samples.")
            fpaths = np.random.choice(fpaths, size=max_samples, replace=False)
        else:
            print(f"  Using all {len(fpaths)} samples for class '{class_name}'.")

        all_fpaths.extend(fpaths)
        all_labels.extend([class_to_label[class_name]] * len(fpaths))

    all_fpaths = np.array(all_fpaths)
    all_labels = np.array(all_labels)
    
    # One-Hot Encode labels
    all_labels_one_hot = keras.utils.to_categorical(all_labels, num_classes=num_classes)
    
    print(f"Total balanced samples: {len(all_fpaths)}")
    
    # 2. Split into training and validation sets
    train_fpaths, val_fpaths, train_labels, val_labels = train_test_split(
        all_fpaths, all_labels_one_hot, test_size=val_split, random_state=seed, stratify=all_labels
    )

    print(f"Train samples: {len(train_fpaths)}")
    print(f"Validation samples: {len(val_fpaths)}\n")

    # 3. Create tf.data.Dataset objects
    
    # Helper function to load and decode image
    def load_and_preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    def create_dataset(fpaths, labels, batch_size):
        # Create dataset from file paths and one-hot labels
        path_ds = tf.data.Dataset.from_tensor_slices(fpaths)
        img_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)

        # Combine image and label datasets
        ds = tf.data.Dataset.zip((img_ds, label_ds))
        
        # Shuffle, batch, and prefetch for performance
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        return ds

    train_ds = create_dataset(train_fpaths, train_labels, BATCH_SIZE)
    val_ds = create_dataset(val_fpaths, val_labels, BATCH_SIZE)

    return train_ds, val_ds, class_names

# Load dataset using the custom function
print("Loading and balancing dataset...")
train_ds, val_ds, class_names = load_balanced_dataset(
    TRAIN_DIR, MAX_SAMPLES_PER_CLASS, VAL_SPLIT
)

print(f"Classes found: {class_names}")
print(f"Training batches: {len(train_ds)}")
print(f"Validation batches: {len(val_ds)}\n")

# Preprocessing - Note: The initial image loading function already resizes, so we only apply the final 1./255 scaling.
normalization_layer = layers.Rescaling(1./255)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply normalization to all datasets
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Apply data augmentation ONLY to the training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

# Memory-efficient prefetch (NO CACHE to prevent memory leak)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Data preprocessing configured (memory-efficient mode)\n")

# Create model
print("Creating model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
# Set the entire base model to trainable
base_model.trainable = True

# Freeze all but the last 20 layers
print("Unfreezing the last 20 layers for fine-tuning...")
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print("Model created")
print(f"  Trainable parameters (including fine-tuned layers): {trainable:,}")
print(f"  Non-trainable parameters: {non_trainable:,}\n")

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',  # Metrics
        AUC(name='auc', curve='ROC', multi_label=True), 
        Precision(name='precision'),
        Recall(name='recall')
    ]
)
print("Model compiled with Accuracy, AUC, Precision, and Recall\n")

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
        'models/simple_oct_model_balanced.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        'models/training_log_balanced.csv',
        append=False
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    MemoryCleanupCallback()
]

print("="*70)
print("STARTING TRAINING")
print("="*70)
print("Expected results:")
print("="*70)
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
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
    print("="*70 + "\n")

    # Save final model
    model.save('models/simple_oct_final_balanced.h5')
    print("Final model saved to: models/simple_oct_final_balanced.h5")

    # Plot and save training curves
    print("Generating training plots...")
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
    plt.savefig('models/training_curves_balanced.png', dpi=100)
    print("Training curves saved to: models/training_curves_balanced.png\n")

    print("="*70)
    print("ALL DONE! 🎉")
    print("="*70)
    print("\nYour trained model is ready at:")
    print("  models/simple_oct_final_balanced.h5")
    print("\nTo use it:")
    print("  python predict.py path/to/image.jpg --model models/simple_oct_final_balanced.h5")
    print("="*70)

except KeyboardInterrupt:
    print("\n\nTraining interrupted")

except Exception as e:
    print(f"\n\nERROR during training: {e}")
    import traceback
    traceback.print_exc()