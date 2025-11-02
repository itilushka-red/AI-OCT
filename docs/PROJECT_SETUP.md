# OCT Image Classification Project Setup

## 📋 Project Overview

Medical image classification for retinal OCT scans using deep learning.

**Goal**: Classify retinal images into 4 categories:
- **NORMAL**: Healthy retina
- **CNV**: Choroidal Neovascularization
- **DME**: Diabetic Macular Edema
- **DRUSEN**: Drusen deposits

## 🏗️ Project Structure

```
AIOCT/
├── notebooks/
│   └── oct_classification.ipynb    # Main Jupyter notebook (17 cells)
├── models/                          # Saved models (created after training)
│   ├── best_oct_model.h5
│   ├── final_oct_model.h5
│   ├── training_history.csv
│   └── model_config.json
├── data/                            # Dataset directory
│   └── kermany2018/
│       └── OCT2017/
│           ├── train/
│           ├── test/
│           └── val/
└── docs/
    └── PROJECT_SETUP.md            # This file
```

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow numpy pandas jupyter
```

**Required versions**:
- TensorFlow >= 2.10
- Python >= 3.8

### Step 2: Download Dataset

**Option A: Kaggle API** (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d paultimothymooney/kermany2018

# Extract to data folder
unzip kermany2018.zip -d data/kermany2018/
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data
2. Download dataset (5.2 GB)
3. Extract to `data/kermany2018/`

### Step 3: Verify Dataset Structure

Your data folder should look like:
```
data/kermany2018/OCT2017/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── test/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── val/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

### Step 4: Run Jupyter Notebook

```bash
# Navigate to project directory
cd AIOCT

# Launch Jupyter
jupyter notebook notebooks/oct_classification.ipynb
```

## 🧠 Model Architecture

**EfficientNetB0 with Transfer Learning**

```
Input (224x224x3)
    ↓
EfficientNetB0 (pre-trained on ImageNet) [FROZEN]
    ↓
Global Average Pooling
    ↓
Dense (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (4 units, Softmax)
```

**Why EfficientNetB0?**
- ✅ State-of-the-art accuracy
- ✅ Fast training (smaller than ResNet)
- ✅ Pre-trained on ImageNet
- ✅ Excellent for medical imaging
- ✅ Easy to implement

## 📊 Expected Performance

Based on similar medical imaging tasks:
- **Accuracy**: 92-96%
- **Training time**: ~30-60 minutes (with GPU)
- **Model size**: ~20 MB

## 🎯 Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224x224 |
| Batch Size | 32 |
| Epochs | 30 (with early stopping) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Categorical Crossentropy |

**Data Augmentation**:
- Rotation: ±10°
- Width/Height shift: 10%
- Horizontal flip
- Zoom: 10%

**Callbacks**:
- Model Checkpoint (save best model)
- Early Stopping (patience=5)
- Reduce LR on Plateau (factor=0.5)

## 📓 Notebook Sections

The notebook contains 17 well-documented cells:

1. **Setup & Dependencies** - Import libraries
2. **Dataset Configuration** - Set paths and parameters
3. **Data Exploration** - Visualize dataset
4. **Class Distribution** - Analyze balance
5. **Sample Images** - Display examples
6. **Data Preprocessing** - Create generators
7. **Augmentation Visualization** - Show augmented images
8. **Model Creation** - Build EfficientNetB0
9. **Model Compilation** - Configure optimizer
10. **Callbacks Setup** - Training callbacks
11. **Model Training** - Train the network
12. **Training History** - Plot metrics
13. **Test Evaluation** - Evaluate on test set
14. **Classification Report** - Detailed metrics
15. **Confusion Matrix** - Visualize errors
16. **Prediction Examples** - Show predictions
17. **Confidence Analysis** - Analyze confidence
18. **Inference Function** - Predict new images
19. **Save Model** - Export trained model

## 💡 Usage Tips

### Training Tips:
1. **GPU Recommended**: Training will be 10-20x faster with GPU
2. **Start Small**: Try 5-10 epochs first to verify everything works
3. **Monitor Validation**: Watch for overfitting
4. **Adjust Batch Size**: Reduce if you get OOM errors

### Prediction Tips:
```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('models/best_oct_model.h5')

# Predict single image
def predict_oct(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    return predicted_class, confidence

# Use it
pred, conf = predict_oct('path/to/scan.jpg')
print(f"Prediction: {pred} ({conf:.2f}% confidence)")
```

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow numpy pandas
```

### Issue: "Cannot find dataset"
- Verify the `BASE_DIR` path in cell 3
- Check that `OCT2017` folder exists in `data/kermany2018/`

### Issue: "Out of Memory"
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Close other applications

### Issue: "Training too slow"
- Check if GPU is detected (cell 2)
- Install CUDA and cuDNN for GPU support
- Reduce `EPOCHS` for quick testing

## 📈 Next Steps After Training

### 1. Fine-Tuning
Unfreeze top layers of EfficientNetB0:
```python
base_model.trainable = True
# Freeze first 100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False
```

### 2. Ensemble Models
Combine multiple models for better accuracy:
- EfficientNetB0
- ResNet50
- MobileNetV2

### 3. Class Activation Maps (CAM)
Visualize what the model focuses on:
```python
from tf_keras_vis.gradcam import Gradcam
```

### 4. Deploy as Web App
Create a Flask/FastAPI application:
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load image, predict, return result
    pass
```

### 5. Cross-Validation
Implement k-fold cross-validation for robust evaluation

## 📚 References

- **Dataset**: [Kermany et al., 2018 - Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **Transfer Learning**: [TensorFlow Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **Medical Imaging**: [Stanford ML Group](https://stanfordmlgroup.github.io/)

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure dataset is properly downloaded
4. Check Python/TensorFlow versions

## 🎓 Learning Objectives

By completing this project, you will:
- ✅ Understand transfer learning with CNNs
- ✅ Implement data augmentation for medical images
- ✅ Train a state-of-the-art image classifier
- ✅ Evaluate model performance with multiple metrics
- ✅ Create inference pipelines for production
- ✅ Work with real medical imaging data

Good luck with your AI project! 🚀
