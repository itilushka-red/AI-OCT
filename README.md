# OCT Image Classification Project

Medical image classification for retinal OCT scans using EfficientNetB0 and Transfer Learning.

## 🎯 Project Goal

Classify retinal OCT (Optical Coherence Tomography) images into 4 categories:
- **NORMAL**: Healthy retina
- **CNV**: Choroidal Neovascularization
- **DME**: Diabetic Macular Edema
- **DRUSEN**: Drusen deposits

## 🏗️ System Requirements

**Your Setup (Optimized)**:
- CPU: Intel Xeon E5-2680 v4 (14 cores, 28 threads) ✅
- RAM: 64 GB ✅
- GPU: AMD RX 5700 XT (CPU training recommended)
- OS: WSL2 on Windows

**Expected Performance**:
- Training time: 2-4 hours
- Accuracy: 92-96%
- Model size: ~20 MB

## 📁 Project Structure

```
AIOCT/
├── notebooks/
│   └── oct_classification.ipynb    # Main notebook (optimized for your system)
├── models/                          # Saved models (created after training)
├── data/
│   └── kermany2018/                # Dataset (download separately)
├── docs/
│   ├── QUICK_START.md             # Start here! ⭐
│   ├── AMD_GPU_SETUP.md           # CPU optimization guide
│   └── PROJECT_SETUP.md           # Detailed documentation
├── setup.sh                        # Automated setup script
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start (3 Steps)

### 1. Run Setup Script
```bash
cd AIOCT
./setup.sh
```

### 2. Download Dataset
```bash
# Option A: Kaggle API (recommended)
pip install kaggle
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/kermany2018/

# Option B: Manual download from
# https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data
```

### 3. Start Training
```bash
source venv/bin/activate
jupyter notebook notebooks/oct_classification.ipynb

# IMPORTANT: Run Cell 1.5 for CPU optimization!
```

## 📖 Documentation

- **🌟 [QUICK_START.md](docs/QUICK_START.md)** - Start here for setup
- **💻 [AMD_GPU_SETUP.md](docs/AMD_GPU_SETUP.md)** - CPU optimization details
- **📚 [PROJECT_SETUP.md](docs/PROJECT_SETUP.md)** - Full documentation

## 🧠 Model Architecture

**EfficientNetB0 with Transfer Learning**

```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet weights, frozen)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (4 units, Softmax)
```

**Why EfficientNetB0?**
- ✅ State-of-the-art accuracy
- ✅ Fast training
- ✅ Pre-trained on ImageNet
- ✅ Proven for medical imaging

## 📊 Dataset Details

- **Source**: Kermany et al., 2018
- **Total images**: 84,495 OCT scans
- **Classes**: 4 (CNV, DME, DRUSEN, NORMAL)
- **Format**: JPEG images
- **Split**: Pre-split into train/val/test
- **Validation**: Triple-verified by medical experts

## ⚙️ Configuration

**Optimized for your Xeon CPU**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image Size | 224×224 | EfficientNet input |
| Batch Size | 64 | Increased for 64GB RAM |
| Epochs | 25 | With early stopping |
| Optimizer | Adam (lr=0.0001) | |
| CPU Threads | 28 | All Xeon threads |

## 📈 Training Process

The notebook guides you through:

1. **Data Exploration** (Cells 1-3)
   - Analyze 84K images
   - Visualize class distribution
   - Display sample OCT scans

2. **Preprocessing** (Cells 4-5)
   - Data augmentation (rotation, flip, zoom)
   - Create train/val/test generators

3. **Model Building** (Cells 6-7)
   - Load EfficientNetB0
   - Add custom classifier head
   - Configure optimizer

4. **Training** (Cells 8-9)
   - Train with callbacks
   - Monitor metrics
   - Save best model

5. **Evaluation** (Cells 10-14)
   - Test set accuracy
   - Confusion matrix
   - Per-class metrics

6. **Inference** (Cells 15-17)
   - Predict new images
   - Export trained model

## 🎯 Expected Results

After training completes:

**Metrics**:
- Test Accuracy: 92-96%
- Precision: ~0.93
- Recall: ~0.92
- F1-Score: ~0.93

**Outputs**:
- `models/best_oct_model.h5` - Best model
- `models/training_history.csv` - Training metrics
- `models/classification_report.csv` - Detailed results
- `models/model_config.json` - Configuration

## 💡 Usage Examples

### Training
```bash
# Run all cells in notebook sequentially
# Monitor progress: ~5-10 min per epoch
# Total time: ~2-4 hours
```

### Inference (After Training)
```python
from tensorflow import keras
from PIL import Image
import numpy as np

# Load model
model = keras.models.load_model('models/best_oct_model.h5')

# Predict
img = Image.open('scan.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
predicted_class = classes[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
```

## 🔧 Troubleshooting

### Training is slow
1. Verify Cell 1.5 was executed (CPU optimization)
2. Check `htop` shows 28 cores at 90-100%
3. Increase BATCH_SIZE to 128

### Out of Memory
```python
BATCH_SIZE = 32  # Reduce from 64
```

### Dataset not found
```python
# Check path in Cell 2
BASE_DIR = '../data/kermany2018/OCT2017'
```

### TensorFlow not found
```bash
source venv/bin/activate
pip install intel-tensorflow
```

## 📚 Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **EfficientNetB0**: Base CNN architecture
- **Pillow**: Image processing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Metrics and evaluation
- **Jupyter**: Interactive development

## 🎓 Learning Outcomes

By completing this project, you will:
- ✅ Understand transfer learning with CNNs
- ✅ Implement data augmentation for medical images
- ✅ Train state-of-the-art image classifiers
- ✅ Evaluate models with multiple metrics
- ✅ Create production-ready inference pipelines
- ✅ Work with real medical imaging datasets

## 📝 For Your Report

Key sections to include:

1. **Introduction**: Medical OCT imaging and classification
2. **Dataset**: 84K images, 4 classes, expert-verified
3. **Architecture**: EfficientNetB0 with transfer learning
4. **Training**: CPU-optimized, 25 epochs, callbacks
5. **Results**: Accuracy, confusion matrix, metrics
6. **Visualizations**: Include plots from notebook
7. **Conclusion**: Model performance and applications

## 🚀 Next Steps (Advanced)

After basic training:

1. **Fine-tuning**: Unfreeze EfficientNet layers
   ```python
   base_model.trainable = True
   for layer in base_model.layers[:100]:
       layer.trainable = False
   ```

2. **Ensemble**: Train multiple models
   - EfficientNetB0
   - ResNet50
   - MobileNetV2

3. **Class Activation Maps**: Visualize model focus

4. **Web Deployment**: Create Flask/FastAPI app

5. **Cross-validation**: K-fold evaluation

## 📞 Support

**If you encounter issues:**
1. Check [QUICK_START.md](docs/QUICK_START.md) first
2. Review [AMD_GPU_SETUP.md](docs/AMD_GPU_SETUP.md) for CPU setup
3. Verify Python environment: `source venv/bin/activate`
4. Check dataset path: `ls data/kermany2018/OCT2017/`

## 📖 References

- **Dataset**: [Kermany et al., 2018 - Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- **EfficientNet**: [Tan & Le, 2019 - ICML](https://arxiv.org/abs/1905.11946)
- **Transfer Learning**: [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/images/transfer_learning)
- **Medical Imaging AI**: [Stanford ML Group](https://stanfordmlgroup.github.io/)

## 📄 License

This project is for educational purposes. Dataset is from:
> Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3

## 🏆 Project Highlights

- ✅ Complete end-to-end ML pipeline
- ✅ Optimized for your specific hardware
- ✅ Production-ready code with best practices
- ✅ Comprehensive documentation
- ✅ Expected 92-96% accuracy
- ✅ Real medical imaging application

---

**Ready to start?** Follow [QUICK_START.md](docs/QUICK_START.md) now! 🚀

**Estimated Time**: Setup (5 min) + Training (2-4 hours) = **~2.5-4 hours total**
