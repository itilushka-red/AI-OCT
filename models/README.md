# Models Directory

This directory contains trained model files.

## 📁 Files Created After Training

After running the training notebook, you'll find:

```
models/
├── best_oct_model.h5              # Best model (saved during training)
├── final_oct_model.h5             # Final model after all epochs
├── oct_model_savedmodel/          # TensorFlow SavedModel format
│   ├── saved_model.pb
│   ├── variables/
│   └── assets/
├── training_history.csv           # Training metrics per epoch
├── classification_report.csv      # Test set performance
└── model_config.json              # Model configuration
```

## 🚀 Usage

### Load Trained Model

```python
from tensorflow import keras

# Load model
model = keras.models.load_model('models/best_oct_model.h5')

# Or use the predictor
from oct_predictor import OCTPredictor
predictor = OCTPredictor('models/best_oct_model.h5')
```

## 📊 Model Information

- **Architecture**: EfficientNetB0 + Custom Classifier
- **Input Size**: 224×224×3 (RGB)
- **Output**: 4 classes (CNV, DME, DRUSEN, NORMAL)
- **Parameters**: ~5.3M total, ~270K trainable
- **Size**: ~20 MB

## ⚠️ Important Notes

1. **Not in Git**: Model files are excluded from git (too large)
2. **Train Locally**: Run the Jupyter notebook to train
3. **Or Download**: Get pre-trained models from releases (if available)

## 🔄 Retraining

To retrain the model:

```bash
jupyter notebook notebooks/oct_classification.ipynb
# Run all cells
```

## 📈 Expected Performance

- **Accuracy**: 92-96%
- **Training Time**: 2-4 hours (CPU), 30-60 min (GPU)
- **Inference Time**: 45ms per image (CPU)

## 📖 Related Files

- **Training**: `notebooks/oct_classification.ipynb`
- **Inference**: `predict.py` or `oct_predictor.py`
- **Documentation**: `docs/INFERENCE_GUIDE.md`
