# OCT Retinal Image Classification

AI/ML project for classifying retinal OCT (Optical Coherence Tomography) images into 4 disease categories.

**Model Accuracy**: 85.13% validation accuracy
**Classes**: CNV, DME, DRUSEN, NORMAL
**Architecture**: MobileNetV2 with transfer learning
**Platform**: Windows (CPU training)

---

## Project Structure

```
AI-OCT/
├── data/
│   └── kermany2018/OCT2017/
│       ├── train/          # Training images (83,484 images)
│       ├── val/            # Validation images (32 images)
│       └── test/           # Test images (~1,000 images)
├── notebooks/
│   └── oct_simple.ipynb    # Training notebook
├── models/
│   ├── simple_oct_final.h5 # Trained model
│   ├── training_log.csv    # Training history
│   └── training_curves.png # Accuracy/loss plots
├── train_simple.py         # Background training script
├── evaluate.py             # Model evaluation script
├── predict.py              # Single image prediction CLI
├── api.py                  # FastAPI backend server
└── test_api.py             # API testing script
```

---

## Quick Start

### 1. Setup Environment

```cmd
cd C:\Users\illia\Desktop\AI\AI-OCT
python -m venv venv
venv\Scripts\activate
pip install tensorflow matplotlib numpy pillow jupyter fastapi uvicorn python-multipart scikit-learn seaborn requests
```

### 2. Train Model

**Option A: Using Jupyter Notebook**
```cmd
cd notebooks
jupyter notebook oct_simple.ipynb
```

Update Cell 4 with Windows path:
```python
TRAIN_DIR = 'C:/Users/illia/Desktop/AI/AI-OCT/data/kermany2018/OCT2017/train'
```

Then: `Cell → Run All`

**Option B: Background Script** (recommended for stability)
```cmd
python train_simple.py
```

Expected training time: **6-8 hours** (10 epochs)

### 3. Evaluate Model

```cmd
python evaluate.py
```

Generates comprehensive evaluation reports and visualizations.

### 4. Run API Server

```cmd
python api.py
```

API available at: **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

---

## Dataset

**Source**: Kermany et al. 2018 - Retinal OCT Images
**Total Images**: 84,495 images
**Split**: 80% train / 20% validation (automatic)

### Classes

| Class | Full Name | Description | Training Images |
|-------|-----------|-------------|-----------------|
| **CNV** | Choroidal Neovascularization | Abnormal blood vessels under retina | ~37,000 |
| **DME** | Diabetic Macular Edema | Fluid buildup from diabetes | ~11,000 |
| **DRUSEN** | Drusen | Yellow deposits, early AMD sign | ~8,000 |
| **NORMAL** | Normal Retina | Healthy retina | ~26,000 |

---

## Model Architecture

### Simple & Reliable Approach

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet weights, ALL FROZEN)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.2)
    ↓
Dense (4 units, Softmax)
```

**Key Design Decisions**:
- ✅ **MobileNetV2**: Lightweight, proven for medical imaging
- ✅ **All layers frozen**: Prevents destroying pre-trained weights
- ✅ **Simple classifier**: Just pooling + dropout + output
- ✅ **Transfer learning only**: No fine-tuning needed

**Total Parameters**: 2.3M (only 6K trainable)

---

## Training Configuration

```python
IMG_SIZE = 224
BATCH_SIZE = 8        # Small batch for memory stability
EPOCHS = 10
OPTIMIZER = Adam(learning_rate=0.001)
```

### Memory Optimizations

- No dataset caching (prevents memory leak)
- Batch size reduced to 8
- Single worker thread
- Memory cleanup callback after each epoch
- Prefetch buffer = 2

---

## Training Results

```
Epoch 1:  Training: 80.6%  |  Validation: 81.5%
Epoch 2:  Training: 82.0%  |  Validation: 84.2%
Epoch 5:  Training: 82.1%  |  Validation: 84.7%
Epoch 7:  Training: 82.3%  |  Validation: 85.1%  ← BEST
Epoch 10: Training: 82.3%  |  Validation: 84.6%

Early stopping triggered at Epoch 10
Best model from Epoch 7: 85.13% validation accuracy
```

### Training Details

- **Hardware**: Xeon E5-2680 v4 (14 cores, 64GB RAM)
- **Platform**: Windows 10 (more stable than WSL)
- **Time per epoch**: 40-50 minutes
- **Total training time**: ~7 hours
- **Memory usage**: 12-16 GB

---

## Evaluation

### Run Evaluation

```cmd
python evaluate.py
```

Evaluates on test set (~1,000 images) and generates:

1. **evaluation_report.txt** - Detailed text report
2. **evaluation_report.json** - Machine-readable metrics
3. **confusion_matrix.png** - Confusion matrix visualization
4. **per_class_metrics.png** - Bar chart of precision/recall/F1
5. **class_distribution.png** - Test set distribution

### Expected Metrics

- **Overall Accuracy**: 84-86%
- **Precision**: 0.82-0.88 per class
- **Recall**: 0.80-0.87 per class
- **F1-Score**: 0.81-0.87 per class

---

## Making Predictions

### Command Line

```cmd
python predict.py path/to/image.jpeg --model models/simple_oct_final.h5
```

**Example Output:**
```json
{
  "predicted_class": "DME",
  "confidence": 89.62,
  "full_name": "Diabetic Macular Edema",
  "probabilities": {
    "CNV": 0.1,
    "DME": 89.62,
    "DRUSEN": 0.29,
    "NORMAL": 10.0
  },
  "inference_time_ms": 1338.98
}
```

---

## API Server

### Start Server

```cmd
python api.py
```

Server runs at: **http://localhost:8000**
Interactive API docs: **http://localhost:8000/docs**

### API Endpoints

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-03T14:40:00"
}
```

#### `POST /predict`
Upload image for classification

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "DME",
  "confidence": 89.62,
  "probabilities": {
    "CNV": 0.1,
    "DME": 89.62,
    "DRUSEN": 0.29,
    "NORMAL": 10.0
  },
  "inference_time_ms": 245.67
}
```

### Frontend Integration

#### JavaScript/React

```javascript
async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();

  console.log('Disease:', result.predicted_class);
  console.log('Confidence:', result.confidence + '%');
  console.log('All probabilities:', result.probabilities);
}

// Usage in React component
<input
  type="file"
  onChange={(e) => classifyImage(e.target.files[0])}
/>
```

#### Python

```python
import requests

with open('oct_scan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}%")
```

#### HTML Form

```html
<form id="uploadForm">
  <input type="file" id="imageInput" accept="image/*">
  <button type="submit">Classify OCT Image</button>
</form>

<div id="result"></div>

<script>
document.getElementById('uploadForm').onsubmit = async (e) => {
  e.preventDefault();

  const formData = new FormData();
  const file = document.getElementById('imageInput').files[0];
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();

  document.getElementById('result').innerHTML = `
    <h3>Prediction: ${result.predicted_class}</h3>
    <p>Confidence: ${result.confidence}%</p>
    <h4>All Probabilities:</h4>
    <ul>
      ${Object.entries(result.probabilities).map(([cls, prob]) =>
        `<li>${cls}: ${prob}%</li>`
      ).join('')}
    </ul>
  `;
};
</script>
```

### Testing API

```cmd
pip install requests
python test_api.py
```

Or use interactive docs: **http://localhost:8000/docs** (Swagger UI with "Try it out" feature)

---

## Performance

### Training Performance

- **CPU**: Xeon E5-2680 v4 (14 cores, 28 threads)
- **Time per epoch**: 40-50 minutes (8,349 steps)
- **Total training time**: 6-8 hours (10 epochs)
- **Memory usage**: 12-16 GB RAM (stable)

### Inference Performance

- **Single image**: 200-300ms
- **Batch of 10**: ~2-3 seconds
- **Model size**: 14 MB
- **Platform**: CPU (no GPU required)

---

## Requirements

### Install All Dependencies

```cmd
pip install tensorflow==2.20.0 numpy==1.26.2 pillow==10.1.0 matplotlib==3.8.2 scikit-learn==1.3.2 seaborn==0.13.0 fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6 jupyter requests
```

Or use requirements files:
```cmd
pip install -r requirements.txt
pip install -r requirements-api.txt
```

---

## Troubleshooting

### Training Issues

**Problem**: Kernel crashes during training
**Solution**: Memory issue. Batch size already reduced to 8. Close other programs or run `train_simple.py` instead of notebook.

**Problem**: Training too slow
**Solution**: Expected on CPU. ~40-50 min/epoch. Run overnight. Total: 6-8 hours.

**Problem**: WSL crashes
**Solution**: Run directly on Windows (more stable for long training).

### API Issues

**Problem**: Model not found
**Solution**: Ensure `models/simple_oct_final.h5` exists. Train model first.

**Problem**: Port 8000 already in use
**Solution**: Change port in `api.py` line 166:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Problem**: CORS errors from frontend
**Solution**: API allows all origins by default. Check browser console for specific error.

### Prediction Issues

**Problem**: FileNotFoundError
**Solution**: Use forward slashes in Windows paths:
```python
path = "C:/Users/illia/Desktop/AI/AI-OCT/data/..."  # Correct
path = r"C:\Users\illia\Desktop\AI\AI-OCT\data\..."  # Also correct
```

**Problem**: Low confidence predictions
**Solution**: Normal for difficult cases. Model has 85% accuracy, not 100%.

---

## Project Evolution

### What Worked ✅

1. **Simple frozen transfer learning** - 85% accuracy without fine-tuning
2. **MobileNetV2** - Lightweight, reliable, medical imaging proven
3. **Small batch size (8)** - Prevents memory crashes
4. **No dataset caching** - Prevents memory leaks
5. **Memory cleanup callbacks** - Stable training
6. **Windows native** - More stable than WSL for long training

### What Didn't Work ❌

1. **EfficientNetB0 with unfrozen layers** - Accuracy dropped to 24%
2. **Large batch sizes (32)** - Memory crashes at step ~1100
3. **Dataset caching** - Memory leak after ~1000 steps
4. **WSL training** - System crashes under sustained CPU load
5. **Complex fine-tuning** - Destroyed pre-trained weights

### Key Learnings

- **Simple beats complex** - Frozen transfer learning > complex fine-tuning
- **Memory management is critical** - Small batches, no cache, cleanup
- **Platform matters** - Windows > WSL for stability
- **CPU training works** - Just slower, but reliable

---

## Files Generated

### During Training
- `models/simple_oct_model.h5` - Best checkpoint (based on val_accuracy)
- `models/simple_oct_final.h5` - Final trained model
- `models/training_log.csv` - Epoch-by-epoch metrics
- `models/training_curves.png` - Accuracy/loss plots

### During Evaluation
- `models/evaluation_report.txt` - Detailed text metrics
- `models/evaluation_report.json` - JSON format
- `models/confusion_matrix.png` - Confusion matrix heatmap
- `models/per_class_metrics.png` - Precision/Recall/F1 bars
- `models/class_distribution.png` - Dataset distribution

---

## For Your Class Report

### Key Points to Include

1. **Problem**: Classify OCT retinal images into 4 disease categories
2. **Dataset**: 84,495 images from Kermany et al. 2018
3. **Approach**: Transfer learning with MobileNetV2 (all frozen)
4. **Results**: 85.13% validation accuracy
5. **Architecture**: Simple and reliable - frozen base + classifier head
6. **Training**: 10 epochs, ~7 hours on CPU
7. **Deployment**: REST API for frontend integration

### Visualizations to Include

- Training curves (`models/training_curves.png`)
- Confusion matrix (`models/confusion_matrix.png`)
- Per-class metrics (`models/per_class_metrics.png`)
- Example predictions from each class

### Metrics to Report

- Overall accuracy: 85.13%
- Per-class precision, recall, F1-scores (from evaluation)
- Training time: ~7 hours
- Inference time: ~250ms per image

---

## Technologies Used

- **TensorFlow 2.20**: Deep learning framework
- **MobileNetV2**: Pre-trained CNN architecture
- **FastAPI**: REST API framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Evaluation metrics
- **Jupyter**: Interactive development

---

## Credits

**Dataset**: Kermany et al. 2018 - Labeled Optical Coherence Tomography (OCT) Images
**Model**: MobileNetV2 (Sandler et al. 2018)
**Framework**: TensorFlow 2.20
**Platform**: Windows 10, Python 3.10

---

## License

Educational project for AI class. Dataset used under academic license.

---

## Next Steps (Optional Enhancements)

1. **Test set evaluation** - Full metrics on independent test set
2. **Fine-tuning** - Unfreeze last few layers for potential improvement
3. **Ensemble** - Combine multiple models for better accuracy
4. **Frontend UI** - Build web interface for the API
5. **Docker deployment** - Containerize for easy deployment
6. **Class activation maps** - Visualize what model focuses on

---

**Ready to start?** Follow the Quick Start section above! 🚀

**Questions?** Check the Troubleshooting section or API docs at http://localhost:8000/docs
