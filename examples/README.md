# OCT Classification - Usage Examples

This directory contains example scripts demonstrating different ways to use the trained OCT classification model.

## 📁 Files

| File | Description | Usage |
|------|-------------|-------|
| `simple_usage.py` | Basic prediction example | Single image with detailed output |
| `batch_processing.py` | Process multiple images | Generate CSV report |
| `flask_api.py` | REST API server | Web/mobile app integration |

## 🚀 Quick Start

### 1. Simple Usage

Predict a single image with detailed output:

```bash
python examples/simple_usage.py
```

**Output:**
```
============================================================
OCT Classification - Simple Usage Example
============================================================

1. Loading model...
Loading model from: models/best_oct_model.h5
✓ Model loaded successfully

2. Making prediction...

============================================================
PREDICTION RESULTS
============================================================
Image:      data/kermany2018/OCT2017/test/NORMAL/NORMAL-1000-1.jpeg
Class:      NORMAL
Condition:  Normal (Healthy)
Confidence: 96.82%
Time:       45.23ms

Probability Distribution:
  CNV       1.2% ▌
  DME       0.8% ▍
  DRUSEN    1.2% ▌
  NORMAL   96.8% ████████████████████████████████████████████████

============================================================
```

### 2. Batch Processing

Process entire directory of images:

```bash
python examples/batch_processing.py data/kermany2018/OCT2017/test/
```

**Output:**
```
============================================================
OCT Classification - Batch Processing
============================================================

Loading model...
✓ Model loaded successfully

Scanning directory: data/kermany2018/OCT2017/test/
Found 1000 images

[  1/1000] ✓ CNV-1000-1.jpeg                      CNV (95.2%)
[  2/1000] ✓ CNV-1000-2.jpeg                      CNV (97.1%)
[  3/1000] ✓ DME-1000-1.jpeg                      DME (93.5%)
...

✓ Results saved to: results.csv

============================================================
SUMMARY
============================================================
Total processed: 1000

Class Distribution:
  CNV     : 250 (25.0%)
  DME     : 250 (25.0%)
  DRUSEN  : 250 (25.0%)
  NORMAL  : 250 (25.0%)

Average confidence: 94.23%
Average inference time: 46.12ms
```

**CSV Output (`results.csv`):**
```csv
filename,path,predicted_class,confidence,full_name,inference_time_ms
CNV-1000-1.jpeg,/path/to/CNV-1000-1.jpeg,CNV,95.2,Choroidal Neovascularization,45.3
DME-1000-1.jpeg,/path/to/DME-1000-1.jpeg,DME,93.5,Diabetic Macular Edema,46.1
...
```

### 3. REST API

Start Flask API server:

```bash
pip install flask  # Install if not already installed
python examples/flask_api.py
```

**Server Output:**
```
============================================================
OCT Classification REST API
============================================================

API running at: http://localhost:5000

Endpoints:
  GET  /           - API information
  GET  /health     - Health check
  POST /predict    - Predict image

Test with:
  curl -X POST -F "file=@scan.jpg" http://localhost:5000/predict

============================================================

 * Running on http://0.0.0.0:5000
```

**Test the API:**

```bash
# Test prediction
curl -X POST -F "file=@scan.jpg" http://localhost:5000/predict

# Response:
{
  "success": true,
  "predicted_class": "DME",
  "confidence": 92.5,
  "full_name": "Diabetic Macular Edema",
  "inference_time_ms": 45.3,
  "probabilities": {
    "CNV": 2.1,
    "DME": 92.5,
    "DRUSEN": 1.8,
    "NORMAL": 3.6
  },
  "filename": "scan.jpg"
}
```

**Other endpoints:**

```bash
# API info
curl http://localhost:5000/

# Health check
curl http://localhost:5000/health
```

## 🔧 Customization

### Modify Simple Usage

Edit `simple_usage.py` to change the image:

```python
# Line 15
image_path = 'path/to/your/image.jpg'
```

### Batch Processing Options

```bash
# Process specific directory
python examples/batch_processing.py data/my_scans/

# Custom output file
python examples/batch_processing.py data/my_scans/ my_results.csv
```

### API Configuration

Edit `flask_api.py` to change settings:

```python
# Line 120 - Change host/port
app.run(host='0.0.0.0', port=8080, debug=False)
```

## 📚 Integration Tips

### Use in Your Python App

```python
# Import the predictor
from oct_predictor import OCTPredictor

# In your application
class MedicalApp:
    def __init__(self):
        self.oct_predictor = OCTPredictor('models/best_oct_model.h5')

    def analyze_scan(self, image_path):
        result = self.oct_predictor.predict(image_path)

        if result['predicted_class'] != 'NORMAL':
            self.alert_doctor(result)

        return result
```

### JavaScript/Web Integration

```javascript
// Upload image to Flask API
async function analyzeOCT(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (result.success) {
        displayResult(result);
    } else {
        showError(result.error);
    }
}
```

### Mobile App Integration

```javascript
// React Native example
const analyzeImage = async (imageUri) => {
    const formData = new FormData();
    formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'scan.jpg'
    });

    try {
        const response = await fetch('http://your-server.com/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Analysis failed:', error);
    }
};
```

## 🐛 Troubleshooting

### Model Not Found

```
Error: Model file not found: models/best_oct_model.h5
```

**Solution:** Train the model first:
```bash
jupyter notebook notebooks/oct_classification.ipynb
# Run all cells to train and save model
```

### Flask Not Installed

```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```bash
source venv/bin/activate
pip install flask
```

### Permission Denied

```
PermissionError: [Errno 13] Permission denied: 'results.csv'
```

**Solution:**
```bash
# Check file permissions
ls -l results.csv

# Make writable
chmod 644 results.csv
```

## 📊 Performance Benchmarks

Based on Intel Xeon E5-2680 v4:

| Operation | Time | Notes |
|-----------|------|-------|
| Single prediction | 45ms | Average |
| Batch (100 images) | 4.5s | Sequential |
| API request | 50ms | Including I/O |
| Model loading | 2s | One-time cost |

## 📝 Next Steps

1. **Deploy to production**: Use Gunicorn/uWSGI for Flask API
2. **Add authentication**: Secure your API endpoints
3. **Database integration**: Store predictions in database
4. **Visualization**: Create dashboard for results
5. **Monitoring**: Add logging and metrics

## 🔗 Related Documentation

- **Main CLI**: `predict.py` - Command-line prediction
- **Module**: `oct_predictor.py` - Python library
- **Guide**: `docs/INFERENCE_GUIDE.md` - Complete usage guide
- **Training**: `notebooks/oct_classification.ipynb` - Model training

---

**Need help?** Check `docs/INFERENCE_GUIDE.md` for detailed documentation!
