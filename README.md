# AI-OCT: Retinal Disease Classification using Optical Coherence Tomography

An end-to-end machine learning application for automated retinal disease classification from OCT images. The system achieves **95.66% accuracy** classifying retinal conditions using a fine-tuned MobileNetV2 model, served via a FastAPI backend with a React frontend.

## Disease Categories

The model classifies OCT images into four categories:

| Category | Description |
|----------|-------------|
| **CNV** | Choroidal Neovascularization |
| **DME** | Diabetic Macular Edema |
| **DRUSEN** | Age-related macular degeneration indicator |
| **NORMAL** | Healthy retinas |

## Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| CNV | 0.90 | 0.99 | 0.94 |
| DME | 0.99 | 0.94 | 0.97 |
| DRUSEN | 0.99 | 0.90 | 0.94 |
| NORMAL | 0.96 | 1.00 | 0.98 |
| **Overall** | | | **95.66%** |

## Project Structure

```
AI-OCT/
├── api.py                      # FastAPI backend server
├── train_class_balance.py      # Model training script
├── evaluate.py                 # Model evaluation and metrics
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Backend container
├── docker-compose.yml          # Multi-container orchestration
│
├── models/
│   ├── simple_oct_final_balanced.h5    # Trained model
│   ├── confusion_matrix.png            # Evaluation visualization
│   ├── per_class_metrics.png           # Performance metrics
│   └── evaluation_report.txt           # Detailed results
│
├── data/
│   └── kermany2018/
│       └── OCT2017/
│           ├── train/          # Training images
│           ├── test/           # Test images
│           └── val/            # Validation images
│
└── Frontend-ai-projectv2/
    └── ai-webpage/
        ├── src/
        │   ├── App.tsx         # Main React application
        │   ├── UploadImages.tsx # Image upload component
        │   └── ResultsBox.tsx  # Results display
        ├── Dockerfile          # Frontend container
        ├── nginx.conf          # Nginx configuration
        └── package.json        # Node dependencies
```

## Quick Start

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up --build

# Access the application:
# - Frontend: http://localhost:3003
# - Backend API: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
```

### Local Development

#### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python api.py
```

The API will be available at `http://localhost:8000`

#### Frontend Setup

```bash
cd Frontend-ai-projectv2/ai-webpage

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will open at `http://localhost:3000`

## Training the Model

To train the model from scratch:

```bash
# Ensure dataset is in data/kermany2018/OCT2017/
python train_class_balance.py
```

Training configuration:
- **Base Model**: MobileNetV2 (ImageNet pretrained)
- **Fine-tuned Layers**: Last 20 layers
- **Optimizer**: Adam (lr=0.0001)
- **Batch Size**: 16
- **Epochs**: 20
- **Class Balancing**: 8,000 samples per class max

## Evaluating the Model

```bash
python evaluate.py
```

This generates:
- Confusion matrix visualizations
- Per-class precision, recall, F1-score
- Classification report
- Evaluation metrics in JSON format

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status and info |
| `/health` | GET | Health check |
| `/predict` | POST | Classify an OCT image |
| `/docs` | GET | Interactive API documentation |

### Prediction Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/oct_image.jpeg"
```

Response:
```json
{
  "success": true,
  "predicted_class": "NORMAL",
  "confidence": 99.59,
  "probabilities": {
    "CNV": 0.0,
    "DME": 0.0,
    "DRUSEN": 0.41,
    "NORMAL": 99.59
  },
  "inference_time_ms": 245.32
}
```

## Technical Details

### Model Architecture

- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 224 x 224 x 3 (RGB)
- **Output**: 4-class softmax probabilities
- **Regularization**: Dropout (0.2)
- **Model Size**: ~19 MB

### Data Augmentation

Applied during training:
- Random horizontal flips
- Random rotations (±10%)
- Random zoom (±10%)

### Tech Stack

**Backend:**
- Python 3.10
- TensorFlow 2.20.0
- FastAPI 0.104.1
- OpenCV 4.8.0
- NumPy, Pandas, Scikit-learn

**Frontend:**
- React 18.2.0
- TypeScript 4.9.5
- Material-UI 5.14.19
- React Router DOM 6.20.1

**Infrastructure:**
- Docker & Docker Compose
- Nginx (frontend reverse proxy)

## Dataset

This project uses the [Kermany2018 OCT dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) containing labeled OCT images for retinal disease classification.

## License

This project is for educational and research purposes.

## Acknowledgments

- Kermany et al. for the OCT2017 dataset
- TensorFlow team for MobileNetV2 implementation
