# Quick Start Guide - OCT Classification

## ⚡ Your System: Intel Xeon E5-2680 v4 + AMD RX 5700 XT + WSL

**Perfect for**: CPU-based training with your powerful Xeon processor!

## 🚀 Setup (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
cd AIOCT
./setup.sh
```

This will:
- Create virtual environment
- Install optimized TensorFlow
- Set up directories
- Guide you through dataset download

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Intel-optimized TensorFlow (FASTEST)
pip install intel-tensorflow matplotlib seaborn scikit-learn pillow numpy pandas jupyter

# 3. Download dataset
# Visit: https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data
# Extract to: data/kermany2018/
```

## 📥 Dataset Download

### Method 1: Kaggle Web Interface
1. Go to https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data
2. Click "Download" (requires Kaggle account)
3. Extract to `data/kermany2018/`

### Method 2: Kaggle API (Faster)
```bash
pip install kaggle

# Put your kaggle.json in ~/.kaggle/
# Get it from: https://www.kaggle.com/settings/account

kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/kermany2018/
```

## 🎯 Running the Training

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Launch Jupyter
jupyter notebook notebooks/oct_classification.ipynb

# 3. Run cells in order:
# - Cell 1: Imports
# - Cell 1.5: CPU OPTIMIZATION (MUST RUN!)
# - Cell 2+: Continue sequentially
```

## ⚙️ Optimizations for Your System

### In Notebook Cell 1.5 (CPU Optimization):
```python
CPU_THREADS = 28  # Your Xeon has 28 threads
```

### In Cell 2 (Configuration):
```python
BATCH_SIZE = 64   # Change from 32 to 64 (you have 64GB RAM!)
EPOCHS = 25       # Reduce from 30 to 25 for faster training
```

## ⏱️ Expected Performance

| Metric | Your System (Xeon E5-2680 v4) |
|--------|------------------------------|
| Training time | 2-4 hours |
| Accuracy | 92-96% |
| Memory usage | 8-12 GB |
| CPU usage | 90-100% (all 28 threads) |

## 🔍 Monitoring Training

### Check CPU Usage
```bash
# In another terminal
htop
# Should show 28 cores at 90-100%
```

### Check Memory
```bash
free -h
# Used should be 8-12 GB during training
```

### Training Progress
The notebook will show:
- Epoch 1/25 progress bars
- Loss and accuracy metrics
- ~5-10 minutes per epoch

## ✅ Verification Checklist

Before starting training:
- [ ] Virtual environment activated
- [ ] Intel TensorFlow installed (`pip list | grep tensorflow`)
- [ ] Dataset in `data/kermany2018/OCT2017/`
- [ ] Cell 1.5 (CPU optimization) executed
- [ ] BATCH_SIZE changed to 64
- [ ] 28 threads configured

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
source venv/bin/activate
pip install intel-tensorflow
```

### "Cannot find dataset"
Check path in Cell 2:
```python
BASE_DIR = '../data/kermany2018/OCT2017'  # Adjust if needed
```

### "Training very slow"
1. Verify Cell 1.5 was run (should show "CPU OPTIMIZATION ENABLED")
2. Check `htop` shows 28 cores active
3. Increase BATCH_SIZE to 64 or 128

### "Out of Memory"
Reduce BATCH_SIZE:
```python
BATCH_SIZE = 32  # or even 16
```

## 📊 Results Interpretation

After training completes, you'll see:
- **Training curves**: Shows learning progress
- **Confusion matrix**: Which classes are confused
- **Classification report**: Per-class accuracy
- **Test accuracy**: ~92-96% expected

Files saved in `models/`:
- `best_oct_model.h5` - Best model during training
- `final_oct_model.h5` - Final model
- `training_history.csv` - Metrics per epoch
- `model_config.json` - Configuration

## 🎓 For Your Report

Key points to include:
1. **Architecture**: EfficientNetB0 with transfer learning
2. **Dataset**: 84,495 OCT images, 4 classes
3. **Training**: 25 epochs, batch size 64
4. **Optimization**: CPU-optimized for Intel Xeon
5. **Results**: Test accuracy, confusion matrix
6. **Visualizations**: Include plots from notebook

## 📚 Advanced Topics (After Basic Training)

Once basic training works:
1. **Fine-tuning**: Unfreeze some EfficientNet layers
2. **Data augmentation**: Experiment with different augmentations
3. **Ensemble**: Train multiple models, average predictions
4. **Deployment**: Create Flask API for predictions

## 💡 Tips

1. **First run**: Use 5 epochs to verify everything works
   ```python
   EPOCHS = 5  # Quick test
   ```

2. **Resume training**: Load saved model and continue
   ```python
   model = keras.models.load_model('models/best_oct_model.h5')
   ```

3. **Monitor via SSH**: You can close laptop, training continues

4. **Save checkpoints**: Models are saved automatically

## 📞 Need Help?

1. Check `docs/AMD_GPU_SETUP.md` for detailed CPU optimization
2. Check `docs/PROJECT_SETUP.md` for full documentation
3. Verify all cells run without errors sequentially

---

**Ready?** Run `./setup.sh` and start training! 🚀

Estimated total time: **Setup (5 min) + Training (2-4 hours) = ~2.5-4 hours**
