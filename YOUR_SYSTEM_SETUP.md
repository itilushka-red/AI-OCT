# Your System-Specific Setup Guide

## 🖥️ Your Hardware Configuration

```
CPU:  Intel Xeon E5-2680 v4
      - 14 cores, 28 threads
      - Excellent for CPU-based training

GPU:  AMD RX 5700 XT
      - Cannot use CUDA (NVIDIA only)
      - ROCm support limited in WSL
      - Recommendation: Use CPU training

RAM:  64 GB
      - Perfect for large batch sizes
      - Can handle full dataset in memory

OS:   Windows + WSL2
      - Good for development
      - Ensure WSL2, not WSL1
```

## 🎯 Recommended Configuration

### ✅ Use CPU Training (Simplest & Most Reliable)

**Why CPU instead of GPU?**
1. AMD RX 5700 XT doesn't support CUDA
2. ROCm for AMD is complex to set up in WSL
3. Your Xeon CPU is powerful (28 threads)
4. Training time is acceptable: 2-4 hours

### Installation Command

```bash
# Install Intel-optimized TensorFlow (FASTEST for Xeon)
pip install intel-tensorflow matplotlib seaborn scikit-learn pillow numpy pandas jupyter
```

**NOT** `tensorflow-gpu` or `tensorflow-rocm` - those won't work well with your setup!

## ⚙️ Notebook Configuration

### Cell 1.5: CPU Optimization (CRITICAL!)

```python
# This cell is ALREADY in your notebook
CPU_THREADS = 28  # Perfect for your Xeon E5-2680 v4

# These settings make TensorFlow use all your CPU cores
os.environ['OMP_NUM_THREADS'] = '28'
os.environ['TF_NUM_INTRAOP_THREADS'] = '28'
```

**Must run this cell before training!**

### Cell 2: Configuration Updates

Change these values from defaults:

```python
# BEFORE (default):
BATCH_SIZE = 32
EPOCHS = 30

# AFTER (optimized for your 64GB RAM):
BATCH_SIZE = 64   # 2x larger - you have the RAM!
EPOCHS = 25       # Slightly fewer, still great results
```

## 📊 Performance Expectations

### Training Timeline (Xeon E5-2680 v4)

| Task | Time | CPU Usage |
|------|------|-----------|
| Data loading | 5 min | 50% |
| Epoch 1-10 | 50-70 min | 95% |
| Epoch 11-20 | 50-70 min | 95% |
| Epoch 21-25 | 25-35 min | 95% |
| Evaluation | 10 min | 70% |
| **Total** | **2-4 hours** | - |

**Expected Results**:
- ✅ Accuracy: 92-96%
- ✅ Same as GPU training!
- ✅ Model quality identical

## 🚀 Step-by-Step Startup

### 1. First-Time Setup (5 minutes)

```bash
cd /home/aristarx/University/AIOCT
./setup.sh

# When prompted, choose option 1:
# 1) intel-tensorflow (RECOMMENDED)
```

### 2. Download Dataset (10-30 minutes)

```bash
# Activate environment first
source venv/bin/activate

# Option A: Kaggle API (recommended)
pip install kaggle
# Put your kaggle.json in ~/.kaggle/
kaggle datasets download -d paultimothymooney/kermany2018
unzip kermany2018.zip -d data/kermany2018/

# Option B: Manual download
# Visit: https://www.kaggle.com/datasets/paultimothymooney/kermany2018/data
# Download and extract to: data/kermany2018/
```

### 3. Verify Dataset Structure

```bash
ls data/kermany2018/OCT2017/train/

# Should show:
# CNV/  DME/  DRUSEN/  NORMAL/
```

### 4. Start Jupyter

```bash
source venv/bin/activate
jupyter notebook notebooks/oct_classification.ipynb

# Browser will open automatically
# If not: Go to http://localhost:8888
```

### 5. Run Notebook Cells

Execute cells in order:
1. ✅ Cell 1: Imports
2. ⚠️ **Cell 1.5: CPU OPTIMIZATION** (MUST RUN!)
3. ✅ Cell 2: Configuration (verify BATCH_SIZE=64)
4. ✅ Cells 3+: Run sequentially

## 💻 Monitoring Your Training

### In Terminal (different window):

```bash
# Monitor CPU usage
htop

# Should see:
# - 28 cores at 90-100%
# - ~8-12 GB RAM used
# - Python process at top

# Monitor memory
watch -n 5 free -h

# Check TensorFlow is using CPU
ps aux | grep python
```

### In Jupyter:

You'll see:
```
Epoch 1/25
████████████████████ 100/100 [05:23<00:00, 3.2s/step]
loss: 0.8234 - accuracy: 0.6745 - val_loss: 0.5432 - val_accuracy: 0.7890
```

~5-10 minutes per epoch is normal!

## 🔧 Optimization Tips

### Speed Up Training:

1. **Increase batch size** (you have 64GB RAM!)
   ```python
   BATCH_SIZE = 128  # Try even larger!
   ```

2. **Reduce epochs for testing**
   ```python
   EPOCHS = 5  # Quick test run
   ```

3. **Use mixed precision** (already in notebook)
   ```python
   # Automatically enabled in Cell 1.5
   ```

### If Training Seems Slow:

1. Verify CPU optimization cell ran:
   - Should show "CPU OPTIMIZATION ENABLED"

2. Check `htop`:
   - All 28 cores should be at 90-100%

3. Check batch size:
   ```python
   print(f"Batch size: {BATCH_SIZE}")
   # Should show 64, not 32
   ```

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T: Try to use GPU
```bash
# Don't do this:
pip install tensorflow  # Will try CUDA, won't work
pip install tensorflow-rocm  # Complex, unstable in WSL
```

### ❌ DON'T: Skip Cell 1.5
Without CPU optimization:
- Training will be 3-5x slower
- Only use 1-2 cores instead of 28

### ❌ DON'T: Use small batch size
```python
BATCH_SIZE = 16  # Too small for 64GB RAM!
BATCH_SIZE = 64  # Perfect! ✅
```

### ✅ DO: Run setup script
```bash
./setup.sh  # Automates everything
```

### ✅ DO: Activate venv
```bash
source venv/bin/activate  # Before every session
```

### ✅ DO: Monitor progress
```bash
htop  # Watch CPU usage
```

## 📈 What to Expect During Training

### Epoch 1-5: Initial Learning
- Accuracy: 60-75%
- Loss decreasing rapidly
- Each epoch: 5-7 minutes

### Epoch 6-15: Refinement
- Accuracy: 75-90%
- Loss decreasing slowly
- Each epoch: 5-7 minutes

### Epoch 16-25: Fine-tuning
- Accuracy: 90-95%
- Loss plateau
- Early stopping may kick in

### Final Results:
- Test accuracy: 92-96%
- Training time: 2-4 hours total
- Model saved automatically

## 🎓 For Your AI Class Report

### System Specifications Section:
```
Hardware:
- CPU: Intel Xeon E5-2680 v4 (14 cores, 28 threads)
- RAM: 64 GB DDR4
- Storage: SSD
- OS: Windows 10/11 with WSL2 (Ubuntu)

Software:
- Python 3.8+
- Intel-optimized TensorFlow 2.13+
- Keras (integrated with TensorFlow)
- CUDA: N/A (CPU training)

Optimization:
- Multi-threaded training (28 threads)
- Large batch size (64) leveraging 64GB RAM
- Intel MKL optimizations
```

### Training Configuration:
```
Model: EfficientNetB0 (pre-trained on ImageNet)
Input size: 224×224×3
Batch size: 64
Optimizer: Adam (lr=0.0001)
Epochs: 25 (with early stopping)
Data augmentation: Rotation, flip, zoom
Training time: ~2.5 hours
```

## 🆘 Troubleshooting Guide

### Problem: "tensorflow not found"
```bash
source venv/bin/activate
pip install intel-tensorflow
```

### Problem: Training using only 1 core
```bash
# In notebook, verify Cell 1.5 ran:
# Should see: "CPU OPTIMIZATION ENABLED"

# If not, go back and run Cell 1.5
```

### Problem: Out of memory
```python
# Reduce batch size in Cell 2
BATCH_SIZE = 32  # or even 16
```

### Problem: "Cannot find dataset"
```bash
# Check path
ls data/kermany2018/OCT2017/train/

# If not found, re-download dataset
# See "Download Dataset" section above
```

### Problem: Jupyter won't start
```bash
# Reinstall jupyter
pip install --upgrade jupyter

# Or run Python directly
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 📞 Quick Reference

### Activate Environment:
```bash
cd /home/aristarx/University/AIOCT
source venv/bin/activate
```

### Start Training:
```bash
jupyter notebook notebooks/oct_classification.ipynb
```

### Monitor CPU:
```bash
htop
```

### Check GPU (should be empty):
```bash
nvidia-smi  # Will fail - that's correct!
```

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [ ] WSL2 installed (not WSL1)
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Intel-tensorflow installed
- [ ] Dataset downloaded and extracted
- [ ] Dataset in: `data/kermany2018/OCT2017/`
- [ ] Cell 1.5 executed (CPU optimization)
- [ ] BATCH_SIZE set to 64
- [ ] CPU_THREADS set to 28
- [ ] `htop` installed (for monitoring)

## 🎉 Ready to Start!

You're all set! Your system is powerful enough for excellent CPU-based training.

**Next step**: Run `./setup.sh` and follow the prompts!

---

**Remember**: CPU training on your Xeon is perfectly fine!
The model quality will be identical to GPU training, just takes a bit longer (2-4 hours vs 30-60 min).
For a class project, this is totally acceptable! 🚀
