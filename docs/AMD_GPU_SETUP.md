# AMD RX 5700 XT Setup for TensorFlow on WSL

## ⚠️ Important: AMD GPU Considerations

Your AMD RX 5700 XT **cannot use NVIDIA CUDA**. You have 2 options:

### **Option 1: CPU-Only TensorFlow (RECOMMENDED for Simplicity)**
- ✅ Easiest setup (5 minutes)
- ✅ Works immediately, no driver issues
- ✅ Your 14-core Xeon E5-2680 v4 is quite capable
- ⏱️ Training time: ~2-4 hours (acceptable for one-time training)

### **Option 2: AMD ROCm (Complex, WSL2 Only)**
- ⚠️ Complex setup (1-2 hours)
- ⚠️ ROCm support for RX 5700 XT is limited
- ⚠️ May have compatibility issues in WSL
- ⚡ Faster training (~30-60 min) IF it works

## 🎯 Recommended Approach: CPU-Optimized TensorFlow

Given your strong CPU (14 cores, 28 threads) and 64GB RAM, CPU training will work well!

### Step 1: Install Optimized TensorFlow

```bash
# Uninstall any existing TensorFlow
pip uninstall tensorflow tensorflow-gpu

# Install CPU-optimized TensorFlow
pip install tensorflow-cpu

# OR install Intel-optimized TensorFlow (MUCH faster on Intel CPUs)
pip install intel-tensorflow

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### Step 2: Enable CPU Optimizations

Add this to your notebook (Cell 2, after imports):

```python
import os

# Optimize for multi-core CPU
os.environ['OMP_NUM_THREADS'] = '28'  # Your CPU threads
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '28'

# Enable Intel MKL optimizations (if using Intel TensorFlow)
os.environ['KMP_BLOCKTIME'] = '1'
os.environ['KMP_SETTINGS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'

# Configure TensorFlow for CPU
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(28)

print(f"CPU cores: {os.cpu_count()}")
print(f"TensorFlow will use {28} threads")
```

### Step 3: Optimize Training Parameters

**Update these in Cell 3:**

```python
# CPU-optimized settings
IMG_SIZE = 224
BATCH_SIZE = 64  # Increase for CPU (you have 64GB RAM!)
EPOCHS = 25      # Slightly fewer epochs
NUM_CLASSES = 4

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision enabled: {policy.name}")
```

### Expected Performance with CPU:

| Task | Time (Xeon E5-2680 v4) |
|------|------------------------|
| Data loading | ~5 min |
| Training (25 epochs) | ~2-4 hours |
| Evaluation | ~10 min |
| **Total** | **~2.5-4.5 hours** |

**This is perfectly acceptable for a class project!**

---

## 🔥 Option 2: AMD ROCm (Advanced Users)

⚠️ **Warning**: This is complex and may not work reliably in WSL.

### ROCm Compatibility Check

RX 5700 XT uses **gfx1010** architecture. Check ROCm support:

```bash
# Check GPU info
lspci | grep -i vga

# Check if ROCm detects GPU
rocm-smi
```

### ROCm Installation (Ubuntu 22.04 on WSL2)

```bash
# Add ROCm repository
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install rocm-dkms rocm-libs

# Add user to video group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Reboot WSL
wsl --shutdown  # Run from Windows PowerShell
```

### Install TensorFlow-ROCm

```bash
# Install TensorFlow for ROCm
pip install tensorflow-rocm

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Troubleshooting ROCm

If GPU not detected:

```bash
# Check ROCm installation
/opt/rocm/bin/rocminfo

# Check TensorFlow-ROCm
python -c "import tensorflow as tf; print(tf.test.is_built_with_rocm())"

# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
```

**Common Issues**:
- RX 5700 XT (gfx1010) may not be officially supported
- WSL2 GPU passthrough for AMD is experimental
- Driver conflicts with Windows

---

## 🎯 My Recommendation for Your Setup

### **Use CPU-Optimized TensorFlow** because:

1. ✅ **Simpler**: 5-minute setup vs 2-hour debugging
2. ✅ **Reliable**: No driver/compatibility issues
3. ✅ **Sufficient**: 2-4 hours training is acceptable
4. ✅ **Your CPU is strong**: 14 cores + 64GB RAM
5. ✅ **Focus on learning**: Not on infrastructure problems

### ROCm is worth it ONLY if:
- You plan to train many models repeatedly
- You have time to debug for 1-2 hours
- You need <1 hour training times

---

## 🚀 Quick Start (CPU-Optimized)

**Complete setup in 3 commands:**

```bash
# 1. Install optimized TensorFlow
pip install intel-tensorflow matplotlib seaborn scikit-learn pillow numpy pandas

# 2. Set CPU threads
export OMP_NUM_THREADS=28

# 3. Run notebook
jupyter notebook notebooks/oct_classification.ipynb
```

**In notebook Cell 2, add:**
```python
import os
os.environ['OMP_NUM_THREADS'] = '28'
os.environ['TF_NUM_INTRAOP_THREADS'] = '28'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(28)
```

**In Cell 3, change:**
```python
BATCH_SIZE = 64  # Increased from 32
```

**Expected results:**
- ⏱️ Training: ~2-4 hours
- 🎯 Accuracy: 92-96% (same as GPU!)
- 💾 Memory usage: ~8-12 GB

---

## 📊 Performance Comparison

| Setup | Training Time | Setup Time | Reliability |
|-------|---------------|------------|-------------|
| **CPU-optimized (Intel TF)** | 2-4 hours | 5 min | ⭐⭐⭐⭐⭐ |
| TensorFlow-CPU (standard) | 4-6 hours | 5 min | ⭐⭐⭐⭐⭐ |
| TensorFlow-ROCm (AMD GPU) | 30-60 min | 1-2 hours | ⭐⭐ |
| NVIDIA GPU (for comparison) | 20-40 min | 30 min | ⭐⭐⭐⭐⭐ |

---

## 🔍 WSL-Specific Tips

### Check WSL Version
```bash
wsl --version  # Run from Windows PowerShell
```

ROCm requires **WSL2**, not WSL1.

### Upgrade to WSL2 (if needed)
```bash
# From Windows PowerShell (as Admin)
wsl --set-version Ubuntu 2
```

### Monitor Resource Usage in WSL

```bash
# Check CPU usage
htop

# Check memory
free -h

# Monitor during training
watch -n 1 "ps aux | grep python | grep -v grep"
```

### Allocate More Resources to WSL

Create/edit `.wslconfig` in Windows user folder (`C:\Users\YourName\.wslconfig`):

```ini
[wsl2]
memory=48GB      # Use 48 of your 64GB
processors=28    # All your CPU threads
swap=16GB        # Swap space
```

Then restart WSL:
```bash
wsl --shutdown
```

---

## 🎓 Summary for Your Project

**Recommended Configuration:**

```bash
# Install
pip install intel-tensorflow matplotlib seaborn scikit-learn pillow numpy pandas jupyter

# Configure (add to notebook)
os.environ['OMP_NUM_THREADS'] = '28'
BATCH_SIZE = 64

# Train
# ~2-4 hours, 92-96% accuracy
```

**Why this works:**
- Your Xeon E5-2680 v4 has 14 cores (28 threads) - excellent for ML
- 64GB RAM - way more than needed
- Intel-optimized TensorFlow uses AVX2/MKL
- Larger batch size = fewer iterations = faster training

**Bottom line**: Don't waste time fighting with ROCm. Your CPU setup will work great! 🚀

---

## 📞 Need Help?

**If training seems slow**, verify:
```python
import tensorflow as tf
print(f"TensorFlow threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
```

Should output: `28`

**Monitor CPU usage during training:**
```bash
htop  # Should show ~28 cores at 90-100%
```

Good luck! Your hardware is more than capable. 💪
