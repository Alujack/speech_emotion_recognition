# 📦 Installation Guide

## Choose Your Installation Type:

### Option 1: Full Installation (For Training) ✅ RECOMMENDED

This installs everything you need to train your own model:

```bash
cd /opt/school-project/Speech_Emotion_Recognition/speech_emotion_recognition
source venv/bin/activate
pip install -r requirements.txt
```

**What you get:**
- ✅ FastAPI backend
- ✅ PyTorch & Transformers (AI)
- ✅ Training capabilities
- ✅ Dataset tools
- ✅ Everything needed!

**Size**: ~2GB
**Time**: 5-10 minutes

---

### Option 2: Alternative - Install from requirements-full.txt

For extra features (dataset downloading, etc.):

```bash
pip install -r requirements-full.txt
```

**What's extra:**
- Dataset downloaders
- Google Drive support
- Additional utilities

---

### Option 3: Minimal (Static Demo Only)

If you just want the backend without AI (static/random results):

```bash
pip install -r requirements-minimal.txt
```

**What you get:**
- ✅ FastAPI backend only
- ❌ No AI model
- ❌ No training

**Size**: ~100MB
**Time**: 1-2 minutes

---

## ✅ Verify Installation

After installing, test it:

```bash
# Test imports
python -c "import torch; import transformers; print('✅ All good!')"

# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check GPU (optional)
python -c "import torch; print('GPU available!' if torch.cuda.is_available() else 'CPU mode (OK!)')"
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Out of space"

**Solution**: ML libraries are big (~2GB). Free up space or use minimal installation.

### Issue: Python 3.14 issues

**Solution**: Some packages don't support Python 3.14 yet. The current requirements work fine!

### Issue: Installation too slow

**Solution**: Use a faster mirror:
```bash
pip install -r requirements.txt -i https://pypi.org/simple
```

---

## 📋 What Gets Installed

### Core ML Libraries (~1.5GB):
- **torch**: Deep learning framework
- **transformers**: Wav2Vec2 model
- **numpy**: Numerical computing
- **scipy**: Scientific computing

### Training Libraries (~200MB):
- **datasets**: HuggingFace datasets
- **evaluate**: Evaluation metrics
- **scikit-learn**: ML utilities
- **pandas**: Data processing

### Backend (~100MB):
- **fastapi**: Web framework
- **uvicorn**: Web server

**Total**: ~2GB

---

## 🚀 Quick Commands

### Fresh install:
```bash
cd /opt/school-project/Speech_Emotion_Recognition/speech_emotion_recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Update dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Check what's installed:
```bash
pip list
```

### Uninstall everything:
```bash
pip freeze | xargs pip uninstall -y
```

---

## ✅ You're Ready When You See:

```bash
$ python -c "import torch; import transformers; print('✅ Ready!')"
✅ Ready!
```

**Then you can:**
- ✅ Run the backend
- ✅ Use pre-trained models
- ✅ Train your own models
- ✅ Deploy to production

---

## 💡 Recommended: Full Installation

For your project, use the **full installation** (`requirements.txt`):

```bash
pip install -r requirements.txt
```

This gives you everything you need! 🎉

