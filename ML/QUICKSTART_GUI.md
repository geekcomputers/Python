# 🚀 NeuralForge GUI - Quick Start

## Launch the GUI

### Windows (Easy Way)
```bash
# Double-click this file:
LAUNCH_GUI.bat

# Or run in terminal:
python tests\gui_test.py
```

### Linux/Mac
```bash
chmod +x LAUNCH_GUI.sh
./LAUNCH_GUI.sh
```

---

## How to Use (3 Steps)

### 1️⃣ Load Your Model

1. Click **"Use Default"** button (loads `models/final_model.pt`)
2. Make sure dataset is set to `cifar10` (or your trained dataset)
3. Click **"Load Model"**
4. Wait for ✓ green checkmark

### 2️⃣ Select an Image

1. Click **"Browse"** under Image Selection
2. Choose any image file (JPG, PNG, etc.)
3. Image preview appears automatically

### 3️⃣ Get Prediction

1. Click **"🔍 Predict"** button
2. See results instantly:
   - **Main prediction** in large green text
   - **Confidence percentage**
   - **Top-5 predictions** with visual bars

---

## 📸 Test with Your Own Images!

**CIFAR-10 Classes:**
- ✈️ airplane
- 🚗 automobile  
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐕 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚛 truck

**Your Results:**
- **Training Accuracy:** 99.98%
- **Validation Accuracy:** 75.81%
- **Model:** ResNet18 (11.2M parameters)

---

## 🎨 GUI Features

### Beautiful Dark Theme
- Professional dark background
- Green accent colors
- Smooth animations
- Easy-to-read fonts

### Real-Time Feedback
- Loading indicators
- Progress bars
- Status messages
- Error handling

### Smart Interface
- Image preview
- Model information display
- Top-5 predictions with bars
- Confidence percentages

---

## 💡 Pro Tips

1. **Load Once, Test Many:** Load model once, test unlimited images
2. **Quick Testing:** Use default button for instant model loading
3. **Best Results:** Use clear, centered images
4. **Fast Predictions:** First prediction initializes, then super fast!
5. **Check Info:** Model info shows parameters and accuracy

---

## 🐛 Troubleshooting

**"No module named 'PyQt6'"**
```bash
pip install PyQt6
```

**"Model file not found"**
- Train a model first: `python train.py --dataset cifar10 --epochs 50`
- Or check `models/` folder exists

**GUI won't start**
```bash
pip install --upgrade PyQt6
```

**Prediction errors**
- Ensure dataset name matches training dataset
- Check image file is valid
- Verify model loaded successfully (green checkmark)

---

## 📊 What You Can Test

### Example Images to Try:

**For CIFAR-10:**
- Photos of cats, dogs, horses
- Pictures of cars, trucks, airplanes
- Images of ships, frogs, birds
- Nature scenes with deer

**Tips:**
- Use clear, single-object images
- Centered subjects work best
- Good lighting improves accuracy
- Any image size works (auto-resized)

---

## 🎯 Expected Results

Based on your training:
- **High confidence (>90%):** Clear images of trained classes
- **Medium confidence (50-90%):** Partial views or similar classes
- **Low confidence (<50%):** Unclear or out-of-distribution images

---

## 🚀 Next Steps

1. **Test Different Images:** Try various images from each class
2. **Check Accuracy:** Compare predictions with actual labels
3. **Train More:** Improve model with more epochs for better accuracy
4. **Try Other Datasets:** Load models trained on different datasets

---

## 📝 Example Session

```
1. Start GUI
2. Click "Use Default"
3. Click "Load Model"
   ✓ Model loaded successfully
4. Click "Browse" → Select cat.jpg
5. Click "🔍 Predict"
   
Results:
   🎯 cat
   Confidence: 94.3%
   
   Top-5:
   1. cat     ████████████████ 94.3%
   2. dog     ██ 3.2%
   3. deer    █ 1.5%
   4. bird    █ 0.7%
   5. frog    ░ 0.3%
```

---

## 🎉 Enjoy Testing Your AI!

Your model achieved **75.81% validation accuracy** - test it on real images and see how it performs!

**Questions or Issues?**
- Check `tests/README_GUI.md` for detailed documentation
- Verify model file exists in `models/` folder
- Ensure PyQt6 is installed: `pip list | grep PyQt6`

---

**Made with 🔥 by NeuralForge**
