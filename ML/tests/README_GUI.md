# NeuralForge GUI Tester

Beautiful PyQt6 GUI application for testing your trained models!

## Features

✅ **Model Selection**
- Browse for any `.pt` model file
- Quick "Use Default" button for `models/final_model.pt`
- Dataset selector (CIFAR-10, MNIST, etc.)
- Real-time model loading with status

✅ **Image Testing**
- Browse and select any image
- Live image preview (auto-scaled)
- Drag-and-drop style interface

✅ **Predictions**
- Large, clear main prediction display
- Confidence percentage
- Top-5 predictions with visual bars
- Progress indicator during inference

✅ **Modern UI**
- Dark theme (easy on eyes)
- Green accent colors
- Smooth animations
- Professional styling

## Installation

```bash
pip install PyQt6
```

## Usage

### Run the GUI

```bash
python tests/gui_test.py
```

### Steps:

1. **Load Model:**
   - Click "Use Default" for your trained model
   - Or browse to select a `.pt` file
   - Select dataset (e.g., `cifar10`)
   - Click "Load Model"

2. **Select Image:**
   - Click "Browse" to select an image
   - Preview appears automatically

3. **Predict:**
   - Click "🔍 Predict" button
   - Results appear instantly!

## Screenshots

### Main Interface
```
┌──────────────────────────────────────────────────────────┐
│  🚀 NeuralForge Model Tester                             │
├───────────────────────┬──────────────────────────────────┤
│                       │                                  │
│  Model Selection      │   Prediction Results             │
│  ┌──────────────┐     │   ┌────────────────────────┐     │
│  │ [Browse]     │     │   │  🎯 cat               │     │
│  │ [Use Default]│     │   │  Confidence: 94.3%     │    │
│  └──────────────┘     │   └────────────────────────┘    │
│                       │                                 │
│  Image Selection      │   Top-5 Predictions             │
│  ┌──────────────┐     │   ┌────────────────────────┐    │
│  │   [Image]    │     │   │ 1. cat    ████████ 94% │    │
│  │   Preview    │     │   │ 2. dog    ██ 3%        │    │
│  │              │     │   │ 3. deer   █ 1%         │    │
│  └──────────────┘     │   └────────────────────────┘    │
│  [🔍 Predict]         │                                 │
└───────────────────────┴──────────────────────────────────┘
```

## Features Explained

### Model Information Display
Shows:
- Model architecture (ResNet18)
- Dataset name
- Number of classes
- Total parameters
- Training epoch
- Best validation loss
- Device (CPU/CUDA)

### Prediction Display
- **Main Prediction:** Large, bold display
- **Confidence:** Percentage score
- **Top-5:** Visual bar chart with percentages
- **Color-coded:** Green for results, red for errors

## Supported Datasets

- CIFAR-10 (10 classes)
- CIFAR-100 (100 classes)
- MNIST (10 classes)
- Fashion-MNIST (10 classes)
- STL-10 (10 classes)
- Tiny ImageNet (200 classes)
- Food-101 (101 classes)
- Caltech-256 (257 classes)
- Oxford Pets (37 classes)
- ImageNet (1000 classes)

## Tips

1. **Best Image Quality:** Use clear, well-lit images
2. **Image Size:** Any size works (auto-resized to 224x224)
3. **Format:** Supports PNG, JPG, JPEG, BMP, GIF
4. **Multiple Tests:** Load once, test many images
5. **Quick Access:** Keep commonly used models in `models/` folder

## Keyboard Shortcuts

- `Ctrl+O` - Browse model
- `Ctrl+I` - Browse image
- `Ctrl+P` - Predict (when ready)
- `Ctrl+D` - Use default model

## Troubleshooting

**GUI won't start:**
```bash
pip install --upgrade PyQt6
```

**Model not loading:**
- Check file path is correct
- Ensure dataset name matches training dataset
- Verify `.pt` file is not corrupted

**Image not displaying:**
- Check image file format
- Ensure file exists
- Try different image

**Slow predictions:**
- First prediction is slower (model warming up)
- GPU mode is much faster than CPU
- Check CUDA availability in Model Info

## Advanced Usage

### Testing Custom Models

```python
# Your model must be compatible with the interface
# Save with: torch.save({'model_state_dict': model.state_dict()}, 'model.pt')
```

### Batch Testing

Run multiple images sequentially:
1. Load model once
2. Browse and predict for each image
3. Results update in real-time

## Theme Customization

The dark theme uses:
- Background: `#1e1e1e`
- Accent: `#4CAF50` (green)
- Text: `#e0e0e0`
- Borders: `#3d3d3d`

To customize, edit the `apply_stylesheet()` method in `gui_test.py`.

## Performance

- **Loading:** ~1-2 seconds
- **Prediction:** ~0.1-0.5 seconds (GPU)
- **Memory:** ~500MB (model loaded)

## Enjoy Testing! 🚀
