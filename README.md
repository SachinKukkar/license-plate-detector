# 🚗 License Plate Detection & YOLO Post-Filtering

This project consists of two classical computer vision tasks:

## 📌 Task 1: License Plate Detection (Without Deep Learning)

Detect license plate-like regions from a vehicle image using **pure image processing techniques**. No pretrained models (like YOLO, Haar cascades, or OpenCV classifiers) are used.

### ✅ Features:
- Grayscale conversion
- Noise reduction with bilateral filtering
- Edge detection with Canny
- Contour detection and filtering based on:
  - Shape (rectangular)
  - High-contrast edges
  - Aspect ratio between 2:1 and 5:1
- Bounding box visualization
- Output saved as `output.jpg`

### 🔧 Constraints:
- No pretrained models
- Only uses standard libraries: `OpenCV`, `NumPy`
- Modular, readable Python code

### 📂 Input:
- `input.jpg` (rename your target vehicle image to this)

### 📂 Output:
- `output.jpg` (with bounding boxes)

### 🔗 Run:
```bash
python anpr.py
