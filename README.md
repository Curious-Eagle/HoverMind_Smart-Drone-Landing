# 🚁 Automated Drone Landing Site Detection

An end-to-end system for identifying safe drone landing zones using Semantic Segmentation. This project leverages Deep Learning to analyze aerial imagery and classify surfaces into 23 distinct categories (grass, gravel, trees, etc.) to determine the safest landing spots in real-time.

---

## 🌟 Key Features

*   **Semantic Segmentation**: Precise pixel-level classification of aerial scenes.
*   **Safety Assessment**: Categorizes surfaces into Safe, Unsafe, and Neutral landing zones.
*   **Multiple Architectures**: Support for U-Net with various backbones (MobileNetV2, EfficientNet-B3/B4).
*   **Optimized Performance**: Lightweight model options suitable for edge deployment on drones.
*   **Comprehensive Visualization**: Interactive results showing original imagery, ground truth, and model predictions.

---

## 📁 Repository Structure

```text
DroneLandingSystem_GitHub/
├── data/               # Sample dataset (6 pairs of images/masks)
│   ├── images/         # Original aerial .jpg files
│   └── masks/          # Semantic segmentation .png labels
├── models/             # Pre-trained model weights
│   └── Unet-Mobilenet.pt
├── notebooks/          # Interactive Jupyter Notebooks
│   └── main_notebook.ipynb
├── src/                # Modular Python scripts
│   ├── train.py        # Core training pipeline
│   └── advanced_train.py # Advanced implementation with optimizations
├── .gitignore          # Files to exclude from Git
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### 2. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage
*   **Interactive**: Open `notebooks/main_notebook.ipynb` to explore the dataset and run predictions interactively.
*   **Training**: Run `python src/train.py` to start the training process with default configurations.
*   **Advanced**: Use `src/advanced_train.py` for features like mixed-precision training and advanced data augmentation.

---

## 📊 Dataset Information

This project uses the **Semantic Drone Dataset** from TU Graz.
*   **Original Source**: [TU Graz Drone Dataset](https://www.tugraz.at/index.php?id=22387)
*   **Classes**: 23 classes including trees, grass, dirt, gravel, rocks, water, paved_area, person, car, etc.
*   **Sample Data**: A small subset of 6 image-mask pairs is included in `data/` for demonstration purposes. To train the full model, please download the complete dataset from the link above.

---

## 🛠 Technical Implementation

### Model Architecture
The primary model is a **U-Net** architecture. We utilize the `segmentation-models-pytorch` library to experiment with different encoders. The default lightweight model uses a **MobileNetV2** backbone, balancing inference speed and segmentation accuracy.

### Loss Function
We use a combination of **Cross-Entropy Loss**, **Focal Loss** (to handle class imbalance), and **Dice Loss** (for better boundary definitions).

### Performance Metrics
*   **mIoU (Mean Intersection over Union)**: Primary metric for segmentation quality.
*   **Pixel Accuracy**: Overall accuracy of classification.
*   **Safety Score**: A custom metric calculating the model's reliability in identifying safe vs. unsafe zones.

---

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve the model accuracy or add new features!

---

## 📜 License
This project is for educational and research purposes. Please refer to the TU Graz dataset license for data usage terms.
