# CIFAR-10 Image Classification using TensorFlow

This project implements a Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras to classify images from the CIFAR-10 dataset.

## 🚀 Features
- CNN architecture built from scratch
- Modern TensorFlow data augmentation layers
- Batch Normalization & Dropout for regularization
- EarlyStopping and ModelCheckpoint callbacks
- Clean evaluation using classification report

## 🧠 Model Architecture
- 3 convolutional blocks (32 → 64 → 128 filters)
- Batch normalization after each convolution
- MaxPooling for spatial reduction
- Fully connected layer with dropout
- Softmax output for multi-class classification

## 📊 Results
- **Test Accuracy:** ~70%
- **Macro F1-score:** ~0.70
- Balanced performance across classes

## 🛠 Tech Stack
- TensorFlow / Keras
- NumPy
- Scikit-learn

## 📁 Project Structure
```
image-classification-cnn/
│
├── src
    ├── data_loader.py
    ├── model.py
    ├── train.py
├── models/
└── README.md
```
## 🚀 How to Run
```
git clone https://github.com/Thashmila-Dewmini/image-classification-cnn.git
cd image-classification-cnn
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```
