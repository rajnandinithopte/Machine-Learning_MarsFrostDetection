# Machine-Learning: Mars Frost Detection
## 🔷 Identification of Frost in Martian HiRISE Images

### 🔶 Overview
This project focuses on **binary classification of Martian terrain** images to **detect frost** using **deep learning models**. A custom **CNN + MLP** model and **transfer learning** (EfficientNetB0, ResNet50, VGG16) were used for **feature extraction and classification**. Data augmentation, dropout, batch normalization, and L2 regularization were applied to improve model generalization.

## 🔷 Dataset Used
- **HiRISE Mars Terrain Images** ([Dataset Link](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA))
- The dataset contains **119,920 image tiles** extracted from **214 HiRISE subframes**.
- Each image tile is labeled as either **"frost"** or **"background"**.
- Provided splits for **training, validation, and testing**.

## 🔷 Libraries Used
- **TensorFlow/Keras** - Building deep learning models.
- **OpenCV** - Image augmentation (cropping, zooming, rotating, flipping).
- **Matplotlib & Seaborn** - Data visualization.
- **scikit-learn** - Model evaluation metrics.
- **Pandas & NumPy** - Data processing and numerical computations.

## 🔷 Steps Taken to Accomplish the Project

### 🔶 1. Data Preprocessing & Augmentation
- Resized images to **299x299 pixels**.
- Applied **image augmentation**: random cropping, zooming, flipping, contrast adjustment, and translation.
- Normalized pixel values between **0 and 1**.

### 🔶 2. Training a CNN + MLP Model
- Built a **3-layer Convolutional Neural Network (CNN)** followed by a **Multi-Layer Perceptron (MLP)**.
- Used **ReLU activation** for all layers.
- Applied **softmax function** for binary classification.
- Regularization techniques used:
  - **Batch Normalization**
  - **Dropout (30%)**
  - **L2 Regularization**
- **ADAM optimizer** and **cross-entropy loss** were used.
- Model trained for **at least 20 epochs**, with **early stopping based on validation loss**.
- **Precision, Recall, and F1-score** were reported.

### 🔶 3. Transfer Learning with Pre-trained CNNs
- Utilized **EfficientNetB0, ResNet50, and VGG16** for feature extraction.
- Froze all layers except the final **fully connected layer**.
- Extracted **features from the penultimate layer** and trained a classifier.
- Used **ReLU activation, batch normalization, dropout (30%)**, and **softmax activation**.
- Trained for **at least 10 epochs** (preferably 20 epochs) with **early stopping**.
- Compared results with **CNN + MLP model**.

### 🔶 4. Model Evaluation & Analysis
- Reported **Precision, Recall, and F1-score** for all models.
- Compared **CNN + MLP** vs. **Transfer Learning** performance.
- Plotted **training and validation loss curves** to analyze convergence.

### 🔶 5. Findings & Comparison
- **CNN + MLP** required **more training data** but performed well with augmentation.
- **Transfer Learning** (EfficientNetB0, ResNet50, VGG16) achieved **higher accuracy** due to pre-trained feature extraction.
- EfficientNetB0 provided **best performance** in terms of **validation loss and classification accuracy**.

  ---
## 📌 **Note**
This repository contains a **Jupyter Notebook** detailing each step, along with **results and visualizations**.
