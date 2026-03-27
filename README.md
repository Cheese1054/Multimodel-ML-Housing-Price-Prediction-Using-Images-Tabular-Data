# Multimodal Housing Price Prediction 🏠📊

This repository contains a **Task 3** implementation for the **Developers Hub Internship**. The project demonstrates a multimodal deep learning approach to predict house prices by combining tabular data and synthetic image data.

## 🌟 Overview
Traditional housing models rely solely on numerical data. This project implements a **Late Fusion Neural Network** that "sees" the house via a CNN and "reads" the specs via an MLP, merging both insights for a final valuation.



## 🛠️ Tech Stack
* **Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Data Handling:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, tqdm

## 🏗️ Architecture
The model consists of two specialized branches:
1.  **Image Branch (CNN):** Three convolutional layers with ReLU activation and MaxPooling to extract spatial features from 64x64 images.
2.  **Tabular Branch (MLP):** A dense neural network to process features like `area`, `bedrooms`, `bathrooms`, and `location_score`.
3.  **Fusion Layer:** Concatenates the outputs of both branches into a final set of dense layers to regress the house price.
