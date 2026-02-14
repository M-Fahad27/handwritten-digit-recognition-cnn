# 🧠 Handwritten Digit Recognition using TensorFlow & Keras

This project demonstrates a **Convolutional Neural Network (CNN)** built to recognize **handwritten digits (0–9)** using the **MNIST dataset**.
The model is trained using **TensorFlow & Keras**, achieves **~98% accuracy**, and is further tested on **real custom images** using **OpenCV preprocessing**.

---

## 📌 Project Overview

The goal of this project is to:

* Understand **deep learning fundamentals**
* Build a **CNN for image classification**
* Perform **model training, evaluation, saving, and loading**
* Test the model on **real-world handwritten digit images**

---

## 📂 Dataset

We used the **MNIST dataset**, which contains:

* **60,000 training images**
* **10,000 test images**
* Grayscale handwritten digits from **0–9**
* Image size: **28 × 28 pixels**

This dataset is widely used as a **benchmark for computer vision and deep learning beginners**.

---

## ⚙️ Technologies Used

* **Python**
* **TensorFlow & Keras** → model building and training
* **NumPy** → numerical operations
* **OpenCV** → image preprocessing for real-world testing

### Why TensorFlow & Keras?

* High-level and **beginner-friendly API**
* Fast **GPU/CPU optimized training**
* Easy **model building, evaluation, and deployment**
* Industry-standard **deep learning framework**

---

## 🏗️ Model Architecture (CNN)

The model is built using a **Sequential CNN** consisting of:

1. **Conv2D** – extracts visual features like edges and shapes
2. **MaxPooling2D** – reduces spatial size while keeping important information
3. **Flatten** – converts 2D feature maps into a 1D vector
4. **Dense (ReLU)** – learns complex patterns
5. **Dropout** – prevents overfitting and improves generalization
6. **Dense (Softmax)** – outputs probabilities for digits **0–9**

---

## 🏋️ Training Details

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 32
* **Epochs:** 10
* **Validation Split:** 20%

### 📊 Result

* **Test Accuracy:** ~98%
* The model performs well on **unseen handwritten digits**.

---

## 🔍 Real-World Testing with OpenCV

To make the project practical, the trained model was tested on **custom handwritten digit images**.

### Preprocessing Steps:

1. Load image in **grayscale**
2. **Resize to 28×28**
3. Convert to **black background with white digit** (MNIST format)
4. **Normalize pixel values (0–1)**
5. **Reshape** to match model input `(1, 28, 28, 1)`
6. Run **model prediction**

✅ The model correctly predicted digits such as **“9”** and performed accurately on multiple custom samples.

---

## 📚 Key Learnings

* Understanding of **CNN architecture** and image feature extraction
* Importance of **data normalization & correct input shape**
* Experience with **model training, evaluation, saving, and loading**
* Practical workflow from **deep learning theory → real AI application**
* Hands-on use of **OpenCV for real-world inference**

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install tensorflow numpy opencv-python
```

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Test on Custom Image

```bash
python test.py
```

---

## 📁 Project Structure

```
├── app.py              # Model training script
├── test.py               # Custom image prediction
├── mnist_classifier.h5   # Saved trained model
└── README.md             # Project documentation
```

---

## 🚀 Future Improvements

* Build a **GUI digit drawing app**
* Deploy as a **web application (Flask/Streamlit)**
* Train on **larger handwritten datasets**
* Convert model to **mobile-friendly format (TensorFlow Lite)**

---

## 🤝 Connect With Me

If you found this project useful or would like to collaborate on **AI, Machine Learning, or Computer Vision**, feel free to connect with me on **LinkedIn**.
www.linkedin.com/in/fahad2703

---

⭐ *If you like this project, don’t forget to star the repository!*
