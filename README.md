# 🔢 Handwritten Digit Recognizer

A web app that recognizes handwritten digits (0–9) using a neural network trained on the MNIST dataset.  
Built with TensorFlow, Streamlit, and Python.

---

## 📸 Features

- **Upload an image** of a handwritten digit and get an instant prediction
- **Draw a digit** directly on the canvas inside the app
- Shows **confidence score** and a **bar chart** for all 10 digits

---

## 🧠 How It Works

1. A neural network is trained on the MNIST dataset (60,000 handwritten digit images)
2. The model learns to classify digits 0–9 with ~98% accuracy
3. The Streamlit app loads the saved model and runs predictions on new input

**Model Architecture:**
```
Input (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

---

## 🗂️ Project Structure

```
digit-recognizer/
├── digit_recognition.ipynb   # Training notebook — builds and saves the model
├── digit_stream.py           # Streamlit web app
├── mnist_model.keras         # Saved trained model (generated after running notebook)
└── requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model (if mnist_model.keras is not present)
Open and run all cells in `digit_recognition.ipynb`.  
This will save `mnist_model.keras` in the same folder.

### Step 3 — Launch the web app
```bash
streamlit run digit_stream.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| TensorFlow / Keras | Neural network training & inference |
| Streamlit | Web app UI |
| streamlit-drawable-canvas | Drawing canvas in the app |
| NumPy | Array operations |
| Pillow | Image processing |
| Scikit-learn | Confusion matrix & classification report |
| Matplotlib | Training history plots |

---

## 📊 Model Performance

- **Dataset:** MNIST (60,000 train / 10,000 test images)
- **Test Accuracy:** ~98%
- **Epochs:** 10
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy

---

## 👤 Author

Made by **Vaishnavi**  
E&TC Diploma Student — Government Polytechnic Nagpur
