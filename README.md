# Multi-class Fashion Classifier with TensorFlow

A deep neural network built **from scratch** using TensorFlow's `GradientTape` to classify clothing images into 10 categories. No `model.fit()` — every part of the training loop is written manually.

---

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 95.48% |
| Test Accuracy | 88.29% |
| Classes | 10 |
| Training Examples | 60,000 |

---

## Dataset

**Fashion-MNIST** — 70,000 grayscale images (28×28 pixels) of clothing items across 10 classes:

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Network Architecture

```
Input (784)  →  Layer 1 (256, ReLU)  →  Layer 2 (128, ReLU)  →  Output (10, Softmax)
```

---

## Techniques Used

- **He Initialization** — smart weight initialization for ReLU networks, prevents vanishing/exploding gradients
- **Mini-batch Gradient Descent** — training on batches of 512 examples at a time
- **Adam Optimizer** — adaptive learning rate optimization
- **L2 Regularization** (λ=0.1) — penalizes large weights to reduce overfitting
- **Softmax + Categorical Cross-Entropy** — for multi-class classification
- **GradientTape** — manual training loop, backprop handled automatically by TensorFlow

---

## Project Structure

```
├── fashion_classifier.ipynb   # Full notebook with training + evaluation
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/aosewify/fashion-mnist-classifier
cd fashion-mnist-classifier
```

**2. Install dependencies**
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

**3. Run the notebook**
```bash
jupyter notebook fashion_classifier.ipynb
```

> Note: Dataset is loaded automatically via `fetch_openml` from scikit-learn (no manual download needed).

---

## Training Cost Curve

The cost decreases smoothly across 70 epochs, confirming stable training with Adam:

```
Epoch 0:   ~0.69
Epoch 10:  ~0.22
Epoch 30:  ~0.13
Epoch 60:  ~0.08
```

---

## Sample Predictions

The model correctly identifies clothing items from the test set with high confidence:

```
True Label:  Ankle boot  →  Predicted: Ankle boot  ✅  (96.2% confidence)
True Label:  Sneaker     →  Predicted: Sneaker      ✅  (91.4% confidence)
True Label:  Shirt       →  Predicted: Shirt        ✅  (84.7% confidence)
```

---

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
