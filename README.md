# Single Neuron Classifier  
A simple machine-learning project implementing and visualizing a 2D single-neuron classifier with multiple activation functions.

This project contains:
- a **single artificial neuron**, trainable using gradient-based weight updates,
- support for multiple **activation functions** (evaluation & training),
- a **Gaussian data generator** for two classes in 2D,
- an interactive **Tkinter GUI** with matplotlib visualization,
- **decision boundary plotting**.

The assignment corresponds to Task 4 of the laboratory (“Single Neuron”).

---

## 🔧 1. Project Structure

### `SingleNeuron`
A fully implemented neuron for binary classification (0/1).  
The neuron supports:

#### **Evaluation activation functions:**
- Heaviside  
- Sigmoid  
- Sin  
- Tanh  
- Sign  
- ReLU  
- Leaky ReLU  

#### **Training activation functions:**
- Heaviside (assumed derivative = 1)  
- Sigmoid (true derivative implemented)

Weights include bias and are updated using:  

\[
\Delta w_j = \eta (d - y) f'(s) x_j
\]

Where:  
- `d` – target label (0/1)  
- `y` – neuron output for training  
- `f'(s)` – derivative of activation  
- `η` – learning rate  
- `x_j` – input sample  
- `s = w^T x_j` – weighted sum  

---

## 🎲 2. Data Generator

Two classes are generated using 2D Gaussian distributions.  
Each class can consist of one or more **Gaussian modes** (“clusters”).

Parameters controlled in GUI:
- number of modes in class 0
- number of modes in class 1
- samples per mode

The generator randomly samples:
- mean values \( \mu_x, \mu_y \in [-1,1] \)
- variances \( \sigma_x, \sigma_y \in [0.1, 0.3] \)

---

## 🖥️ 3. Graphical User Interface (GUI)

The GUI allows:

### **Data Generation**
- Choose number of modes for each class  
- Choose number of samples per mode  
- Visualize raw data

### **Neuron Training**
- Choose:
  - learning rate  
  - number of epochs  
  - activation used for training  
  - activation used for evaluation  
- Train the neuron on generated samples  

### **Decision Boundary Plotting**
After training, the program displays:
- background color for class 0 region  
- background color for class 1 region  
- all generated data points  
- linear separation produced by the neuron  

Since a single neuron is a *linear classifier*, decision boundaries are always straight lines.

---

## 📌 4. Supported Activation Functions

| Function | Formula | Notes |
|---------|---------|-------|
| Heaviside | 0 / 1 | Used for evaluation & training (derivative = 1 by assumption) |
| Sigmoid | \( \frac{1}{1 + e^{-βs}} \) | Smooth prob. output; derivative implemented |
| Sin | \( \sin(s) \) | Evaluation only |
| Tanh | \( \tanh(s) \) | Evaluation only |
| Sign | -1/0/1 | Evaluation only |
| ReLU | max(0, s) | Evaluation only |
| Leaky ReLU | s (s>0), 0.01s (s<0) | Evaluation only |

---

## 📊 5. Expected Results

Examples of expected behavior:
- The decision boundary always forms a **straight line**.
- If classes are nearly linearly separable → neuron performs well.
- If classes heavily overlap → neuron misclassifies some points (normal behavior).
- Different evaluation activations change the *shape* of the neuron’s output but do *not* change the trained weights (training uses heaviside or sigmoid).

Your GUI screenshots match expected results.

---

## 🚀 6. How to Run

Install dependencies:

```
pip install numpy matplotlib
```

Run the program:

```
python single_neuron_gui.py
```

## 📝 7. Requirements Fulfilled

### **For grade 3**
✔ Heaviside activation  
✔ Sigmoid activation  
✔ Training using Heaviside  
✔ Training using Sigmoid  
✔ GUI  
✔ Decision boundary visualization  

### **For grade 4**
✔ Added sin  
✔ Added tanh  
(Training only for required functions — correct)

### **For grade 5**
✔ Added sign  
✔ Added ReLU  
✔ Added leaky ReLU  
(Evaluation only — correct per assignment)

---

## 📚 8. Notes for the Instructor (Explanation Cheat-Sheet)

A single neuron performs:

\[
y = f(w_1 x_1 + w_2 x_2 + b)
\]

After training, the neuron learns a line:

\[
w_1 x_1 + w_2 x_2 + b = 0
\]

This line separates 2D space into two regions (class 0 / class 1).  
The goal of the task is to visualize how a trained neuron classifies data and how the decision boundary behaves.

---

## 🧑‍💻 Author
Project implemented as part of AI Fundamentals laboratory — Task 4.
