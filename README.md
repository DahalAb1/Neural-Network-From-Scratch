# Neural Network from Scratch: XOR

A neural network built without libraries to deeply understand forward propagation, backpropagation, and learn how equations solve problems.

---

## Why XOR?

XOR (exclusive or) returns 1 when inputs differ, 0 when they match.

```
XOR Truth Table:
A | B | A XOR B
--|---|--------
0 | 0 |    0
0 | 1 |    1
1 | 0 |    1
1 | 1 |    0
```

**Historical Significance:** In 1969, Minsky and Papert published *Perceptrons*, proving that a single-layer perceptron cannot solve XOR. Why? Because XOR is not linearly separable—you cannot draw a single straight line to separate the 0s from the 1s. This limitation caused the first "AI Winter," stalling neural network research for over a decade.

The solution came with multi-layer perceptrons (MLPs) and the backpropagation algorithm, which allows networks to learn non-linear decision boundaries.

---

## Network Architecture

```
    INPUT           HIDDEN LAYER          OUTPUT
   LAYER
                   ┌─────────┐
                ┌──┤ Neuron 1├──┐
    ┌───┐       │  │  z1, a1 │  │       ┌─────────┐
    │x1 │───────┤  └─────────┘  ├───────┤ Output  │
    └───┘       │               │       │ z_out   │────► a_out
                │               │       │ a_out   │
    ┌───┐       │  ┌─────────┐  │       └─────────┘
    │x2 │───────┤  │ Neuron 2├──┘
    └───┘       └──┤  z2, a2 │
                   └─────────┘
```

**Structure:**
- **Input Layer:** 2 neurons (x₁, x₂)
- **Hidden Layer:** 2 neurons with sigmoid activation
- **Output Layer:** 1 neuron with sigmoid activation

**Parameters (9 total):**
- Hidden Neuron 1: w₁₁, w₁₂, b₁
- Hidden Neuron 2: w₂₁, w₂₂, b₂
- Output Neuron: wₒ₁, wₒ₂, bₒᵤₜ

---

## The Sigmoid Function

**Equation:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative (used in backpropagation):**

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

**Why sigmoid?**
- Outputs bounded between (0, 1)
- Differentiable everywhere
- Non-linear — allows learning complex patterns

---

## FORWARD PASS

The forward pass computes the network's prediction given inputs.

---

### Hidden Neuron 1

**Equation:**

$$z_1 = w_{11} \cdot x_1 + w_{12} \cdot x_2 + b_1$$

$$a_1 = \sigma(z_1) = \frac{1}{1 + e^{-z_1}}$$

**Calculation:** (x₁=1, x₂=0, weights=0.5, bias=0)

$$z_1 = 0.5 \cdot 1 + 0.5 \cdot 0 + 0 = 0.5$$

$$a_1 = \frac{1}{1 + e^{-0.5}} = \frac{1}{1.606} = 0.6225$$

**Code:**
```python
z_n1 = w1_n1 * x1 + w2_n1 * x2 + b_n1
a_n1 = sigmoid(z_n1)
```

---

### Hidden Neuron 2

**Equation:**

$$z_2 = w_{21} \cdot x_1 + w_{22} \cdot x_2 + b_2$$

$$a_2 = \sigma(z_2) = \frac{1}{1 + e^{-z_2}}$$

**Calculation:**

$$z_2 = 0.5 \cdot 1 + 0.5 \cdot 0 + 0 = 0.5$$

$$a_2 = \frac{1}{1 + e^{-0.5}} = 0.6225$$

**Code:**
```python
z_n2 = w1_n2 * x1 + w2_n2 * x2 + b_n2
a_n2 = sigmoid(z_n2)
```

---

### Output Neuron

**Equation:**

$$z_{out} = w_{o1} \cdot a_1 + w_{o2} \cdot a_2 + b_{out}$$

$$a_{out} = \sigma(z_{out}) = \frac{1}{1 + e^{-z_{out}}}$$

**Calculation:**

$$z_{out} = 0.5 \cdot 0.6225 + 0.5 \cdot 0.6225 + 0 = 0.6225$$

$$a_{out} = \frac{1}{1 + e^{-0.6225}} = 0.6508$$

**Code:**
```python
z_out = w1_out * a_n1 + w2_out * a_n2 + b_out
a_out = sigmoid(z_out)
```

---

## LOSS FUNCTION

**Equation:**

$$L = (a_{out} - y)^2$$

**Calculation:** (target y=1 for XOR(1,0))

$$L = (0.6508 - 1)^2 = (-0.3492)^2 = 0.1220$$

**Code:**
```python
loss = (a_out - y) ** 2
```

---

## BACKWARD PASS (Backpropagation)

The goal: find how each weight affects the loss using the chain rule, then adjust weights to minimize loss.

---

### Output Neuron Delta

**Equation:**

$$\delta_{out} = \frac{\partial L}{\partial a_{out}} \cdot \frac{\partial a_{out}}{\partial z_{out}}$$

Where:

$$\frac{\partial L}{\partial a_{out}} = \frac{\partial}{\partial a_{out}}(a_{out} - y)^2 = 2(a_{out} - y)$$

$$\frac{\partial a_{out}}{\partial z_{out}} = \sigma'(z_{out}) = a_{out}(1 - a_{out})$$

Therefore:

$$\delta_{out} = 2(a_{out} - y) \cdot a_{out}(1 - a_{out})$$

**Calculation:**

$$\frac{\partial L}{\partial a_{out}} = 2(0.6508 - 1) = 2(-0.3492) = -0.6984$$

$$\frac{\partial a_{out}}{\partial z_{out}} = 0.6508 \cdot (1 - 0.6508) = 0.6508 \cdot 0.3492 = 0.2273$$

$$\delta_{out} = -0.6984 \cdot 0.2273 = -0.1587$$

**Code:**
```python
delta_out = 2 * (a_out - y) * a_out * (1 - a_out)
```

---

### Neuron 1 Delta

**Equation:**

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial z_{out}} \cdot \frac{\partial z_{out}}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1}$$

Where:
- $\frac{\partial L}{\partial z_{out}} = \delta_{out}$
- $\frac{\partial z_{out}}{\partial a_1} = w_{o1}$
- $\frac{\partial a_1}{\partial z_1} = a_1(1 - a_1)$

Therefore:

$$\delta_1 = \delta_{out} \cdot w_{o1} \cdot a_1(1 - a_1)$$

**Calculation:**

$$\delta_1 = -0.1587 \cdot 0.5 \cdot 0.6225 \cdot (1 - 0.6225)$$

$$\delta_1 = -0.1587 \cdot 0.5 \cdot 0.6225 \cdot 0.3775$$

$$\delta_1 = -0.0187$$

**Code:**
```python
delta_n1 = delta_out * w1_out * a_n1 * (1 - a_n1)
```

---

### Neuron 2 Delta

**Equation:**

$$\delta_2 = \delta_{out} \cdot w_{o2} \cdot a_2(1 - a_2)$$

**Calculation:**

$$\delta_2 = -0.1587 \cdot 0.5 \cdot 0.6225 \cdot 0.3775 = -0.0187$$

**Code:**
```python
delta_n2 = delta_out * w2_out * a_n2 * (1 - a_n2)
```

---

## UPDATE WEIGHTS (Gradient Descent)

**General equation:**

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Where $\eta$ (eta) is the learning rate.

For our network, this becomes:

$$w_{new} = w_{old} - \eta \cdot \delta \cdot input$$

---

### Output Neuron Weights

**Equations:**

$$w_{o1} = w_{o1} - \eta \cdot \delta_{out} \cdot a_1$$

$$w_{o2} = w_{o2} - \eta \cdot \delta_{out} \cdot a_2$$

$$b_{out} = b_{out} - \eta \cdot \delta_{out}$$

**Calculation:** ($\eta$ = 0.5)

$$w_{o1} = 0.5 - 0.5 \cdot (-0.1587) \cdot 0.6225 = 0.5 + 0.0494 = 0.5494$$

$$w_{o2} = 0.5 - 0.5 \cdot (-0.1587) \cdot 0.6225 = 0.5494$$

$$b_{out} = 0 - 0.5 \cdot (-0.1587) = 0.0794$$

**Code:**
```python
w1_out = w1_out - learning_rate * delta_out * a_n1
w2_out = w2_out - learning_rate * delta_out * a_n2
b_out = b_out - learning_rate * delta_out
```

---

### Neuron 1 Weights

**Equations:**

$$w_{11} = w_{11} - \eta \cdot \delta_1 \cdot x_1$$

$$w_{12} = w_{12} - \eta \cdot \delta_1 \cdot x_2$$

$$b_1 = b_1 - \eta \cdot \delta_1$$

**Calculation:** (x₁=1, x₂=0)

$$w_{11} = 0.5 - 0.5 \cdot (-0.0187) \cdot 1 = 0.5 + 0.0093 = 0.5093$$

$$w_{12} = 0.5 - 0.5 \cdot (-0.0187) \cdot 0 = 0.5$$

$$b_1 = 0 - 0.5 \cdot (-0.0187) = 0.0093$$

**Code:**
```python
w1_n1 = w1_n1 - learning_rate * delta_n1 * x1
w2_n1 = w2_n1 - learning_rate * delta_n1 * x2
b_n1 = b_n1 - learning_rate * delta_n1
```

---

### Neuron 2 Weights

**Equations:**

$$w_{21} = w_{21} - \eta \cdot \delta_2 \cdot x_1$$

$$w_{22} = w_{22} - \eta \cdot \delta_2 \cdot x_2$$

$$b_2 = b_2 - \eta \cdot \delta_2$$

**Calculation:**

$$w_{21} = 0.5 - 0.5 \cdot (-0.0187) \cdot 1 = 0.5093$$

$$w_{22} = 0.5 - 0.5 \cdot (-0.0187) \cdot 0 = 0.5$$

$$b_2 = 0 - 0.5 \cdot (-0.0187) = 0.0093$$

**Code:**
```python
w1_n2 = w1_n2 - learning_rate * delta_n2 * x1
w2_n2 = w2_n2 - learning_rate * delta_n2 * x2
b_n2 = b_n2 - learning_rate * delta_n2
```

---

## LESSONS LEARNED

### 1. Symmetry Breaking

**Mistake:** Initialized all weights to the same value (0.5).

**What happened:** Both hidden neurons computed identical outputs and received identical gradient updates. They stayed identical forever—effectively acting as one neuron. One neuron cannot solve XOR.

**Fix:** Initialize weights randomly between -1 and 1. Seed the random generator for reproducibility.

---

### 2. Learning Rate

**Mistake:** Started with learning_rate = 0.5 (too high).

**What happened:** The network overshot the optimal weights and oscillated instead of converging.

**Fix:** Start small (Andrew Ng recommends 0.01). Increase if learning is too slow. Decrease if loss oscillates.

---

### 3. Epochs

**Mistake:** Assumed more epochs = better results.

**What happened:** After loss plateaued, additional training was wasted computation.

**Fix:** Monitor the loss. Stop when it stops decreasing.

---

## Run

```bash
python Neural_Network.py
```

**Expected Output:**
```
Input: (0, 0) | Expected: 0 | Predicted: 0
Input: (0, 1) | Expected: 1 | Predicted: 1
Input: (1, 0) | Expected: 1 | Predicted: 1
Input: (1, 1) | Expected: 0 | Predicted: 0
```
