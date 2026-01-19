# Neural Network from Scratch: XOR

A neural network built without libraries to deeply understand forward propagation, backpropagation, and gradient descent.

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
- **Input Layer:** 2 neurons (x1, x2)
- **Hidden Layer:** 2 neurons with sigmoid activation
- **Output Layer:** 1 neuron with sigmoid activation

**Parameters (weights and biases):**
- Hidden Neuron 1: `w1_n1`, `w2_n1`, `b_n1`
- Hidden Neuron 2: `w1_n2`, `w2_n2`, `b_n2`
- Output Neuron: `w1_out`, `w2_out`, `b_out`

Total: 9 learnable parameters

---

## The Sigmoid Function

The sigmoid (σ) squashes any input to a value between 0 and 1:

```
σ(x) = 1 / (1 + e^(-x))
```

**Why sigmoid?**
- Output is bounded (0, 1) — interpretable as probability
- Differentiable everywhere — required for gradient descent
- Non-linear — allows learning complex patterns

**Key property for backpropagation:**
```
σ'(x) = σ(x) · (1 - σ(x))
```

This elegant derivative makes calculations simple: if you already have `a = σ(z)`, then the derivative is just `a · (1 - a)`.

---

## Forward Pass

The forward pass computes the network's prediction given inputs.

### Step 1: Hidden Layer

**Neuron 1 — Weighted Sum:**
```
z1 = x1·w1_n1 + x2·w2_n1 + b_n1
```

**Neuron 1 — Activation:**
```
a1 = σ(z1) = 1 / (1 + e^(-z1))
```

**Neuron 2 — Weighted Sum:**
```
z2 = x1·w1_n2 + x2·w2_n2 + b_n2
```

**Neuron 2 — Activation:**
```
a2 = σ(z2) = 1 / (1 + e^(-z2))
```

### Step 2: Output Layer

**Weighted Sum:**
```
z_out = a1·w1_out + a2·w2_out + b_out
```

**Activation (Final Prediction):**
```
a_out = σ(z_out) = 1 / (1 + e^(-z_out))
```

### Example Calculation

Given: `x1 = 1`, `x2 = 0`, all weights = 0.5, all biases = 0

```
z1 = 1·0.5 + 0·0.5 + 0 = 0.5
a1 = 1 / (1 + e^(-0.5)) = 1 / 1.606 = 0.6225

z2 = 1·0.5 + 0·0.5 + 0 = 0.5
a2 = 0.6225

z_out = 0.6225·0.5 + 0.6225·0.5 + 0 = 0.6225
a_out = 1 / (1 + e^(-0.6225)) = 0.6508
```

Prediction: **0.6508** (Expected: 1 for XOR(1,0))

---

## Loss Function

We need a way to measure how wrong our prediction is. We use Mean Squared Error (MSE):

```
L = (a_out - y)²
```

Where:
- `a_out` = network's prediction
- `y` = actual target value

### Example

```
a_out = 0.6508
y = 1 (target for XOR(1,0))

L = (0.6508 - 1)² = (-0.3492)² = 0.1220
```

The goal of training: **minimize this loss**.

---

## Backward Pass (Backpropagation)

Backpropagation answers: "How much does each weight contribute to the error?"

We use the **chain rule** to propagate the error backward through the network, computing gradients (partial derivatives) for each weight.

### The Chain Rule

If `y = f(g(x))`, then:
```
dy/dx = dy/dg · dg/dx
```

This lets us break complex derivatives into simpler pieces.

---

### Step 1: Output Layer Gradient (δ_out)

We want: `∂L/∂z_out` — how does changing z_out affect the loss?

**Apply chain rule:**
```
∂L/∂z_out = ∂L/∂a_out · ∂a_out/∂z_out
```

**Compute each part:**

```
∂L/∂a_out = ∂/∂a_out[(a_out - y)²] = 2(a_out - y)
```

```
∂a_out/∂z_out = σ'(z_out) = a_out · (1 - a_out)
```

**Combine:**
```
δ_out = 2(a_out - y) · a_out · (1 - a_out)
```

### Example Calculation

```
a_out = 0.6508, y = 1

∂L/∂a_out = 2(0.6508 - 1) = 2(-0.3492) = -0.6984

∂a_out/∂z_out = 0.6508 · (1 - 0.6508) = 0.6508 · 0.3492 = 0.2273

δ_out = -0.6984 · 0.2273 = -0.1587
```

---

### Step 2: Hidden Layer Gradients (δ1, δ2)

Now we propagate the error back to the hidden layer.

**For hidden neuron 1:**
```
∂L/∂z1 = ∂L/∂z_out · ∂z_out/∂a1 · ∂a1/∂z1
```

Breaking it down:
- `∂L/∂z_out = δ_out` (computed above)
- `∂z_out/∂a1 = w1_out` (from z_out = a1·w1_out + ...)
- `∂a1/∂z1 = a1 · (1 - a1)` (sigmoid derivative)

**Combine:**
```
δ1 = δ_out · w1_out · a1 · (1 - a1)
```

**For hidden neuron 2:**
```
δ2 = δ_out · w2_out · a2 · (1 - a2)
```

### Example Calculation

```
δ_out = -0.1587, w1_out = 0.5, a1 = 0.6225

δ1 = -0.1587 · 0.5 · 0.6225 · (1 - 0.6225)
   = -0.1587 · 0.5 · 0.6225 · 0.3775
   = -0.0187
```

---

### Step 3: Weight Gradients

Now we compute how much each individual weight affects the loss.

**Output layer weights:**
```
∂L/∂w1_out = δ_out · a1
∂L/∂w2_out = δ_out · a2
∂L/∂b_out  = δ_out
```

**Hidden layer weights:**
```
∂L/∂w1_n1 = δ1 · x1
∂L/∂w2_n1 = δ1 · x2
∂L/∂b_n1  = δ1

∂L/∂w1_n2 = δ2 · x1
∂L/∂w2_n2 = δ2 · x2
∂L/∂b_n2  = δ2
```

---

## Gradient Descent (Weight Updates)

Once we have the gradients, we update weights to reduce the loss:

```
w_new = w_old - η · ∂L/∂w
```

Where `η` (eta) is the **learning rate** — a small number controlling step size.

**Output layer updates:**
```
w1_out = w1_out - η · δ_out · a1
w2_out = w2_out - η · δ_out · a2
b_out  = b_out  - η · δ_out
```

**Hidden layer updates:**
```
w1_n1 = w1_n1 - η · δ1 · x1
w2_n1 = w2_n1 - η · δ1 · x2
b_n1  = b_n1  - η · δ1

w1_n2 = w1_n2 - η · δ2 · x1
w2_n2 = w2_n2 - η · δ2 · x2
b_n2  = b_n2  - η · δ2
```

### Example Calculation

```
η = 0.5, δ_out = -0.1587, a1 = 0.6225

w1_out_new = 0.5 - 0.5 · (-0.1587) · 0.6225
           = 0.5 - (-0.0494)
           = 0.5494
```

The weight increased because the gradient was negative (we were undershooting).

---

## Training Loop

Repeat for many epochs:
1. **Forward pass:** Compute prediction
2. **Compute loss:** How wrong are we?
3. **Backward pass:** Compute gradients
4. **Update weights:** Move toward lower loss

```python
for epoch in range(epochs):
    for x1, x2, y in training_data:
        # Forward pass
        z1 = x1*w1_n1 + x2*w2_n1 + b_n1
        a1 = sigmoid(z1)
        z2 = x1*w1_n2 + x2*w2_n2 + b_n2
        a2 = sigmoid(z2)
        z_out = a1*w1_out + a2*w2_out + b_out
        a_out = sigmoid(z_out)

        # Backward pass
        delta_out = a_out*(1-a_out)*2*(a_out-y)
        delta_z1 = delta_out*a1*(1-a1)*w1_out
        delta_z2 = delta_out*a2*(1-a2)*w2_out

        # Update weights
        w1_out -= learning_rate * delta_out * a1
        # ... (all other weights)
```

---

## Lessons Learned

### 1. Symmetry Breaking

**Problem:** Initialized all weights to the same value (0.5).

**What happened:** Both hidden neurons computed identical outputs and received identical gradient updates. They stayed identical forever—effectively collapsing into one neuron. A single hidden neuron cannot solve XOR.

**Solution:** Initialize weights randomly. Each neuron starts different and learns different features.

---

### 2. Learning Rate

**Problem:** Started with `learning_rate = 0.5` (too high).

**What happened:** The network overshot the optimal weights and oscillated around the minimum instead of converging.

**Solution:** Start small (Andrew Ng recommends 0.01). If learning is too slow, gradually increase. If loss oscillates or explodes, decrease.

```
Too high: overshoots the minimum
Too low:  takes forever to converge
Just right: smooth decrease in loss
```

---

### 3. Epochs

**Problem:** Assumed more epochs always means better results.

**What happened:** After the loss plateaued, additional training was wasted computation. In some cases, too many epochs can lead to overfitting or numerical instability.

**Solution:** Monitor the loss during training. Stop when it stops decreasing significantly (early stopping).

---

## Run the Code

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

---

## Key Takeaways

1. **Forward pass** computes predictions layer by layer
2. **Loss function** measures prediction error
3. **Backpropagation** uses the chain rule to find how each weight affects loss
4. **Gradient descent** updates weights to minimize loss
5. **Hyperparameters** (learning rate, epochs) require tuning—there's no universal correct value
