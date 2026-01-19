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
```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Derivative (used in backpropagation):**
```
σ'(x) = σ(x) · (1 - σ(x))
```

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
```
z₁ = w₁₁ · x₁ + w₁₂ · x₂ + b₁

a₁ = σ(z₁) = 1 / (1 + e⁻ᶻ¹)
```

**Calculation:** (x₁=1, x₂=0, weights=0.5, bias=0)
```
z₁ = 0.5 · 1 + 0.5 · 0 + 0 = 0.5

a₁ = 1 / (1 + e⁻⁰·⁵) = 1 / 1.606 = 0.6225
```

**Code:**
```python
z_n1 = w1_n1 * x1 + w2_n1 * x2 + b_n1
a_n1 = sigmoid(z_n1)
```

---

### Hidden Neuron 2

**Equation:**
```
z₂ = w₂₁ · x₁ + w₂₂ · x₂ + b₂

a₂ = σ(z₂) = 1 / (1 + e⁻ᶻ²)
```

**Calculation:**
```
z₂ = 0.5 · 1 + 0.5 · 0 + 0 = 0.5

a₂ = 1 / (1 + e⁻⁰·⁵) = 0.6225
```

**Code:**
```python
z_n2 = w1_n2 * x1 + w2_n2 * x2 + b_n2
a_n2 = sigmoid(z_n2)
```

---

### Output Neuron

**Equation:**
```
zₒᵤₜ = wₒ₁ · a₁ + wₒ₂ · a₂ + bₒᵤₜ

aₒᵤₜ = σ(zₒᵤₜ) = 1 / (1 + e⁻ᶻᵒᵘᵗ)
```

**Calculation:**
```
zₒᵤₜ = 0.5 · 0.6225 + 0.5 · 0.6225 + 0 = 0.6225

aₒᵤₜ = 1 / (1 + e⁻⁰·⁶²²⁵) = 0.6508
```

**Code:**
```python
z_out = w1_out * a_n1 + w2_out * a_n2 + b_out
a_out = sigmoid(z_out)
```

---

## LOSS FUNCTION

**Equation:**
```
L = (aₒᵤₜ - y)²
```

**Calculation:** (target y=1 for XOR(1,0))
```
L = (0.6508 - 1)² = (-0.3492)² = 0.1220
```

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
```
δₒᵤₜ = ∂L/∂aₒᵤₜ · ∂aₒᵤₜ/∂zₒᵤₜ

∂L/∂aₒᵤₜ = 2(aₒᵤₜ - y)

∂aₒᵤₜ/∂zₒᵤₜ = aₒᵤₜ(1 - aₒᵤₜ)

δₒᵤₜ = 2(aₒᵤₜ - y) · aₒᵤₜ(1 - aₒᵤₜ)
```

**Calculation:**
```
∂L/∂aₒᵤₜ = 2(0.6508 - 1) = -0.6984

∂aₒᵤₜ/∂zₒᵤₜ = 0.6508 · (1 - 0.6508) = 0.6508 · 0.3492 = 0.2273

δₒᵤₜ = -0.6984 · 0.2273 = -0.1587
```

**Code:**
```python
delta_out = 2 * (a_out - y) * a_out * (1 - a_out)
```

---

### Neuron 1 Delta

**Equation:**
```
δ₁ = ∂L/∂zₒᵤₜ · ∂zₒᵤₜ/∂a₁ · ∂a₁/∂z₁

δ₁ = δₒᵤₜ · wₒ₁ · a₁(1 - a₁)
```

**Calculation:**
```
δ₁ = -0.1587 · 0.5 · 0.6225 · (1 - 0.6225)
   = -0.1587 · 0.5 · 0.6225 · 0.3775
   = -0.0187
```

**Code:**
```python
delta_n1 = delta_out * w1_out * a_n1 * (1 - a_n1)
```

---

### Neuron 2 Delta

**Equation:**
```
δ₂ = δₒᵤₜ · wₒ₂ · a₂(1 - a₂)
```

**Calculation:**
```
δ₂ = -0.1587 · 0.5 · 0.6225 · 0.3775 = -0.0187
```

**Code:**
```python
delta_n2 = delta_out * w2_out * a_n2 * (1 - a_n2)
```

---

## UPDATE WEIGHTS

**General equation:**
```
wₙₑᵥ = wₒₗₐ - η · δ · input
```

Where η (eta) is the learning rate.

---

### Output Neuron Weights

**Equations:**
```
wₒ₁ = wₒ₁ - η · δₒᵤₜ · a₁
wₒ₂ = wₒ₂ - η · δₒᵤₜ · a₂
bₒᵤₜ = bₒᵤₜ - η · δₒᵤₜ
```

**Calculation:** (η = 0.5)
```
wₒ₁ = 0.5 - 0.5 · (-0.1587) · 0.6225 = 0.5 + 0.0494 = 0.5494
wₒ₂ = 0.5 - 0.5 · (-0.1587) · 0.6225 = 0.5494
bₒᵤₜ = 0 - 0.5 · (-0.1587) = 0.0794
```

**Code:**
```python
w1_out = w1_out - learning_rate * delta_out * a_n1
w2_out = w2_out - learning_rate * delta_out * a_n2
b_out = b_out - learning_rate * delta_out
```

---

### Neuron 1 Weights

**Equations:**
```
w₁₁ = w₁₁ - η · δ₁ · x₁
w₁₂ = w₁₂ - η · δ₁ · x₂
b₁ = b₁ - η · δ₁
```

**Calculation:** (x₁=1, x₂=0)
```
w₁₁ = 0.5 - 0.5 · (-0.0187) · 1 = 0.5 + 0.0093 = 0.5093
w₁₂ = 0.5 - 0.5 · (-0.0187) · 0 = 0.5
b₁ = 0 - 0.5 · (-0.0187) = 0.0093
```

**Code:**
```python
w1_n1 = w1_n1 - learning_rate * delta_n1 * x1
w2_n1 = w2_n1 - learning_rate * delta_n1 * x2
b_n1 = b_n1 - learning_rate * delta_n1
```

---

### Neuron 2 Weights

**Equations:**
```
w₂₁ = w₂₁ - η · δ₂ · x₁
w₂₂ = w₂₂ - η · δ₂ · x₂
b₂ = b₂ - η · δ₂
```

**Calculation:**
```
w₂₁ = 0.5 - 0.5 · (-0.0187) · 1 = 0.5093
w₂₂ = 0.5 - 0.5 · (-0.0187) · 0 = 0.5
b₂ = 0 - 0.5 · (-0.0187) = 0.0093
```

**Code:**
```python
w1_n2 = w1_n2 - learning_rate * delta_n2 * x1
w2_n2 = w2_n2 - learning_rate * delta_n2 * x2
b_n2 = b_n2 - learning_rate * delta_n2
```

---

## TRAINING LOOP

Repeat for many epochs:
1. Forward pass — compute prediction
2. Compute loss — measure error
3. Backward pass — compute gradients
4. Update weights — reduce error

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
        w2_out -= learning_rate * delta_out * a2
        b_out -= learning_rate * delta_out

        w1_n1 -= learning_rate * delta_z1 * x1
        w2_n1 -= learning_rate * delta_z1 * x2
        b_n1 -= learning_rate * delta_z1

        w1_n2 -= learning_rate * delta_z2 * x1
        w2_n2 -= learning_rate * delta_z2 * x2
        b_n2 -= learning_rate * delta_z2
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
