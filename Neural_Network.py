'''
 This .py holds a basic implementation of a Neural Network trained to predict a computation XOR
 i. XOR was specifically choosen because AND and OR can be solved with linear model 
 ii. XOR is pedagogical as historically XOR had put a hault in neural network research, 
     The XOR problem was later solved using multi-layer perceptrons (MLPs) 
 iii. XOR is non linear, and requires multiple layers

 
************************************************************
XOR Table: 

A B | A XOR B
------------
0 0 |   0
0 1 |   1
1 0 |   1
1 1 |   0
************************************************************

'''

###############################################################################################
#PART 1 : the first part of the code focuses on single pass, forward as well as backward pass 
###############################################################################################

x1 = 1 #input 1 
x2 = 0 #input 2

# A default weight for a simple first pass 
w1_n1, w2_n1, b_n1 = 0.5, 0.5, 0.0   # Neuron 1
w1_n2, w2_n2, b_n2 = 0.5, 0.5, 0.0   # Neuron 2
w1_out, w2_out, b_out = 0.5, 0.5, 0.0   # Output

z1 = x1*w1_n1+x2*w2_n1+b_n1
z2 = x1*w1_n2+x2*w2_n2+b_n2

def sigmoid(x):
    # I've not used math.exp(-x) , or np.exp(-x)
    # This approach forces me to understand what's going on under the hood. 
    # To improve precision, we could use Taylor Series to compute e, but it adds un-necessary complexity. 
    e = 2.718281828459045 
    return 1 / (1 + e ** (-x))

a1 = sigmoid(z1)
a2 = sigmoid(z2)

z_out = a1*w1_out+a2*w2_out+b_out
a_out = sigmoid(z_out)

print(f"First Pass: {a_out}")

#LOSS: 
# y is because XOR (1,0) -> (1)
# refer back to the table at the top for reference 

y = 1 
loss = (a_out-y)**2
print(f"First passe's Loss: {loss}\n")


#Back Propogation 

# 1. Partial derivative of loss WRT z_out (More precisely: Partial derivative to weights at z_out)
delta_out = a_out*(1-a_out)*2*(a_out-y)

# 2. Partial derivative of loss WRT z1
delta_z1 = delta_out*a1*(1-a1)*w1_out

# 3. Partial derivative of loss WRT z2
delta_z2 = delta_out*a2*(1-a2)*w2_out


# Finding the minimum loss 
learning_rate = 0.01

# output neurone 
w1_out_new = w1_out - learning_rate * delta_out * a1
w2_out_new = w2_out - learning_rate * delta_out * a2
b_out_new = b_out - learning_rate * delta_out


# neurone 1
w1_n1_new = w1_n1 - learning_rate * delta_z1 * x1
w2_n1_new = w2_n1 - learning_rate * delta_z1 * x2
b_n1_new = b_n1 - learning_rate * delta_z1

# neurone 2 
w1_n2_new = w1_n2 - learning_rate * delta_z2* x1
w2_n2_new = w2_n2 - learning_rate * delta_z2* x2
b_n2_new = b_n2 - learning_rate * delta_z2


'''
w1_out, w2_out, b_out = w1_out_new, w2_out_new, b_out_new
w1_n1, w2_n1, b_n1 = w1_n1_new, w2_n1_new, b_n1_new
w1_n2, w2_n2, b_n2 = w1_n2_new, w2_n2_new, b_n2_new
'''

# Compare old vs new weights/biases
print("Weight & Bias Comparison:")
print("-" * 50)
print(f"{'Parameter':<12} {'Old':>12} {'New':>12} {'Change':>12}")
print("-" * 50)
for name, old, new in [
    ("w1_n1", w1_n1, w1_n1_new), ("w2_n1", w2_n1, w2_n1_new), ("b_n1", b_n1, b_n1_new),
    ("w1_n2", w1_n2, w1_n2_new), ("w2_n2", w2_n2, w2_n2_new), ("b_n2", b_n2, b_n2_new),
    ("w1_out", w1_out, w1_out_new), ("w2_out", w2_out, w2_out_new), ("b_out", b_out, b_out_new)]:
    print(f"{name:<12} {old:>12.6f} {new:>12.6f} {new-old:>12.6f}")




###############################################################################################
#PART 2 : This part focuses on creating a fully trained model that predicts XOR
###############################################################################################


training_data = [
    (0, 0, 0),  # x1=0, x2=0, target=0
    (0, 1, 1),  # x1=0, x2=1, target=1
    (1, 0, 1),  # x1=1, x2=0, target=1
    (1, 1, 0),  # x1=1, x2=1, target=0
]

# Reset weights to starting values for fresh training
import random 
random.seed(11)

w1_n1, w2_n1, b_n1 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
w1_n2, w2_n2, b_n2 = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
w1_out, w2_out, b_out = random.uniform(-1, 1), random.uniform(-1, 1), 0.0
learning_rate = 0.01

epochs = 200000
for epoch in range(epochs):
    for x1, x2, y in training_data:

        # Forward pass
        z1 = x1*w1_n1 + x2*w2_n1 + b_n1
        z2 = x1*w1_n2 + x2*w2_n2 + b_n2
        a1 = sigmoid(z1)
        a2 = sigmoid(z2)

        z_out = a1*w1_out + a2*w2_out + b_out
        a_out = sigmoid(z_out)

        # Backpropagation
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


# Test the trained network
print("\n" + "="*50)
print("PART 2: Trained Network Results")
print("="*50)
for x1, x2, y in training_data:
    z1 = x1*w1_n1 + x2*w2_n1 + b_n1
    z2 = x1*w1_n2 + x2*w2_n2 + b_n2
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    z_out = a1*w1_out + a2*w2_out + b_out
    a_out = sigmoid(z_out)
    new_val = 1 if a_out > 0.5 else 0  

    print(f"Input: ({x1}, {x2}) | Expected: {y} | Predicted: {new_val}")



'''
************************************************************
LESSONS LEARNED
************************************************************

1. Symmetry Breaking
   Mistake: Used identical weights (0.5) for both hidden neurons.

   What happened: Both neurons computed the same thing and received
   the same updates. They stayed identical forever—essentially acting
   as one neuron. One neuron can't solve XOR.

   Fix: Initialize weights randomly between -1 and 1. I seeded the
   random generator for reproducible results across machines.


2. Learning Rate
   Mistake: Started with learning_rate = 0.5, which was too high.

   What happened: The network overshot the optimal weights and
   oscillated instead of converging.

   Fix: Start small (Andrew Ng suggests 0.01), then adjust.
   Too high = overshoots. Too low = learns too slowly.


3. Epochs
   Mistake: Assumed more epochs = better results.

   What happened: Training for too long led to wasted computation
   and didn't improve results once loss plateaued.

   Fix: Monitor the loss. Stop when it stops decreasing.


Key Takeaway: There's no universal "correct" hyperparameter.
Start conservative, observe, and adjust.
'''