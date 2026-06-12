## Step 1: The Single Neuron (Linear Regression via Gradient Descent)

### The Physics Problem: Calibrating a Spring (Hooke's Law)

Imagine you are in a lab, and you have an ideal, linear spring. You hang various known masses $m$ from it and measure the equilibrium displacement $x$ from its natural length.

According to Hooke's Law:


$$F = k \cdot x$$

Where $F = m \cdot g$ is the gravitational force applied. We want to find the spring constant $k$. Rewriting this as a mapping from displacement to force:


$$F(x) = k \cdot x + b$$

Here, $b$ is an experimental bias term (accounting for an uncalibrated zero-point on your ruler or the mass of the spring pan itself).

In machine learning terms, this is a **single-neuron linear network** where:

* **Input ($x$):** The displacement.
* **Weight ($w$):** The spring constant $k$ we want to learn.
* **Bias ($b$):** The systematic shift we want to eliminate.
* **Output ($\hat{y}$):** The predicted force.

### The Synthetic Dataset

Let's assume the true underlying physics has $k = 3.5\text{ N/m}$ and $b = 0.2\text{ N}$. We collect 5 noisy experimental data points $(x, y)$:

```python
# Synthetic lab data: (displacement in meters, measured force in Newtons)
data = [
    (0.10, 0.53),
    (0.20, 0.88),
    (0.30, 1.22),
    (0.40, 1.61),
    (0.50, 1.94)
]

```

---

### The Mathematical Framework

#### 1. The Forward Pass

For a given input $x_i$, our model predicts the force:


$$\hat{y}_i = w \cdot x_i + b$$

#### 2. The Loss Function (Mean Squared Error)

To quantify how wrong our neuron is across all $N$ data points, we calculate the Mean Squared Error (MSE):


$$L = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$


*(Note: The factor of $\frac{1}{2}$ is a standard convention that cleanly cancels out during differentiation).*

#### 3. The Backward Pass (Analytical Optimization)

To minimize the loss, we calculate the gradient of $L$ with respect to our parameters $w$ and $b$ using the chain rule.

For a single data point's contribution to the error $E_i = \frac{1}{2}(\hat{y}_i - y_i)^2$:


$$\frac{\partial E_i}{\partial w} = \frac{\partial E_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial w} = (\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial E_i}{\partial b} = \frac{\partial E_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial b} = (\hat{y}_i - y_i) \cdot 1$$

Averaging over the entire dataset gives the total gradients:


$$\frac{\partial L}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

#### 4. Parameter Update

We update our weights in the opposite direction of the gradient to descend the loss landscape, scaled by a learning rate $\eta$:


$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

---

### Pure Python Implementation

Here is the entire training loop written without `numpy`, `scipy`, or any other external dependencies.

```python
# 1. Experimental Data
data = [
    (0.10, 0.53),
    (0.20, 0.88),
    (0.30, 1.22),
    (0.40, 1.61),
    (0.50, 1.94)
]
N = len(data)

# 2. Initialize Parameters (Arbitrary starting guesses)
w = 0.0  # Initial guess for spring constant k
b = 0.0  # Initial guess for instrument bias b

# Hyperparameters
learning_rate = 0.1
epochs = 2000

print(f"Initial State: Force(x) = {w:.4f}*x + {b:.4f}\n")

# 3. Training Loop
for epoch in range(epochs):
    total_loss = 0.0
    grad_w = 0.0
    grad_b = 0.0
    
    # Iterate through all experimental samples
    for x, y in data:
        # Forward pass
        y_pred = w * x + b
        
        # Accumulate Loss (MSE)
        total_loss += 0.5 * (y_pred - y) ** 2
        
        # Accumulate Gradients
        error = y_pred - y
        grad_w += error * x
        grad_b += error
        
    # Average the loss and gradients over the dataset
    mse_loss = total_loss / N
    grad_w /= N
    grad_b /= N
    
    # Update parameters (Gradient Descent step)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    # Print progress every 400 epochs
    if epoch % 400 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Loss: {mse_loss:.6f} | w (k): {w:.4f} | b: {b:.4f}")

print(f"\nFinal Calibrated System:")
print(f"Learned Spring Constant k = {w:.4f} N/m (True value: 3.5)")
print(f"Learned Systematic Bias b = {b:.4f} N   (True value: 0.2)")

```

### Steps to Execute and Observe

1. Run this script in a clean Python environment.
2. Observe how the loss steadily drops across epochs.
3. Notice how $w$ and $b$ converge tightly toward the hidden physical parameters ($k=3.5, b=0.2$) despite the simulated experimental noise in the dataset.

How does this setup look to you as a foundation? If you are comfortable with this single-element behavior, we can proceed to add our first non-linear activation function or scale this up to a multi-variable problem using basic vector dot products implemented via list comprehensions.
