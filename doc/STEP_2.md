## Step 2: The Non-Linear Perceptron (Classification)

### The Physics Problem: Phase Boundary Identification

Instead of fitting a continuous line, let's look at a thermodynamic or phase-space system. Imagine you are monitoring a fluid characterized by two state variables: **Temperature ($T$)** and **Pressure ($P$)**.

Under a simplified model, there is a distinct boundary separating two phases (e.g., Liquid vs. Gas). Your goal is to build a classifier that takes a coordinate $(T, P)$ and predicts whether the state is Liquid (Class `1`) or Gas (Class `0`).

The boundary isn't just a simple mapping to an infinite real number; the output must be a bound probability between $0$ and $1$.

---

### The Mathematical Framework

#### 1. Multivariable Linear Combination

Since we now have two inputs instead of one, our input is a vector $\mathbf{x} = [T, P]^T$, and our weights form a vector $\mathbf{w} = [w_1, w_2]^T$.
The pre-activation net input $z$ is the dot product plus a bias:


$$z = \mathbf{w} \cdot \mathbf{x} + b = w_1 T + w_2 P + b$$

#### 2. The Non-Linear Activation Function (Sigmoid)

To map this infinite scalar $z$ to a probability space $a \in (0, 1)$, we pass it through the logistic sigmoid function $\sigma(z)$:


$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### 3. The Loss Function (Binary Cross-Entropy)

For classification, Mean Squared Error behaves poorly during gradient descent due to non-convexity when combined with sigmoid activations. Instead, we use Binary Cross-Entropy (BCE) loss for $N$ samples:


$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(a_i) + (1 - y_i) \ln(1 - a_i) \right]$$


Where $y_i \in \{0, 1\}$ is the true phase label, and $a_i$ is the predicted probability.

#### 4. The Backward Pass (The Chain Rule)

To find how the loss changes with respect to our weights and bias, we compute the gradient via the chain rule:


$$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

When you compute the analytical derivatives, an elegant cancellation occurs. The localized gradient with respect to the net input $z$ simplifies beautifully to:


$$\frac{\partial L}{\partial z_i} = a_i - y_i$$

Thus, the final total gradients for your parameters across the dataset are:


$$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (a_i - y_i) \cdot x_{ji}$$

$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (a_i - y_i)$$

---

### Your Algorithmic Blueprint

To implement this without any libraries (using standard Python loops, lists, and the built-in `math` module for $e^{-z}$ and $\ln$), follow these structural steps:

1. **Synthesize or Define the Dataset:** * Create a list of tuples or lists representing your training points: `(Temperature, Pressure, Phase_Label)`.
* For example, pick a line like $P = 2T + 1$. Points significantly above it are Liquid (`1`), points below are Gas (`0`). Add a tiny bit of overlap near the line if you want to simulate real-world uncertainty.


2. **Initialization:**
* Initialize a weight list/vector `w` with two small arbitrary values (e.g., `[0.1, -0.1]`).
* Initialize a scalar bias `b` to `0.0`.
* Set your hyperparameters: a learning rate (e.g., `0.05`) and the number of epochs.


3. **The Core Training Loop (Repeat for each epoch):**
* **Reset Accumulators:** Initialize variables to accumulate the total epoch loss, `grad_w1`, `grad_w2`, and `grad_b` to `0.0`.
* **Forward Pass (Per Sample):**
* Calculate the dot product: $z = w_1 \cdot T + w_2 \cdot P + b$.
* Compute the activation: $a = \frac{1}{1 + \exp(-z)}$.
* Compute the sample's BCE loss and add it to the running total.


* **Backward Pass (Per Sample):**
* Compute the prediction error: $\delta = a - y$.
* Accumulate the gradients: `grad_w1 += error * T`, `grad_w2 += error * P`, and `grad_b += error`.


* **Averages & Updates:**
* Divide the accumulated loss and gradients by the total number of samples $N$.
* Update your parameters using vanilla gradient descent: $w_j \leftarrow w_j - \eta \cdot \text{grad\_w}_j$.


* **Reporting:** Print the average BCE loss every few hundred epochs to ensure it is strictly minimizing.



Once you implement this, you will have built a classic Logistic Regressor entirely from first principles. Give this a shot in your environment. Let me know when your phase-boundary classifier converges, and we will move to Step 3: connecting these neurons into a multi-layer network where backpropagation truly comes alive.
