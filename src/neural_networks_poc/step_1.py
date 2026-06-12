
# 1. Experimental Data

data = [
    (0.10, 0.53),
    (0.20, 0.88),
    (0.30, 1.22),
    (0.40, 1.61),
    (0.50, 1.94)
]
N = len(data)

def main() -> None:
    print("Running  Step 1: Basic Neural Network Structure")

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

if __name__ == "__main__":
    main()
