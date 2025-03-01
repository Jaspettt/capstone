import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Question3_Final_CP.csv")

# Extract input features (X) and target (Y)
X = df[['X1', 'X2', 'X3']].values
y = df['Y'].values

# Normalize features using Z-score
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

# Add bias term (column of ones)
m, n = X_norm.shape
X_norm = np.c_[np.ones(m), X_norm]  # Add column of ones for bias term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function with regularization
def compute_cost(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)  # Regularization term (excluding bias)
    return cost + reg

# Gradient Descent with Regularization
def gradient_descent(X, y, theta, alpha, lambda_, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        gradient[1:] += (lambda_ / m) * theta[1:]  # Apply regularization (excluding bias term)
        theta -= alpha * gradient
    return theta

# Run the model for different iteration settings
settings = [
    (100, 0.1, 0.1),
    (1000, 0.2, 1),
    (10000, 0.3, 10)
]

results = {}

for iterations, alpha, lambda_ in settings:
    theta = np.zeros(n + 1)  # Initialize theta to zero
    theta_optimal = gradient_descent(X_norm, y, theta, alpha, lambda_, iterations)
    
    cost = compute_cost(theta_optimal, X_norm, y, lambda_)
    max_theta = np.round(np.max(theta_optimal), 2)

    results[(iterations, alpha, lambda_)] = {
        "Cost Function": np.round(cost, 2),
        "Optimal Theta": theta_optimal,
        "Max Theta": max_theta
    }

# Print results
for key, value in results.items():
    iterations, alpha, lambda_ = key
    print(f"Iterations: {iterations}, Alpha: {alpha}, Lambda: {lambda_}")
    print(f"Cost Function: {value['Cost Function']}")
    print(f"Max Theta: {value['Max Theta']}\n")

# Make predictions for last setting (10,000 iterations)
final_theta = results[(10000, 0.3, 10)]['Optimal Theta']
predictions = sigmoid(X_norm @ final_theta) >= 0.5  # Threshold at 0.5

# Count number of ones in the first 10 rows
num_ones = np.sum(predictions[:10])
print("Number of ones in first 10 predictions:", num_ones)
