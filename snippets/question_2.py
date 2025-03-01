import numpy as np
import pandas as pd
# Load the new dataset from CSV
file_path_csv = "./capstone_question/Question2_Dataset.csv"

# Read the CSV file
df_new = pd.read_csv(file_path_csv)

# Display the first few rows to inspect its structure
df_new.head()
# Extract features and target
X = df_new.iloc[:, :-1].values  # Features: X1, X2, X3, X4
y = df_new.iloc[:, -1].values   # Target: Y
m = len(y)  # Number of training examples

# Normalize features: Z = (x - mean) / std
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Add intercept term (column of ones) to X
X_norm = np.c_[np.ones(m), X_norm]  # Shape: (m, 5), with first column = 1

# Function to perform gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        gradient = (1 / m) * X.T @ (X @ theta - y)  # Compute gradient
        theta -= alpha * gradient  # Update theta
        cost = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)  # Compute cost
        J_history.append(cost)

    return theta, J_history

# Set hyperparameters
alpha = 0.1  # Learning rate
iterations_list = [10, 100, 1000]  # Iterations to evaluate

# Run gradient descent for different iterations
results = []
for n in iterations_list:
    theta_opt, cost_history = gradient_descent(X_norm, y, np.zeros(X_norm.shape[1]), alpha, n)
    max_theta = round(np.max(theta_opt), 2)  # Maximum theta value
    cost_rounded = round(cost_history[-1])  # Rounded cost function value
    results.append((n, cost_rounded, max_theta))

# Display results
print(results)