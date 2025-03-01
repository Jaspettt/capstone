import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Example input (flattened image vectors)
X = np.array([[0.25, 0.5, 0.75, 1.0],  # Dog image example (simplified)
              [0.1, 0.3, 0.5, 0.7]])  # Cat image example (simplified)

# Example output (1 for dog, 0 for cat)
y = np.array([[1], [0]])

# Network architecture
input_layer_size = 4  # Number of features
hidden_layer1_size = 7  # First hidden layer neurons
hidden_layer2_size = 5  # Second hidden layer neurons
hidden_layer3_size = 3  # Third hidden layer neurons
output_layer_size = 1  # Binary classification

# Fixed weights and biases
W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
               [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])

W2 = np.array([[0.2, 0.3, 0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9, 1.0, 1.1],
               [1.2, 1.3, 1.4, 1.5, 1.6],
               [1.7, 1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5, 2.6],
               [2.7, 2.8, 2.9, 3.0, 3.1],
               [3.2, 3.3, 3.4, 3.5, 3.6]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

W3 = np.array([[0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0],
               [1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6]])
b3 = np.array([[0.1, 0.2, 0.3]])

W4 = np.array([[0.2], [0.3], [0.4]])
b4 = np.array([[0.1]])

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    
    z2 = #YOUR CODE HERE
    a2 = #YOUR CODE HERE
    
    z3 = #YOUR CODE HERE
    a3 = #YOUR CODE HERE
    
    z4 = #YOUR CODE HERE
    a4 = #YOUR CODE HERE
    
    # Compute error
    error = y - a4
    
    # Backpropagation
    d_a4 = error * sigmoid_derivative(a4)
    d_W4 = np.dot(a3.T, d_a4) * learning_rate
    d_b4 = np.sum(d_a4, axis=0, keepdims=True) * learning_rate
    
    d_a3 = #YOUR CODE HERE
    d_W3 = #YOUR CODE HERE
    d_b3 = #YOUR CODE HERE
    
    d_a2 = #YOUR CODE HERE
    d_W2 = #YOUR CODE HERE
    d_b2 = #YOUR CODE HERE
    
    d_a1 = #YOUR CODE HERE
    d_W1 = #YOUR CODE HERE
    d_b1 = #YOUR CODE HERE
    
    # Update weights and biases
    W4 += d_W4
    b4 += d_b4
    W3 += d_W3
    b3 += d_b3
    W2 += d_W2
    b2 += d_b2
    W1 += d_W1
    b1 += d_b1
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
y_pred = a4
print("Final Predictions:", y_pred)

