# **Linear Regression with PyTorch**

## **Project Overview**

This Python script implements **Linear Regression from scratch** using the **PyTorch** library. The goal is to predict multiple output values (e.g., apples and oranges) based on a set of input features (e.g., temperature, rainfall, and humidity) using a simple linear regression model.

The script demonstrates:
- Loading and preparing data.
- Defining a linear regression model.
- Calculating **Mean Squared Error (MSE)** as the loss function.
- Performing backpropagation and updating model weights using **Gradient Descent**.

---

## **Features**
- **Simple Linear Regression**: Models the relationship between input features (e.g., temperature, rainfall) and output targets (e.g., apples, oranges).
- **Training from Scratch**: Implements basic training using PyTorch without any high-level APIs.
- **Mean Squared Error (MSE)**: Calculates the loss for model predictions.
- **Backpropagation**: Updates weights and biases using gradient descent.
- **Training Loop**: Runs for 100 epochs, adjusting weights to minimize the loss.

---

## **Installation Instructions**

### **Prerequisites**

- Python 3.x
- PyTorch

### **Install PyTorch**

To run the script, you need to install the PyTorch library. You can do this with the following command:

```bash
pip install torch
```

---

## **Usage**

### **Step 1**: Clone or Download the Script

You can clone this repository or download the script to your local machine.

### **Step 2**: Install Dependencies

Run the following command in your terminal to install the necessary dependencies:

```bash
pip install torch
```

### **Step 3**: Run the Script

Execute the Python script with the following command:

```bash
python linear_regression_with_pytorch.py
```

The model will train for 100 epochs, and you will see the predictions and loss printed after each update.

---

## **Code Breakdown**

### **1. Data Preparation**

The input data (`inputs`) represents the features (e.g., temperature, rainfall, humidity), and the target data (`targets`) represents the actual values to predict (e.g., apples and oranges).

```python
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
```

### **2. Model Definition**

The model is defined as a linear equation:  
\[
y = X \cdot W^T + b
\]
Where:
- \(X\) is the input matrix.
- \(W\) is the weight matrix (randomly initialized).
- \(b\) is the bias vector (randomly initialized).

```python
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
```

### **3. Loss Function**

The loss function used is **Mean Squared Error (MSE)**, which calculates the difference between predicted values and true values.

```python
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()
```

### **4. Gradient Descent and Backpropagation**

In each training step, the gradients of the loss are computed and used to update the weights and biases. The learning rate is set to `1e-5`.

```python
w.grad.zero_()
b.grad.zero_()
w -= w.grad * 1e-5
b -= b.grad * 1e-5
```

### **5. Training Loop**

The model is trained for **100 epochs**. For each epoch:
- Predictions are made.
- The loss is calculated.
- Gradients are computed.
- Weights are updated.

```python
for i in range(100):
    pred = model(inputs)
    loss = mse(pred, targets)
    loss.backward()

    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5

    w.grad.zero_()
    b.grad.zero_()
```

### **6. Final Output**

After 100 epochs, the final predictions and the loss are printed.

```python
pred = model(inputs)
loss = mse(pred, targets)
print(loss)
```

---

## **Expected Output**

Upon running the script, you will see output like:

```
Initial Weights: tensor([[ 0.3800, -0.2764,  0.2487],
                          [-1.2553,  0.6790, -0.4033]])
Initial Bias: tensor([ 0.1883, -0.2632])

Predictions: tensor([...])
Initial Loss: tensor([...])

Training for 100 epochs...
Final Loss: tensor([...])
```

---

## **Technical Notes**

- **Model Optimization**: This script uses **Gradient Descent** with a small learning rate (`1e-5`) to minimize the loss and adjust the weights and biases.
- **Backpropagation**: PyTorch automatically computes the gradients via `loss.backward()` and applies them to update the model parameters.
- **Data Handling**: All input data and target data are converted to PyTorch tensors to leverage GPU acceleration if available.

---

## **Conclusion**

This project demonstrates the fundamentals of linear regression using PyTorch. It teaches the basic concepts of model training, loss function computation, backpropagation, and gradient descent in a simple and understandable way.


### **License**
This project is open-source and available under the [MIT License](LICENSE).

---

This README is structured with **clear sections**, making it easy for someone to understand what the project is about and how to use the code.
