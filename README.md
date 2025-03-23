# TDQN: Trading Deep Q-Network

## Overview
TDQN (Trading Deep Q-Network) is a deep reinforcement learning-based algorithmic trading model designed to optimize trading strategies using a modified Deep Q-Network (DQN). This implementation adapts DQN to handle financial time-series data, incorporating various deep learning techniques for improved stability, generalization, and performance.

This project draws significant inspiration from the following research papers:
- **"An Application of Deep Reinforcement Learning to Algorithmic Trading" by Thibaut Théate and Damien Ernst**
- **"Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review"**

## Model Architecture
The neural network follows a feedforward design with batch normalization and dropout layers to improve training stability:
```python
x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
output = self.fc5(x)
```

## Key Enhancements over Classical DQN
### 1. **Deep Neural Network (DNN) Architecture**
- The traditional convolutional neural network (CNN) used in DQN is replaced with a fully connected deep neural network, making it more suitable for processing financial time-series data.
- Leaky ReLU activation functions are used to prevent neuron dead zones and improve training dynamics.

### 2. **Double DQN Implementation**
- Reduces overestimation bias in Q-values, improving decision-making stability.
- Action selection and action evaluation are separated as proposed in van Hasselt et al. (2015).

### 3. **Optimized Training with ADAM**
- Replaces RMSProp with the ADAM optimizer, as introduced by Kingma and Ba (2015).
- Enhances training stability and speeds up convergence.

### 4. **Huber Loss for Robust Training**
- Unlike Mean Squared Error (MSE), Huber loss mitigates the impact of outliers.
- Prevents instability during updates by blending squared loss and absolute loss components.

### 5. **Gradient Clipping**
- Addresses exploding gradient issues, ensuring controlled updates to network weights.

### 6. **Xavier Initialization**
- Prevents vanishing/exploding gradients by keeping the variance of activations stable across layers.

### 7. **Batch Normalization Layers**
- Accelerates convergence and improves generalization by normalizing activations within layers.

### 8. **Regularization Techniques**
- Implements **Dropout**, **L2 Regularization**, and **Early Stopping** to reduce overfitting and improve model robustness.

### 9. **Preprocessing & Normalization**
- High-frequency noise reduction via low-pass filtering to extract meaningful trading patterns.
- Transformation of data to better capture market movements.

### 10. **Data Augmentation Techniques**
- Addresses limited data availability through:
  - **Signal shifting**
  - **Signal filtering**
  - **Artificial noise addition**
- These techniques generate synthetic data, improving model generalization.

## Installation & Dependencies
To run the TDQN algorithm, ensure you have the following dependencies installed:
```
Python 3.7.4  
Pytorch 1.5.0  
Tensorboard  
Gym  
Numpy  
Pandas  
Matplotlib  
Scipy  
Seaborn  
Statsmodels  
Requests  
Pandas-datareader  
TQDM  
Tabulate  
```

## Usage
To train the TDQN model:
```bash
python main.py -strategy STRATEGY -stock STOCK
```
with:
* STRATEGY being the name of the trading strategy (by default TDQN),
* STOCK being the name of the stock (by default Apple).

## References
1. Théate, T., & Ernst, D. (2020). "An Application of Deep Reinforcement Learning to Algorithmic Trading."
2. "Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review."

## Contribution
Aman K. Foujdar – Led the understanding and implementation of the TDQN architecture, worked on the RL environment, and fine-tuned the model for improved performance.

Keshav Kumar – Focused on the financial modeling aspect, translating the trading problem into a well-structured reinforcement learning framework.


