# Neural Network From Scratch - Interactive Visualization

## Overview

This repository demonstrates how to implement a neural network from scratch with **interactive 3Blue1Brown-inspired visualizations**. Watch neurons fire, weights change, and data flow through the network in real-time!

### Features

- **Interactive Network Visualization**: See neurons, connections, and weights with beautiful color-coded graphics
- **Real-time Training**: Watch the network learn step-by-step or in batches
- **Forward Propagation Animation**: See how data flows through each layer
- **Custom Input Testing**: Provide your own inputs and see predictions
- **Weight Evolution**: Visualize how weights change during training
- **Flexible Architecture**: Build networks with any number of layers and neurons

## Quick Start

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

```bash
# Clone the repository
git clone https://github.com/rajeshwar-vempaty/Neural-Network-Scratch-.git
cd Neural-Network-Scratch-

# Install dependencies
pip install numpy matplotlib ipywidgets jupyter

# For interactive widgets in Jupyter Lab
pip install ipympl
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Running the Interactive Visualization

#### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook Interactive_Neural_Network_Demo.ipynb
```

This opens an interactive dashboard where you can:
- Train the network with sliders to control learning rate
- Watch weights and activations update in real-time
- Test custom inputs and see predictions
- View training progress with loss and accuracy curves

#### Option 2: Python Script

```bash
# Run interactive matplotlib visualization
python interactive_neural_network.py

# Run forward propagation animation
python interactive_neural_network.py --animate
```

#### Option 3: Original Basic Notebook

```bash
jupyter notebook Neural_Network_Scratch.ipynb
```

## Project Structure

```
Neural-Network-Scratch-/
│
├── README.md                           # This file
├── interactive_neural_network.py       # Standalone interactive visualization
├── Interactive_Neural_Network_Demo.ipynb  # Full interactive Jupyter demo
└── Neural_Network_Scratch.ipynb        # Original basic implementation
```

## Interactive Features Guide

### 1. Network Visualization

The visualization uses a 3Blue1Brown-inspired color scheme:
- **Blue connections**: Positive weights (enhance signals)
- **Red connections**: Negative weights (inhibit signals)
- **Line thickness**: Weight magnitude (thicker = stronger)
- **Neuron brightness**: Activation level (brighter = higher activation)

### 2. Interactive Controls

| Control | Function |
|---------|----------|
| Learning Rate Slider | Adjust how fast the network learns |
| Sample Index Slider | View different training samples |
| Weight Layer Slider | Inspect weights between different layers |
| Train Button | Train for multiple epochs |
| Step Button | Train for a single step |
| Reset Button | Reinitialize all weights |
| Forward Pass | Test custom input values |

### 3. Step-by-Step Forward Propagation

Watch data flow through the network:
```python
from interactive_neural_network import NeuralNetwork, ForwardPropagationAnimator, NetworkVisualizer

# Create network
nn = NeuralNetwork([4, 6, 4, 1])

# Create visualizer and animator
viz = NetworkVisualizer(nn)
animator = ForwardPropagationAnimator(nn, viz)

# Animate forward pass
animator.animate_forward_pass([0.5, 0.8, 0.2, 0.9], interval=800)
```

### 4. Custom Network Architecture

Build your own network:
```python
from interactive_neural_network import NeuralNetwork, NetworkVisualizer

# Create a deep network
nn = NeuralNetwork(
    layer_sizes=[8, 16, 12, 8, 4, 1],  # Custom architecture
    activation='relu',                  # Options: 'sigmoid', 'relu', 'tanh'
    weight_init='xavier'                # Options: 'random', 'xavier', 'he'
)

# Visualize
viz = NetworkVisualizer(nn)
viz.create_interactive_visualization(X_data, y_data)
```

## Understanding the Visualization

### What the Colors Mean

| Element | Color | Meaning |
|---------|-------|---------|
| Connections | Blue | Positive weight (enhances signal) |
| Connections | Red | Negative weight (inhibits signal) |
| Neurons | Bright blue | High activation (neuron is "firing") |
| Neurons | Dark blue | Low activation (neuron is quiet) |
| Input layer edge | Purple | Input neurons |
| Output layer edge | Green | Output neurons |

### Training Curves

- **Loss Curve**: Should decrease over time (network is learning)
- **Accuracy Curve**: Should increase over time (better predictions)

## Examples

### Basic Training Loop

```python
from interactive_neural_network import NeuralNetwork, create_sample_data

# Create data
X, y = create_sample_data(n_samples=200, n_features=4)

# Create network
nn = NeuralNetwork([4, 8, 4, 1], activation='sigmoid')

# Train
for epoch in range(1000):
    loss, acc = nn.train_step(X, y, learning_rate=0.1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f}")
```

### Making Predictions

```python
import numpy as np

# After training
test_input = np.array([[0.5, 0.3, 0.8, 0.2]])
prediction = nn.forward(test_input)
print(f"Prediction: {prediction[0,0]:.4f}")
print(f"Class: {'1' if prediction >= 0.5 else '0'}")
```

## Key Concepts Demonstrated

1. **Forward Propagation**: How inputs are transformed layer by layer
2. **Backpropagation**: How errors are propagated back to update weights
3. **Gradient Descent**: How weights are adjusted to minimize loss
4. **Activation Functions**: How neurons decide whether to "fire"
5. **Weight Initialization**: Why starting weights matter
6. **Learning Rate**: How step size affects learning

## Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `ipywidgets`: Interactive controls (Jupyter)
- `ipympl`: Interactive matplotlib backend for Jupyter

## Contributing

Pull requests are welcome! Some ideas for contributions:

- Add more activation functions (LeakyReLU, Softmax, etc.)
- Implement regularization (L1, L2, Dropout)
- Add more optimizers (Adam, RMSprop, etc.)
- Create decision boundary visualizations
- Add convolutional layer visualization

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Inspired by [3Blue1Brown's Neural Network series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Built from scratch to understand the fundamentals
