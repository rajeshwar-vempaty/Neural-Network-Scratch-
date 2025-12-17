"""
Interactive Neural Network Visualization
=========================================
A 3Blue1Brown-inspired interactive visualization of neural networks from scratch.
Allows users to see neurons, layers, weights, and activations change in real-time.

Author: Enhanced from Neural_Network_Scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class NeuralNetwork:
    """
    A flexible neural network implementation from scratch.
    Supports arbitrary number of layers and neurons per layer.
    """

    def __init__(self, layer_sizes, activation='sigmoid', weight_init='xavier'):
        """
        Initialize the neural network.

        Parameters:
        -----------
        layer_sizes : list
            List of integers representing neurons in each layer.
            e.g., [4, 8, 6, 2] means 4 inputs, two hidden layers (8, 6), 2 outputs
        activation : str
            Activation function: 'sigmoid', 'relu', 'tanh'
        weight_init : str
            Weight initialization: 'random', 'xavier', 'he'
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_name = activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            if weight_init == 'xavier':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            elif weight_init == 'he':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5

            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Store activations and z values for visualization
        self.activations = []
        self.z_values = []

        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.weight_history = []

    def _activation(self, z):
        """Apply activation function."""
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        return z

    def _activation_derivative(self, a):
        """Compute derivative of activation function."""
        if self.activation_name == 'sigmoid':
            return a * (1 - a)
        elif self.activation_name == 'relu':
            return (a > 0).astype(float)
        elif self.activation_name == 'tanh':
            return 1 - a**2
        return np.ones_like(a)

    def forward(self, X, store=True):
        """
        Forward propagation through the network.

        Parameters:
        -----------
        X : ndarray
            Input data of shape (n_samples, n_features)
        store : bool
            Whether to store intermediate values for visualization

        Returns:
        --------
        ndarray : Output predictions
        """
        if store:
            self.activations = [X]
            self.z_values = []

        current = X
        for i in range(self.num_layers - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            if store:
                self.z_values.append(z)

            # Apply activation (use sigmoid for output layer in classification)
            if i == self.num_layers - 2:
                current = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Always sigmoid for output
            else:
                current = self._activation(z)

            if store:
                self.activations.append(current)

        return current

    def backward(self, X, y, learning_rate=0.01):
        """
        Backward propagation to update weights.

        Parameters:
        -----------
        X : ndarray
            Input data
        y : ndarray
            True labels
        learning_rate : float
            Learning rate for gradient descent
        """
        m = X.shape[0]

        # Compute output error
        output = self.activations[-1]
        delta = output - y

        # Backpropagate through layers
        deltas = [delta]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.activations[i])
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m

    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = (self.forward(X, store=False) >= 0.5).astype(int)
        return np.mean(predictions == y)

    def train_step(self, X, y, learning_rate=0.01):
        """Perform one training step."""
        output = self.forward(X, store=True)
        self.backward(X, y, learning_rate)
        loss = self.compute_loss(y, output)
        accuracy = self.compute_accuracy(X, y)
        return loss, accuracy

    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True, store_history=True):
        """
        Train the neural network.

        Parameters:
        -----------
        X : ndarray
            Training data
        y : ndarray
            Labels
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        verbose : bool
            Print progress
        store_history : bool
            Store training history for visualization
        """
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(epochs):
            loss, accuracy = self.train_step(X, y, learning_rate)

            if store_history:
                self.loss_history.append(loss)
                self.accuracy_history.append(accuracy)

                if epoch % 100 == 0:
                    self.weight_history.append([w.copy() for w in self.weights])

            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        if verbose:
            print(f"Final - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


class NetworkVisualizer:
    """
    3Blue1Brown-inspired neural network visualizer.
    Shows neurons, connections, weights, and activations in real-time.
    Optimized for fast incremental updates.
    """

    def __init__(self, network, figsize=(16, 10)):
        """
        Initialize the visualizer.

        Parameters:
        -----------
        network : NeuralNetwork
            The neural network to visualize
        figsize : tuple
            Figure size
        """
        self.network = network
        self.figsize = figsize

        # Color schemes (3Blue1Brown inspired)
        self.positive_color = '#3498db'  # Blue for positive
        self.negative_color = '#e74c3c'  # Red for negative
        self.neutral_color = '#2c3e50'   # Dark blue-gray
        self.background_color = '#1a1a2e' # Dark background
        self.text_color = '#ecf0f1'       # Light text
        self.highlight_color = '#f39c12'  # Orange highlight

        # Neuron positions
        self.neuron_positions = self._compute_neuron_positions()

        # Cached artists for incremental updates (performance optimization)
        self._connection_lines = []  # List of Line2D objects
        self._neuron_circles = []    # List of Circle patches
        self._neuron_texts = []      # List of Text objects
        self._highlight_circles = [] # List of highlight Circle patches
        self._artists_initialized = False
        self._cached_max_weight = None

    def _compute_neuron_positions(self):
        """Compute x, y positions for each neuron."""
        positions = []
        layer_sizes = self.network.layer_sizes
        max_neurons = max(layer_sizes)

        for layer_idx, n_neurons in enumerate(layer_sizes):
            layer_positions = []
            x = layer_idx / (len(layer_sizes) - 1) if len(layer_sizes) > 1 else 0.5

            # Center neurons vertically
            for neuron_idx in range(n_neurons):
                if n_neurons == 1:
                    y = 0.5
                else:
                    y = (neuron_idx / (n_neurons - 1)) * 0.8 + 0.1
                layer_positions.append((x, y))

            positions.append(layer_positions)

        return positions

    def _get_weight_color(self, weight, max_weight=None):
        """Get color based on weight value."""
        if max_weight is None or max_weight == 0:
            max_weight = 1

        normalized = np.clip(weight / max_weight, -1, 1)

        if normalized >= 0:
            # Interpolate from neutral to positive (blue)
            return mcolors.to_rgba(self.positive_color, alpha=0.3 + 0.7 * normalized)
        else:
            # Interpolate from neutral to negative (red)
            return mcolors.to_rgba(self.negative_color, alpha=0.3 + 0.7 * abs(normalized))

    def _get_activation_color(self, activation):
        """Get color based on activation value (0 to 1)."""
        # Create a gradient from dark to bright
        activation = np.clip(activation, 0, 1)

        # Color intensity based on activation
        r = 0.2 + 0.6 * activation
        g = 0.6 + 0.4 * activation
        b = 0.8 + 0.2 * activation

        return (r, g, b, 0.9)

    def _initialize_artists(self, ax, show_weights=True):
        """Initialize all matplotlib artists once for incremental updates."""
        self._connection_lines = []
        self._neuron_circles = []
        self._neuron_texts = []
        self._highlight_circles = []
        self._layer_labels = []

        neuron_radius = 0.035

        # Create connection lines
        if show_weights:
            for layer_idx in range(len(self.network.weights)):
                for i, pos1 in enumerate(self.neuron_positions[layer_idx]):
                    for j, pos2 in enumerate(self.neuron_positions[layer_idx + 1]):
                        line, = ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                       color='gray', linewidth=1, alpha=0.6, zorder=1)
                        self._connection_lines.append({
                            'line': line,
                            'layer_idx': layer_idx,
                            'from_idx': i,
                            'to_idx': j,
                            'pos1': pos1,
                            'pos2': pos2
                        })

        # Create neuron circles and texts
        for layer_idx, layer_positions in enumerate(self.neuron_positions):
            for neuron_idx, (x, y) in enumerate(layer_positions):
                # Highlight circle (initially invisible)
                highlight_circle = Circle((x, y), neuron_radius + 0.008,
                                        facecolor=self.highlight_color,
                                        edgecolor=self.highlight_color, linewidth=3,
                                        zorder=4, visible=False)
                ax.add_patch(highlight_circle)
                self._highlight_circles.append({
                    'circle': highlight_circle,
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx
                })

                # Neuron circle
                circle = Circle((x, y), neuron_radius, facecolor=(0.3, 0.5, 0.7, 0.9),
                               edgecolor='white', linewidth=2, zorder=5)
                ax.add_patch(circle)
                self._neuron_circles.append({
                    'circle': circle,
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx
                })

                # Value text
                text = ax.text(x, y, '', ha='center', va='center',
                              fontsize=8, color='white', fontweight='bold', zorder=6)
                self._neuron_texts.append({
                    'text': text,
                    'layer_idx': layer_idx,
                    'neuron_idx': neuron_idx
                })

        # Create layer labels (static, don't need updating)
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(self.network.layer_sizes) - 2)] + ['Output']
        for idx, name in enumerate(layer_names):
            x = idx / (len(self.network.layer_sizes) - 1) if len(self.network.layer_sizes) > 1 else 0.5
            ax.text(x, -0.02, name, ha='center', va='top', fontsize=11,
                   color=self.text_color, fontweight='bold')
            n_neurons = self.network.layer_sizes[idx]
            neuron_text = f'{n_neurons} neuron' if n_neurons == 1 else f'{n_neurons} neurons'
            ax.text(x, 1.02, neuron_text,
                   ha='center', va='bottom', fontsize=9, color=self.text_color, alpha=0.7)

        self._artists_initialized = True

    def _update_artists(self, show_weights=True, show_values=True, highlight_path=None):
        """Update existing artists with new values (fast incremental update)."""
        # Convert highlight_path to set for O(1) lookup
        highlight_set = set(highlight_path) if highlight_path else set()
        highlight_edges = set()
        if highlight_path:
            for k in range(len(highlight_path) - 1):
                highlight_edges.add((highlight_path[k], highlight_path[k+1]))

        # Update max weight (cache it for performance)
        max_weight = max(np.abs(w).max() for w in self.network.weights) if self.network.weights else 1

        # Update connection lines
        if show_weights:
            for conn in self._connection_lines:
                line = conn['line']
                layer_idx = conn['layer_idx']
                i, j = conn['from_idx'], conn['to_idx']

                weight = self.network.weights[layer_idx][i, j]
                color = self._get_weight_color(weight, max_weight)
                linewidth = 0.5 + 2.5 * abs(weight) / max_weight

                # Check highlight using set (O(1) lookup)
                edge = ((layer_idx, i), (layer_idx + 1, j))
                is_highlighted = edge in highlight_edges

                if is_highlighted:
                    line.set_color(self.highlight_color)
                    line.set_linewidth(linewidth + 2)
                    line.set_alpha(0.9)
                    line.set_zorder(2)
                else:
                    line.set_color(color)
                    line.set_linewidth(linewidth)
                    line.set_alpha(0.6)
                    line.set_zorder(1)

        # Update neuron circles and texts
        for i, (circle_data, text_data, highlight_data) in enumerate(
            zip(self._neuron_circles, self._neuron_texts, self._highlight_circles)):

            layer_idx = circle_data['layer_idx']
            neuron_idx = circle_data['neuron_idx']
            circle = circle_data['circle']
            text = text_data['text']
            highlight_circle = highlight_data['circle']

            # Get activation value
            if self.network.activations and layer_idx < len(self.network.activations):
                activation = self.network.activations[layer_idx]
                if activation.shape[0] > 0:
                    value = float(activation[0, neuron_idx]) if activation.ndim > 1 else float(activation[neuron_idx])
                    color = self._get_activation_color(value)
                else:
                    value = 0
                    color = self._get_activation_color(0)
            else:
                value = 0
                color = (0.3, 0.5, 0.7, 0.9)

            # Update circle color
            circle.set_facecolor(color)

            # Update highlight visibility
            is_highlighted = (layer_idx, neuron_idx) in highlight_set
            highlight_circle.set_visible(is_highlighted)

            # Update text
            if show_values and self.network.activations:
                text.set_text(f'{value:.2f}')
            else:
                text.set_text('')

    def draw_network(self, ax, show_weights=True, show_values=True,
                     input_values=None, highlight_path=None, force_redraw=False):
        """
        Draw the neural network on the given axes.
        Uses incremental updates for better performance.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        show_weights : bool
            Show weight connections
        show_values : bool
            Show activation values in neurons
        input_values : ndarray
            Input values to propagate through network
        highlight_path : list
            List of (layer, neuron) tuples to highlight
        force_redraw : bool
            Force complete redraw (slower but necessary after reset)
        """
        # If input values provided, do forward pass
        if input_values is not None:
            if input_values.ndim == 1:
                input_values = input_values.reshape(1, -1)
            self.network.forward(input_values, store=True)

        # Initialize or reinitialize artists if needed
        if not self._artists_initialized or force_redraw:
            ax.clear()
            ax.set_facecolor(self.background_color)
            ax.set_xlim(-0.15, 1.15)
            ax.set_ylim(-0.05, 1.05)
            ax.axis('off')
            self._initialize_artists(ax, show_weights)

        # Fast incremental update
        self._update_artists(show_weights, show_values, highlight_path)

    def reset_artists(self):
        """Reset cached artists (call after network reset)."""
        self._artists_initialized = False
        self._connection_lines = []
        self._neuron_circles = []
        self._neuron_texts = []
        self._highlight_circles = []

    def draw_weight_matrix(self, ax, layer_idx=0):
        """Draw weight matrix as a heatmap."""
        ax.clear()
        ax.set_facecolor(self.background_color)

        if layer_idx >= len(self.network.weights):
            return

        weights = self.network.weights[layer_idx]

        # Create heatmap
        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto',
                      vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())

        ax.set_title(f'Weights: Layer {layer_idx} → Layer {layer_idx + 1}',
                    color=self.text_color, fontsize=12)
        ax.set_xlabel(f'To Layer {layer_idx + 1}', color=self.text_color)
        ax.set_ylabel(f'From Layer {layer_idx}', color=self.text_color)
        ax.tick_params(colors=self.text_color)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight Value')

    def create_interactive_visualization(self, X_data=None, y_data=None):
        """
        Create a fully interactive visualization with sliders and controls.
        Optimized for fast, responsive interactions.

        Parameters:
        -----------
        X_data : ndarray, optional
            Training data for live training visualization
        y_data : ndarray, optional
            Labels for training
        """
        # Create figure with dark theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=self.figsize, facecolor=self.background_color)

        # Improved layout with better spacing
        # Main network visualization - larger area
        ax_network = fig.add_axes([0.03, 0.32, 0.58, 0.62])

        # Training curves - better positioned
        ax_accuracy = fig.add_axes([0.66, 0.72, 0.31, 0.22])
        ax_loss = fig.add_axes([0.66, 0.48, 0.31, 0.20])

        # Weight matrix visualization
        ax_weights = fig.add_axes([0.66, 0.32, 0.31, 0.13])

        # Sliders area - improved spacing
        ax_lr = fig.add_axes([0.15, 0.22, 0.35, 0.025])
        ax_layer = fig.add_axes([0.15, 0.17, 0.35, 0.025])
        ax_sample = fig.add_axes([0.15, 0.12, 0.35, 0.025])

        # Buttons - better organized
        ax_train_btn = fig.add_axes([0.58, 0.19, 0.13, 0.045])
        ax_step_btn = fig.add_axes([0.58, 0.12, 0.13, 0.045])
        ax_reset_btn = fig.add_axes([0.74, 0.19, 0.13, 0.045])
        ax_forward_btn = fig.add_axes([0.74, 0.12, 0.13, 0.045])

        # Status/progress text area
        ax_status = fig.add_axes([0.58, 0.06, 0.29, 0.04])
        ax_status.axis('off')
        status_text = ax_status.text(0.5, 0.5, '', ha='center', va='center',
                                     fontsize=10, color=self.highlight_color,
                                     transform=ax_status.transAxes)

        # Input text boxes
        ax_input = fig.add_axes([0.15, 0.05, 0.38, 0.035])

        # Create widgets with improved styling
        slider_lr = Slider(ax_lr, 'Learning Rate', 0.001, 0.5, valinit=0.1, valstep=0.005,
                          color=self.positive_color)
        slider_lr.label.set_color(self.text_color)
        slider_lr.valtext.set_color(self.text_color)

        max_layers = max(1, len(self.network.weights) - 1)
        slider_layer = Slider(ax_layer, 'Weight Layer', 0, max_layers, valinit=0, valstep=1,
                             color=self.positive_color)
        slider_layer.label.set_color(self.text_color)
        slider_layer.valtext.set_color(self.text_color)

        n_samples = X_data.shape[0] if X_data is not None else 1
        slider_sample = Slider(ax_sample, 'Sample Index', 0, max(0, n_samples - 1), valinit=0, valstep=1,
                              color=self.positive_color)
        slider_sample.label.set_color(self.text_color)
        slider_sample.valtext.set_color(self.text_color)

        btn_train = Button(ax_train_btn, 'Train 100 epochs', color='#2ecc71', hovercolor='#27ae60')
        btn_step = Button(ax_step_btn, 'Train 10 steps', color='#3498db', hovercolor='#2980b9')
        btn_reset = Button(ax_reset_btn, 'Reset Network', color='#e74c3c', hovercolor='#c0392b')
        btn_forward = Button(ax_forward_btn, 'Forward Pass', color='#9b59b6', hovercolor='#8e44ad')

        text_input = TextBox(ax_input, 'Custom Input: ',
                            initial=','.join(['0.5'] * self.network.layer_sizes[0]))

        # Cached line objects for training curves (performance optimization)
        loss_line = None
        accuracy_line = None

        # State
        state = {
            'epoch': 0,
            'learning_rate': 0.1,
            'is_training': False
        }

        def update_status(message):
            """Update status text."""
            status_text.set_text(message)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        def update_display(force_redraw=False, update_graphs=True):
            """Update all visualizations with optimized rendering."""
            nonlocal loss_line, accuracy_line

            # Get current sample
            sample_idx = int(slider_sample.val)
            if X_data is not None and sample_idx < X_data.shape[0]:
                current_input = X_data[sample_idx:sample_idx+1]
            else:
                current_input = np.array([[0.5] * self.network.layer_sizes[0]])

            # Draw network (uses incremental updates internally)
            self.draw_network(ax_network, input_values=current_input,
                            show_weights=True, show_values=True, force_redraw=force_redraw)
            ax_network.set_title(f'Neural Network | Epoch: {state["epoch"]}',
                               color=self.text_color, fontsize=14, fontweight='bold', pad=10)

            # Draw weight matrix (only when needed)
            if update_graphs:
                layer_idx = int(slider_layer.val)
                self.draw_weight_matrix(ax_weights, layer_idx)

            # Update training curves efficiently
            if update_graphs:
                # Update loss plot
                ax_loss.clear()
                ax_loss.set_facecolor(self.background_color)
                if self.network.loss_history:
                    ax_loss.plot(self.network.loss_history, color='#e74c3c', linewidth=1.5)
                    current_loss = self.network.loss_history[-1]
                    ax_loss.set_title(f'Loss: {current_loss:.4f}', color=self.text_color, fontsize=10)
                else:
                    ax_loss.set_title('Loss', color=self.text_color, fontsize=10)
                ax_loss.set_xlabel('Epoch', color=self.text_color, fontsize=8)
                ax_loss.tick_params(colors=self.text_color, labelsize=7)
                ax_loss.grid(True, alpha=0.2)

                # Update accuracy plot
                ax_accuracy.clear()
                ax_accuracy.set_facecolor(self.background_color)
                if self.network.accuracy_history:
                    ax_accuracy.plot(self.network.accuracy_history, color='#2ecc71', linewidth=1.5)
                    current_acc = self.network.accuracy_history[-1]
                    ax_accuracy.set_title(f'Accuracy: {current_acc:.2%}', color=self.text_color, fontsize=10)
                else:
                    ax_accuracy.set_title('Accuracy', color=self.text_color, fontsize=10)
                ax_accuracy.tick_params(colors=self.text_color, labelsize=7)
                ax_accuracy.grid(True, alpha=0.2)
                ax_accuracy.set_ylim(0, 1.05)

            fig.canvas.draw_idle()

        def on_train(event):
            """Train for 100 epochs with batched updates."""
            if X_data is not None and y_data is not None:
                state['is_training'] = True
                lr = slider_lr.val
                total_epochs = 100
                update_interval = 10  # Update display every N epochs

                update_status(f'Training... 0/{total_epochs}')
                fig.canvas.flush_events()

                for i in range(total_epochs):
                    loss, acc = self.network.train_step(X_data, y_data, lr)
                    self.network.loss_history.append(loss)
                    self.network.accuracy_history.append(acc)
                    state['epoch'] += 1

                    # Update display at intervals for smoother performance
                    if (i + 1) % update_interval == 0:
                        update_status(f'Training... {i+1}/{total_epochs} | Loss: {loss:.4f}')
                        update_display(update_graphs=True)
                        fig.canvas.flush_events()

                state['is_training'] = False
                update_status(f'Done! Final Loss: {loss:.4f}, Acc: {acc:.2%}')
                update_display()

        def on_step(event):
            """Train for 10 steps."""
            if X_data is not None and y_data is not None:
                lr = slider_lr.val
                for _ in range(10):
                    loss, acc = self.network.train_step(X_data, y_data, lr)
                    self.network.loss_history.append(loss)
                    self.network.accuracy_history.append(acc)
                    state['epoch'] += 1
                update_status(f'Step complete | Loss: {loss:.4f}, Acc: {acc:.2%}')
                update_display()

        def on_reset(event):
            """Reset the network."""
            layer_sizes = self.network.layer_sizes
            activation = self.network.activation_name
            self.network.__init__(layer_sizes, activation=activation)
            self.reset_artists()  # Reset cached artists
            state['epoch'] = 0
            update_status('Network reset!')
            update_display(force_redraw=True)

        def on_forward(event):
            """Perform forward pass with custom input."""
            try:
                input_text = text_input.text.strip()
                values = [float(x.strip()) for x in input_text.split(',')]

                expected_inputs = self.network.layer_sizes[0]
                if len(values) != expected_inputs:
                    update_status(f'Error: Expected {expected_inputs} values, got {len(values)}')
                    return

                input_arr = np.array([values])
                self.draw_network(ax_network, input_values=input_arr,
                                 show_weights=True, show_values=True)
                output = self.network.activations[-1][0]
                output_str = ', '.join([f'{v:.4f}' for v in output])
                ax_network.set_title(f'Forward Pass | Output: {output_str}',
                                   color=self.text_color, fontsize=14, fontweight='bold', pad=10)
                update_status(f'Output: {output_str}')
                fig.canvas.draw_idle()
            except ValueError as e:
                update_status(f'Error: Invalid input format. Use comma-separated numbers.')
            except Exception as e:
                update_status(f'Error: {str(e)[:50]}')

        def on_slider_change(val):
            """Handle slider changes with debouncing."""
            if not state['is_training']:
                update_display(update_graphs=True)

        # Connect callbacks
        btn_train.on_clicked(on_train)
        btn_step.on_clicked(on_step)
        btn_reset.on_clicked(on_reset)
        btn_forward.on_clicked(on_forward)
        slider_lr.on_changed(on_slider_change)
        slider_layer.on_changed(on_slider_change)
        slider_sample.on_changed(on_slider_change)

        # Initial display
        update_display(force_redraw=True)
        update_status('Ready! Click "Train" to start.')

        # Add title
        fig.suptitle('Interactive Neural Network Visualization',
                    color=self.text_color, fontsize=16, fontweight='bold', y=0.98)

        plt.show()
        return fig


class ForwardPropagationAnimator:
    """
    Animates the forward propagation process step by step,
    showing how data flows through the network.
    """

    def __init__(self, network, visualizer):
        """
        Initialize the animator.

        Parameters:
        -----------
        network : NeuralNetwork
            The neural network
        visualizer : NetworkVisualizer
            The visualizer instance
        """
        self.network = network
        self.visualizer = visualizer

    def animate_forward_pass(self, input_values, interval=300):
        """
        Animate forward propagation step by step.
        Optimized for smooth animation performance.

        Parameters:
        -----------
        input_values : ndarray
            Input values to propagate
        interval : int
            Milliseconds between frames (default: 300ms for smoother animation)
        """
        if input_values.ndim == 1:
            input_values = input_values.reshape(1, -1)

        # Do forward pass to get all activations
        self.network.forward(input_values, store=True)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.visualizer.background_color)

        # Animation state
        state = {'layer': 0, 'neuron': 0, 'initialized': False}

        # Store reference to info text for updating (avoids text accumulation)
        info_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', va='bottom',
                           fontsize=12, color=self.visualizer.text_color,
                           bbox=dict(boxstyle='round', facecolor=self.visualizer.highlight_color, alpha=0.8))

        def get_highlight_path(layer, neuron):
            """Get path to highlight for current state."""
            if layer == 0:
                return [(0, neuron)]

            path = []
            # Add connections from previous layer
            for prev_neuron in range(self.network.layer_sizes[layer - 1]):
                path.append((layer - 1, prev_neuron))
            path.append((layer, neuron))
            return path

        def animate(frame):
            layer = state['layer']
            neuron = state['neuron']

            # Get current highlight path
            path = get_highlight_path(layer, neuron)

            # Draw network with highlight (first frame initializes, rest updates)
            force_redraw = not state['initialized']
            self.visualizer.draw_network(ax, highlight_path=path,
                                        input_values=input_values,
                                        show_weights=True, show_values=True,
                                        force_redraw=force_redraw)
            state['initialized'] = True

            # Update info text (reuse existing text object)
            if layer < len(self.network.layer_sizes):
                activation = self.network.activations[layer][0, neuron] if layer < len(self.network.activations) else 0
                info_text.set_text(f'Layer {layer} | Neuron {neuron} | Activation: {activation:.4f}')

            # Update state for next frame
            state['neuron'] += 1
            if state['neuron'] >= self.network.layer_sizes[state['layer']]:
                state['neuron'] = 0
                state['layer'] += 1

            return [info_text]

        # Calculate total frames
        total_frames = sum(self.network.layer_sizes)

        ani = FuncAnimation(fig, animate, frames=total_frames,
                          interval=interval, blit=False, repeat=False)

        plt.show()
        return ani


def create_sample_data(n_samples=200, n_features=4, random_state=42):
    """
    Create sample binary classification data for demonstration.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    random_state : int
        Random seed

    Returns:
    --------
    X, y : ndarrays
        Features and labels
    """
    np.random.seed(random_state)

    # Generate two clusters
    n_per_class = n_samples // 2

    # Class 0: centered around negative values
    X0 = np.random.randn(n_per_class, n_features) * 0.5 - 1

    # Class 1: centered around positive values
    X1 = np.random.randn(n_per_class, n_features) * 0.5 + 1

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class).reshape(-1, 1)

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    # Normalize to 0-1 range for better visualization
    X = (X - X.min()) / (X.max() - X.min())

    return X, y


def demo_interactive():
    """Run an interactive demonstration."""
    print("=" * 60)
    print("  Interactive Neural Network Visualization")
    print("  Inspired by 3Blue1Brown")
    print("=" * 60)
    print()

    # Create sample data
    print("Creating sample data...")
    X, y = create_sample_data(n_samples=200, n_features=4)
    print(f"  - {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  - Binary classification task")
    print()

    # Create network
    print("Creating neural network...")
    network = NeuralNetwork(layer_sizes=[4, 6, 4, 1], activation='sigmoid')
    print(f"  - Architecture: {network.layer_sizes}")
    print(f"  - Activation: {network.activation_name}")
    print()

    # Create visualizer
    print("Starting interactive visualization...")
    print()
    print("Controls:")
    print("  - 'Train 100 epochs': Train the network for 100 epochs")
    print("  - 'Train 1 step': Train for a single step (see gradual learning)")
    print("  - 'Reset Network': Reinitialize all weights")
    print("  - 'Forward Pass': Run forward pass with custom input")
    print("  - Sliders: Adjust learning rate, view different weight matrices")
    print()

    visualizer = NetworkVisualizer(network)
    visualizer.create_interactive_visualization(X, y)


def demo_animation():
    """Run a forward propagation animation demonstration."""
    print("=" * 60)
    print("  Forward Propagation Animation")
    print("=" * 60)
    print()

    # Create a simple network
    network = NeuralNetwork(layer_sizes=[3, 4, 2, 1], activation='sigmoid')
    visualizer = NetworkVisualizer(network)
    animator = ForwardPropagationAnimator(network, visualizer)

    # Create sample input
    input_values = np.array([0.5, 0.8, 0.2])

    print("Animating forward pass...")
    print(f"Input: {input_values}")
    print()

    animator.animate_forward_pass(input_values, interval=800)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--animate':
        demo_animation()
    else:
        demo_interactive()
