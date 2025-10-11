#%% md
# # Getting Started with BrainState
# 
# Welcome to **BrainState**! This tutorial will guide you through the basics of using BrainState, a state-based transformation system designed for brain modeling and neural network programming.
# 
# By the end of this tutorial, you will:
# - Understand what BrainState is and why it's useful
# - Know how to install and set up BrainState
# - Learn the core concepts and design philosophy
# - Build your first simple neural network with BrainState
#%% md
# ## What is BrainState?
# 
# **BrainState** is a powerful Python library built on top of JAX that provides:
# 
# - 🧠 **Stateful Programming Model**: Manage mutable states in a JAX-compatible way
# - 🚀 **High Performance**: Leverage JAX's JIT compilation, automatic differentiation, and vectorization
# - 🔧 **Modular Design**: Build complex models from simple, composable components
# - 🌐 **Brain Modeling**: Specialized tools for computational neuroscience and brain-inspired computing
# 
# BrainState bridges the gap between the functional programming paradigm of JAX and the intuitive, stateful programming style commonly used in neural network frameworks.
#%% md
# ## Installation and Environment Setup
# 
# ### Prerequisites
# 
# Before installing BrainState, ensure you have:
# - Python 3.9 or higher
# - pip package manager
# 
# ### Installing BrainState
# 
# The easiest way to install BrainState is via pip:
# 
# ```bash
# pip install brainstate --upgrade
# ```
# 
# ### Installing the Complete Ecosystem
# 
# For a complete brain modeling ecosystem, you can install BrainX, which bundles BrainState with other compatible packages:
# 
# ```bash
# pip install BrainX -U
# ```
# 
# This includes:
# - **brainstate**: Core state management and transformations
# - **brainunit**: Physical units and dimensional analysis
# - **braintools**: Optimization algorithms and utilities
# - **brainpy**: Spiking neural network modeling
# 
# ### Verifying Installation
# 
# Let's verify that BrainState is installed correctly:
#%%
import brainstate
import braintools
import jax.numpy as jnp

print(f"BrainState version: {brainstate.__version__}")
print(f"Installation successful! ✓")
#%% md
# ## Core Concepts Overview
# 
# BrainState is built around several key concepts that work together to enable stateful, high-performance neural network programming.
# 
# ### 1. State: Managing Mutable Variables
# 
# In pure functional programming (like JAX), all data is immutable. However, neural networks and brain models inherently involve mutable states (e.g., neuron membrane potentials, network weights).
# 
# **BrainState's `State`** provides a solution by wrapping mutable variables in a way that's compatible with JAX transformations.
#%%
# Creating a State object
voltage = brainstate.State(jnp.array([0.0, -70.0, -55.0]))
print("Initial voltage:", voltage.value)

# Updating the state
voltage.value = voltage.value + 10.0
print("Updated voltage:", voltage.value)
#%% md
# **Key Types of States:**
# 
# - `State`: Generic mutable state
# - `ParamState`: Trainable parameters (weights, biases)
# - `HiddenState`: Hidden activations (membrane potentials, hidden layer outputs)
# - `ShortTermState`: Temporary states (spike times, current values)
# - `LongTermState`: Long-term states (running statistics, momentum)
# 
# We'll explore these in detail in the next tutorial.
#%% md
# ### 2. Module: Building Blocks of Neural Networks
# 
# The `Module` class (actually `graph.Node`) is the base class for all neural network components in BrainState. It automatically manages states and provides a clean interface for building complex models.
#%%
class SimpleNeuron(brainstate.nn.Module):
    """A simple leaky integrate-and-fire neuron."""
    
    def __init__(self, threshold=1.0, reset=0.0, tau=10.0):
        super().__init__()
        self.threshold = threshold
        self.reset = reset
        self.tau = tau
        
        # Membrane potential is a hidden state
        self.V = brainstate.HiddenState(jnp.array(0.0))
    
    def __call__(self, I_input):
        """Update neuron state given input current."""
        # Leaky integration
        dV = (-self.V.value + I_input) / self.tau
        self.V.value = self.V.value + dV
        
        # Spike and reset
        spike = self.V.value >= self.threshold
        self.V.value = jnp.where(spike, self.reset, self.V.value)
        
        return spike

# Create and test the neuron
neuron = SimpleNeuron()
print("Initial voltage:", neuron.V.value)

# Simulate with input current
for t in range(20):
    spike = neuron(2.0)  # constant input
    if spike:
        print(f"Spike at time {t}! V={neuron.V.value}")
#%% md
# ### 3. Transform: JAX Transformations with States
# 
# BrainState provides state-aware versions of JAX transformations:
# 
# - `brainstate.transform.jit`: Just-in-time compilation
# - `brainstate.transform.grad`: Automatic differentiation
# - `brainstate.transform.vmap`: Vectorization (batching)
# - `brainstate.transform.scan`: Efficient loops
# 
# These transformations automatically handle state management for you.
#%%
# Reset neuron
neuron.V.value = jnp.array(0.0)

# Simulate with varying input
inputs = jnp.array([1.5, 2.0, 2.5, 3.0, 1.0] * 4)
spikes = brainstate.transform.for_loop(neuron, inputs)
print("Spike train:", spikes.astype(int))
#%% md
# ### 4. Random: Stateful Random Number Generation
# 
# BrainState provides a stateful random number generator that's compatible with JAX's functional random number generation while maintaining a simple, NumPy-like interface.
#%%
# Set random seed for reproducibility
brainstate.random.seed(42)

# Generate random numbers
uniform_samples = brainstate.random.rand(5)
normal_samples = brainstate.random.randn(5)

print("Uniform samples:", uniform_samples)
print("Normal samples:", normal_samples)
#%% md
# ## Hello World: Building Your First Neural Network
# 
# Let's build a simple feedforward neural network to classify handwritten digits. This example demonstrates the key concepts working together.
# 
# ### Step 1: Define the Network
#%%
class MLP(brainstate.nn.Module):
    """A simple multi-layer perceptron."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Initialize weights and biases as trainable parameters
        self.w1 = brainstate.ParamState(brainstate.random.randn(input_dim, hidden_dim) * 0.1)
        self.b1 = brainstate.ParamState(jnp.zeros(hidden_dim))
        
        self.w2 = brainstate.ParamState(brainstate.random.randn(hidden_dim, output_dim) * 0.1)
        self.b2 = brainstate.ParamState(jnp.zeros(output_dim))
    
    def __call__(self, x):
        """Forward pass through the network."""
        # Hidden layer with ReLU activation
        hidden = jnp.maximum(0, x @ self.w1.value + self.b1.value)
        
        # Output layer
        logits = hidden @ self.w2.value + self.b2.value
        
        return logits

# Create the network
brainstate.random.seed(0)
model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
print("Network created!")
print(f"Total parameters: {784*128 + 128 + 128*10 + 10:,}")
#%% md
# ### Step 2: Define Loss Function and Training Step
#%%
def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss."""
    # One-hot encode labels
    one_hot_labels = jnp.eye(10)[labels]
    
    # Compute log-softmax
    log_probs = logits - jnp.log(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True))
    
    # Compute loss
    loss = -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))
    return loss

def accuracy(logits, labels):
    """Compute classification accuracy."""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

def loss_fn(x, y):
    """Compute loss for the model."""
    logits = model(x)
    return cross_entropy_loss(logits, y)

#%%
# Generate dummy data for demonstration
brainstate.random.seed(42)
X_train = brainstate.random.randn(100, 784) * 0.1  # 100 samples
y_train = brainstate.random.randint(0, 10, 100)     # Random labels
#%%
# Create gradient function
param_states = brainstate.transform.StateFinder(loss_fn, brainstate.ParamState)(X_train, y_train)
grad_fn = brainstate.transform.grad(loss_fn, grad_states=param_states)
#%% md
# ### Step 3: Training Loop
#%%
optimizer = braintools.optim.SGD(1e-1)
_ = optimizer.register_trainable_weights(param_states)
#%%
@brainstate.transform.jit
def train_step(x, y):
    """Perform one training step."""
    # Compute gradients
    grads = grad_fn(x, y)
    
    # Update parameters using gradient descent
    optimizer.update(grads)
    
    # Compute metrics
    logits = model(x)
    loss = cross_entropy_loss(logits, y)
    acc = accuracy(logits, y)
    
    return loss, acc

# Training loop
print("Starting training...\n")
for epoch in range(10):
    loss, acc = train_step(X_train, y_train)
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1:2d}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

print("\nTraining complete!")
#%% md
# ### Step 4: Making Predictions
#%%
@brainstate.transform.jit
def predict(x):
    """Make predictions with the model."""
    logits = model(x)
    return jnp.argmax(logits, axis=-1)

# Generate test data
X_test = brainstate.random.randn(10, 784) * 0.1
predictions = predict(X_test)

print("Predictions on test data:")
print(predictions)
#%% md
# ## Key Takeaways
# 
# Congratulations! You've just built your first neural network with BrainState. Here are the key concepts we covered:
# 
# 1. **States** wrap mutable variables and make them compatible with JAX transformations
# 2. **Modules** (via `nn.Module`) provide a clean way to organize neural network components
# 3. **Transformations** like `jit` and `grad` work seamlessly with stateful code
# 4. **Random number generation** is stateful yet reproducible
# 
# ## What's Next?
# 
# Now that you understand the basics, continue with the following tutorials:
# 
# 1. **State Management** - Deep dive into different types of states and advanced state management techniques
# 2. **Random Number Generation** - Learn about BrainState's random number generation system
# 3. **Neural Network Modules** - Explore pre-built layers and learn to create custom modules
# 4. **Program Transformations** - Master JIT compilation, automatic differentiation, and vectorization
# 
# ## Additional Resources
# 
# - 📚 [BrainState Documentation](https://brainstate.readthedocs.io/)
# - 🌐 [BrainX Ecosystem](https://brainmodeling.readthedocs.io/)
# - 💻 [GitHub Repository](https://github.com/chaobrain/brainstate)
# - 🐛 [Issue Tracker](https://github.com/chaobrain/brainstate/issues)
# 
# Happy coding with BrainState! 🧠✨