#%% md
# # State Management in BrainState
# 
# In dynamical brain modeling, time-varying state variables are often encountered, such as the membrane potential `V` of neurons or the firing rate `r` in firing rate models. **BrainState** provides the `State` data structure, which helps users intuitively define and manage computational states.
# 
# This tutorial provides a detailed introduction to state management in BrainState. By following this tutorial, you will learn:
# 
# - The basic concepts and fundamental usage of `State` objects
# - How to create `State` objects and use its subclasses: `ShortTermState`, `LongTermState`, `HiddenState`, and `ParamState`
# - State and JAX PyTree compatibility
# - How to use `StateTraceStack` to track State objects in your programs
# - Advanced state management patterns with `StateDictManager`
#%%
import jax.numpy as jnp
import brainstate
#%% md
# ## 1. Basic Concepts and Usage of State Objects
# 
# `State` is a key data structure in **BrainState** used to encapsulate state variables in models. These variables primarily represent values that change over time within the model.
# 
# ### Why States?
# 
# JAX is built on functional programming principles, which means:
# - All data is immutable by default
# - Functions cannot have side effects
# - State must be explicitly threaded through computations
# 
# This creates a challenge for neural network programming, where we naturally think in terms of mutable states (weights, neuron voltages, etc.). **BrainState's `State`** solves this by:
# 
# ✅ Providing a mutable interface for state variables  
# ✅ Automatically managing state updates during JAX transformations  
# ✅ Maintaining compatibility with JAX's functional paradigm  
# 
# ### Creating States
# 
# A `State` can wrap any Python data type, such as integers, floating-point numbers, arrays, `jax.Array`, or any of these encapsulated in dictionaries or lists. Unlike native Python data structures, the data within a `State` object remains mutable after program compilation.
#%%
# Create a simple State with an array
example = brainstate.State(jnp.ones(10))
example
#%% md
# ### States and PyTrees
# 
# `State` supports arbitrary [PyTree](https://jax.readthedocs.io/en/latest/working-with-pytrees.html) structures, which means you can encapsulate complex nested data structures within a `State` object. This is particularly useful for models with hierarchical state representations.
#%%
# State can hold complex PyTree structures
example2 = brainstate.State({'a': jnp.ones(3), 'b': jnp.zeros(4)})
example2
#%%
# State can also hold nested structures
complex_state = brainstate.State({
    'neurons': {
        'V': jnp.zeros(100),
        'u': jnp.zeros(100)
    },
    'synapses': {
        'g': jnp.zeros((100, 100)),
        'weights': jnp.ones((100, 100)) * 0.1
    }
})
print("Complex state structure:")
print(complex_state)
#%% md
# ### Accessing and Updating States
# 
# Users can access and modify state data through the `State.value` attribute.
#%%
# Access the state value
print("Current value:", example.value)
#%%
# Update the state value
example.value = brainstate.random.random(3)
print("Updated state:")
example
#%% md
# ### Core Features of State
# 
# **✅ Mutable after compilation**: State values can be updated even in JIT-compiled functions
# 
# **✅ Type and shape safety**: States enforce consistent types and shapes
# 
# **✅ Integration with JAX**: Works seamlessly with JAX transformations
# 
# ### Important Notes
# 
# ⚠️ **Static Data in JIT Compilation**: Any data not marked as a state variable will be treated as static during JIT compilation. Modifying static data in a JIT-compiled environment has no effect.
# 
# ⚠️ **Constraints on Modifying State Data**: When updating via the `value` attribute, the assigned data must have the same PyTree structure as the original. The shape and dtype should generally match, though some flexibility is allowed.
#%%
# Demonstrate tree structure checking
state = brainstate.ShortTermState(jnp.zeros((2, 3)))

with brainstate.check_state_value_tree():
    # This works - same tree structure
    state.value = jnp.zeros((2, 3))
    print("✓ Successfully updated state with matching structure")
    
    # This fails - different tree structure
    try:
        state.value = (jnp.zeros((2, 3)), jnp.zeros((2, 3)))
    except Exception as e:
        print(f"✗ Error: {e}")
#%% md
# ## 2. Subclasses of State
# 
# **BrainState** provides several subclasses of `State` to help organize different types of state variables in your models. While these subclasses are functionally identical to the base `State` class, they serve as semantic markers that:
# 
# - 📝 Improve code readability
# - 🔍 Enable selective filtering (e.g., finding all trainable parameters)
# - 🎯 Clarify the role of each state variable
# 
# ### Overview of State Types
# 
# | State Type | Purpose | Examples |
# |------------|---------|----------|
# | `ParamState` | Trainable parameters | Weights, biases |
# | `HiddenState` | Hidden activations | Membrane potentials, RNN hidden states |
# | `ShortTermState` | Transient states | Last spike time, current input |
# | `LongTermState` | Persistent states | Running averages, momentum |
# 
# ### 2.1 ParamState - Trainable Parameters
# 
# `ParamState` is used for trainable parameters in neural networks. These are the values that get updated during training via gradient descent.
#%%
# Example: Neural network parameters
weight = brainstate.ParamState(brainstate.random.randn(10, 10) * 0.1)
bias = brainstate.ParamState(jnp.zeros(10))

print("Weight:")
print(weight)
print("\nBias:")
print(bias)
#%% md
# ### 2.2 HiddenState - Hidden Activations
# 
# `HiddenState` encapsulates hidden activation variables in models. These states are updated during every simulation iteration and retained between iterations, representing the internal dynamics of the model.
#%%
# Example: Neuron membrane potential
V = brainstate.HiddenState(jnp.full(10, -70.0))  # Resting potential

# Example: RNN hidden state
h = brainstate.HiddenState(jnp.zeros((32, 128)))  # (batch_size, hidden_dim)

print("Membrane potential:")
print(V)
print("\nRNN hidden state:")
print(h)
#%% md
# ### 2.3 ShortTermState - Transient States
# 
# `ShortTermState` is designed for short-term, transient state variables. These states capture instantaneous values that may not carry long-term dependencies.
#%%
# Example: Last spike time
t_last_spike = brainstate.ShortTermState(jnp.full(10, -1e7))  # Very old time

# Example: Current input
current_input = brainstate.ShortTermState(jnp.zeros(10))

print("Last spike times:")
print(t_last_spike)
print("\nCurrent input:")
print(current_input)
#%% md
# ### 2.4 LongTermState - Persistent States
# 
# `LongTermState` is used for long-term state variables that accumulate information over many iterations. These are commonly used for statistics tracking and optimization algorithms.
#%%
# Example: Running mean for batch normalization
running_mean = brainstate.LongTermState(jnp.zeros(64))
running_var = brainstate.LongTermState(jnp.ones(64))

# Example: Optimizer momentum
momentum = brainstate.LongTermState(jnp.zeros((100, 100)))

print("Running mean:")
print(running_mean)
print("\nMomentum:")
print(momentum)
#%% md
# ### Practical Example: LIF Neuron Model
# 
# Let's see how different state types work together in a realistic model:
#%%
class LIFNeuron(brainstate.nn.Module):
    """Leaky Integrate-and-Fire neuron model."""
    
    def __init__(self, n_neurons, tau=10.0, V_th=1.0, V_reset=0.0):
        super().__init__()
        self.tau = tau
        self.V_th = V_th
        self.V_reset = V_reset
        
        # Hidden state: membrane potential (evolves continuously)
        self.V = brainstate.HiddenState(jnp.full(n_neurons, V_reset))
        
        # Short-term state: refractory period counter
        self.t_last_spike = brainstate.ShortTermState(jnp.full(n_neurons, -1e7))
        
        # Parameters: input weights
        self.w_in = brainstate.ParamState(brainstate.random.randn(n_neurons, n_neurons) * 0.1)
    
    def __call__(self, I_ext, t):
        # Membrane potential dynamics
        dV = (-self.V.value + I_ext) / self.tau
        self.V.value = self.V.value + dV
        
        # Spike generation
        spike = self.V.value >= self.V_th
        
        # Reset
        self.V.value = jnp.where(spike, self.V_reset, self.V.value)
        self.t_last_spike.value = jnp.where(spike, t, self.t_last_spike.value)
        
        return spike

# Create and test the neuron
neuron = LIFNeuron(n_neurons=5)
print("Initial state:")
print(f"V: {neuron.V.value}")

# Simulate
for t in range(20):
    I_ext = jnp.ones(5) * 0.2  # External current
    spikes = neuron(I_ext, t)
    if jnp.any(spikes):
        print(f"t={t}: Spikes at neurons {jnp.where(spikes)[0]}")
#%% md
# ## 3. State Tracking with StateTraceStack
# 
# `StateTraceStack` is a powerful debugging and introspection tool that tracks which `State` objects are accessed during program execution.
# 
# ### Why Track States?
# 
# - 🔍 **Debugging**: Understand which states are being read/written
# - 📊 **Profiling**: Identify state access patterns
# - 🎯 **Selective updates**: Apply operations only to specific state types
# - 🧪 **Testing**: Verify expected state interactions
# 
# ### Basic Usage
#%%
class Linear(brainstate.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(d_in, d_out) * 0.1)
        self.b = brainstate.ParamState(jnp.zeros(d_out))
        self.y = brainstate.HiddenState(jnp.zeros(d_out))
    
    def __call__(self, x):
        self.y.value = x @ self.w.value + self.b.value
        return self.y.value

model = Linear(2, 5)

# Track state access
with brainstate.StateTraceStack() as stack:
    output = model(brainstate.random.randn(2))
    
    # Get accessed states
    read_states = list(stack.get_read_states())
    write_states = list(stack.get_write_states())

print(f"States read: {len(read_states)}")
print(f"States written: {len(write_states)}")
#%% md
# ### Inspecting State Access
# 
# `StateTraceStack` provides four main methods:
# 
# - `get_read_states()`: Returns State objects that were read
# - `get_read_state_values()`: Returns the values of read states
# - `get_write_states()`: Returns State objects that were written
# - `get_write_state_values()`: Returns the values of written states
#%%
# Inspect read states
print("=== Read States ===")
for i, state in enumerate(read_states):
    print(f"{i+1}. {type(state).__name__}: shape={state.value.shape}")
#%%
# Inspect written states
print("=== Written States ===")
for i, state in enumerate(write_states):
    print(f"{i+1}. {type(state).__name__}: shape={state.value.shape if hasattr(state.value, 'shape') else 'N/A'}")
#%% md
# ## Summary
# 
# In this tutorial, you learned:
# 
# ✅ **States** provide mutable variables compatible with JAX  
# ✅ Different **state types** serve different purposes:  
#   - `ParamState` for trainable parameters  
#   - `HiddenState` for hidden activations  
#   - `ShortTermState` for transient states  
#   - `LongTermState` for persistent states  
# ✅ **StateTraceStack** tracks state access for debugging  
# ✅ States support **PyTree structures** for complex data  
# 
# ### Best Practices
# 
# 1. 🎯 Use specific state types (`ParamState`, etc.) rather than generic `State`
# 2. 📝 Keep state updates simple and explicit
# 3. 🔍 Use `StateTraceStack` for debugging unexpected behavior
# 4. ⚠️ Remember: only `State` values are mutable; regular variables are static
# 
# ### Next Steps
# 
# Continue with:
# - **Random Number Generation** - Learn about stateful random number generation
# - **Neural Network Modules** - Build complex models using states
# - **Program Transformations** - Use states with JIT, grad, and vmap