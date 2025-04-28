from flax import nnx        # neural network package
import jax

# class for ANN creation with one hidden layer with n nodes and relu activiation
class ANN_1_n_1(nnx.Module):
    def __init__(self, n, scale, seed: int):
        # Generate the main PRNG key from the seed
        rng = jax.random.PRNGKey(seed)

        # Split the PRNG key to get separate keys for each layer
        rng_linear, rng_linear_out = jax.random.split(rng)

        # Create rngs for each layer
        rngs_linear = nnx.Rngs({'params': rng_linear})
        rngs_linear_out = nnx.Rngs({'params': rng_linear_out})

        self.linear = nnx.Linear(1, n, kernel_init=self.custom_init(scale),  rngs=rngs_linear)
        self.linear_out = nnx.Linear(n, 1, kernel_init=self.custom_init(scale),  rngs=rngs_linear_out)

    def custom_init(self, scale: float):
        def init(key, shape,dtype):
            return jax.random.normal(key, shape, dtype=dtype) * scale
        return init

    def __call__(self, x):
        x = nnx.tanh((self.linear(x)))
        return self.linear_out(x)

class ANN_1_n_n_1(nnx.Module):
    def __init__(self, n1, n2, scale, seed: int):
        # Generate the main PRNG key from the seed
        rng = jax.random.PRNGKey(seed)

        # Split the PRNG key to get separate keys for each layer
        rng_linear_1, rng_linear_2, rng_linear_out = jax.random.split(rng, 3)

        # Create rngs for each layer
        rngs_linear_1 = nnx.Rngs({'params': rng_linear_1})
        rngs_linear_2 = nnx.Rngs({'params': rng_linear_2})
        rngs_linear_out = nnx.Rngs({'params': rng_linear_out})

        # Define the layers: two hidden layers and one output layer
        self.linear_1 = nnx.Linear(1, n1, kernel_init=self.custom_init(scale), rngs=rngs_linear_1)  # Hidden Layer 1
        self.linear_2 = nnx.Linear(n1, n2, kernel_init=self.custom_init(scale), rngs=rngs_linear_2)  # Hidden Layer 2
        self.linear_out = nnx.Linear(n2, 1, kernel_init=self.custom_init(scale), rngs=rngs_linear_out)  # Output Layer

    def custom_init(self, scale: float):
        def init(key, shape,dtype):
            return jax.random.normal(key, shape, dtype=dtype) * scale
        return init

    def __call__(self, x):
        # Forward pass: Apply ReLU activation after each hidden layer
        x = nnx.relu(self.linear_1(x))  # Hidden layer 1
        x = nnx.tanh(self.linear_2(x))  # Hidden layer 2
        return self.linear_out(x)  # Output layer

'''
#Short overview how to save and load model paramters

import orbax.checkpoint as ocp

ckpt_dir = ocp.test_utils.erase_and_create_empty(os.path.abspath('./my-checkpoints/'))

_, state = nnx.split(controler)
nnx.display(state)

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir / 'state', state)

import jax.tree as tree

# Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
abstract_model = nnx.eval_shape(lambda: ANN_1_n_1(8, 1, 1))
graphdef, abstract_state = nnx.split(abstract_model)
print('The abstract NNX state (all leaves are abstract arrays):')
nnx.display(abstract_state)

state_restored = checkpointer.restore(ckpt_dir / 'state', abstract_state)
tree.map(np.testing.assert_array_equal, state, state_restored)
print('NNX State restored: ')
nnx.display(state_restored)

# The model is now good to use!
model = nnx.merge(graphdef, state_restored)

'''
