# Distributed Composite Optimization (DCO)
Distributed Composite Optimization (DCO) is a Python package for solving composite optimization problems of the form

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{n} \sum_{i=1}^n f_i(x) + g(x),
$$

where each $f_i: \mathbb{R}^d \rightarrow \mathbb{R}$ is a smooth loss function associated with a local dataset or agent, and $g(x)$ is a (possibly non-smooth) regularization term. DCO enables efficient and robust distributed optimization across multiple nodes, making it suitable for federated learning, multi-agent systems, and large-scale machine learning tasks.

This package contains the experimental code for the paper *A Unified Framework for Robust Distributed Optimization under Bounded Disturbances*.

## Features

- Support for distributed and parallel optimization
- Modular and extensible architecture
- Formula-style API based on NumPy and autograd (JAX), allowing you to define and deploy distributed optimization algorithms across multiple machines as naturally as writing mathematical expressions

## Installation
Install via pip:

```bash
pip install git+https://github.com/rui-huang-opt/dco.git
```

Or, for development:

```bash
git clone https://github.com/rui-huang-opt/dco.git
cd dco
pip install -e .
```

## Usage Example

The typical workflow of DCO consists of two main steps:

1. **Defining the network topology**: Specify the communication structure among nodes in the distributed system.
2. **Defining the local problem at each node**: Set up the local objective function and related parameters for each node individually.

The following example demonstrates distributed ridge regression, where each node solves a local least squares problem with $\ell_2$ regularization:

$$
\min_{x \in \mathbb{R}^d} \quad \frac{1}{4} \sum_{i = 1}^4 (u_i^\top x - v_i)^2 + \rho \| x \|^2.
$$

The distributed ridge regression problem can be efficiently addressed using the [`EXTRA (EXact firsT-ordeR Algorithm)`](https://epubs.siam.org/doi/abs/10.1137/14096668X):

$$
\mathbf{x}^{k + 2} = (I + W) \mathbf{x}^{k + 1} - \frac{I + W}{2} \mathbf{x}^k - \gamma [\nabla f(\mathbf{x}^{k + 1}) - \nabla f(\mathbf{x}^{k})],
$$

where $W$ is a symmetric mixing matrix determined by the network topology.

Below are code templates for both steps

### 1. Specify the network topology (mixing matrix $W$) on the server

The server only assists nodes in establishing connections to form an undirected graph network. It does not participate in communication during computation.
For more details, please refer to the [`topolink`](https://github.com/rui-huang-opt/topolink) repository.

```python
import numpy as np
from logging import basicConfig, INFO
from topolink import Graph

basicConfig(level=INFO)

L = np.array([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]])
W = np.eye(4) - L * 0.2

graph = Graph.from_mixing_matrix(W)

graph.deploy()

```

### 2. Define the local optimization problem at each node
Each node specifies its own data and parameters, such as the local feature vector `u_i`, target value `v_i`, regularization parameter `rho`, and the optimization step size.
The local objective function `f_i(x_i)` is defined using these parameters as

$$
f_i(x_i) = (u_i^\top x_i - v_i)^2 + \rho \| x_i \|^2.
$$

The `LocalObjective` class allows you to specify the smooth part of the objective, and you can set the `g_type` parameter to include different types of regularization, such as `"zero"` (no regularization, default value), `"l1"` (L1 regularization), or others as needed.
The `Optimizer` class is then used to set up and solve the optimization problem.

```python
# Configure logging to display INFO level messages
from logging import basicConfig, INFO

basicConfig(level=INFO)

# Distributed optimization
node_id = ...  # "1", "2", "3", or "4"
u_i = ...
v_i = ...
rho = ...
dimension = ...
step_size = ...


def f_i(x_i: NDArray[np.float64]) -> NDArray[np.float64]:
    return (u_i @ x_i - v_i) ** 2 + rho * x_i @ x_i


local_obj = LocalObjective(dimension, f_i)  # LocalObjective(dimension, f_i, g_type="zero")
optimizer = Optimizer.create(node_id, local_obj, step_size, algorithm="EXTRA")
optimizer.solve_sync(max_iter=500)

print(f"Solution for node {node_id}: {optimizer.x_i}")
```

> **Note:**  
> If you do not specify the `server_address` parameter when creating the optimizer, you will be prompted to enter the graph server address after running the script.  
>  
> **Terminal output example:**  
> `Please enter the server address (IP:Port):`
>  
> If you set the logging level to `INFO` on the server, the server will print its own address in the terminal.

### Running Distributed Algorithms with Multiple Processes on a Single Machine

If you do not have access to multiple machines, you can still experiment with and test distributed optimization algorithms by launching multiple processes on a single machine. Each process acts as an independent node and communicates with others via network ports.
For implementation details and configuration examples, please refer to the sample code in the [`examples/notebooks`](./examples/notebooks/) directory.

## Documentation

> **Note:** The detailed documentation is still in progress. You can refer to the [`examples`](./examples/) directory for sample code and additional usage information.

## License

This project is licensed under the MIT License.