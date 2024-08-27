import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Calculate the degrees of each node
node_degrees = data.edge_index[0].bincount()

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(node_degrees.numpy(), bins=range(1, node_degrees.max().item() + 2), align='left')
plt.xlabel('Node degree')
plt.ylabel('Number of nodes')
plt.title('Node Degree Distribution')
plt.grid(True)
plt.show()


from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt

numbers = Counter(degrees)


numbers = Counter(degrees)


