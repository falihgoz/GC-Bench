import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np


class APT_Dummy(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Number of nodes, features, and classes
        num_nodes = 500
        num_features = 30
        num_classes = 2

        # Create random features, labels, and edges
        nodes = torch.randn((num_nodes, num_features), dtype=torch.float)
        labels = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)

        # Generate random edges
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        edge_index = torch.tensor(
            np.random.randint(0, num_nodes, (2, num_edges)), dtype=torch.long
        )

        # Compute adjacency matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocoo()
        adj = torch.sparse_coo_tensor(
            indices=torch.tensor([adj.row, adj.col]),
            values=torch.tensor(adj.data),
            size=(num_nodes, num_nodes),
            dtype=torch.float,
        )

        # Split indices into train, val, and test sets
        all_indices = torch.arange(num_nodes)
        idx_train, idx_test = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )
        idx_train, idx_val = train_test_split(
            idx_train, test_size=0.25, random_state=42
        )

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # Create the data object
        data = Data(
            x=nodes,
            y=labels,
            edge_index=edge_index,
            adj=adj,  # Include adjacency matrix
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Save the processed data
        torch.save(self.collate([data]), self.processed_paths[0])
