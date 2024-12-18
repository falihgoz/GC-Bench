import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from utils_apt.util_file_path import get_names_of_related_data_files, w2v_model_save_file
from utils_apt.dataset_prep_util import prep_dataframe
from utils_apt.graph_prep_util import prepare_graph
from utils_apt.w2v_util import PositionalEncoder, FLASH_W2V_DIMENSION, load_w2v_model, w2v_infer

def get_graph_dataset(dataset_name: str) -> Data:
    print(f"****Executing function get_graph_dataset({dataset_name}):")
    
    txt_processed_source, json_attribute_source = get_names_of_related_data_files(dataset_name)
    print(f"source data files: {txt_processed_source}, {json_attribute_source}")
    
    data_frame = prep_dataframe(dataset_name, txt_processed_source, json_attribute_source)
    node_features, labels, edges, mapp = prepare_graph(dataset_name, data_frame)
    
    encoder = PositionalEncoder(FLASH_W2V_DIMENSION)
    w2v_model_file = w2v_model_save_file(dataset_name)
    w2v_model = load_w2v_model(w2v_model_file)
    print(f"w2v model loaded from {w2v_model_file}")

    nodes = [w2v_infer(x, w2v_model, encoder) for x in node_features]
    nodes = np.array(nodes)
    
    num_nodes = len(nodes)
    print(f"After w2v: {num_nodes} nodes")
    
    nodes = torch.tensor(nodes, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long)

    # Compute adjacency matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocoo()
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([adj.row, adj.col]),
        values=torch.tensor(adj.data),
        size=(num_nodes, num_nodes),
        dtype=torch.float,
    )
    print(f"Adjacency matrix {adj.shape}")
    
    # Stratified split for train, test, and validation
    all_indices = torch.arange(num_nodes)
    print(f"all_indices:{all_indices.shape}")
    idx_train, idx_test = train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=labels
    )
    idx_train, idx_val = train_test_split(
        idx_train, test_size=0.25, random_state=42, stratify=labels[idx_train]
    )
    print(f"{num_nodes} nodes splitted to: train {len(idx_train)}, test {len(idx_test)}, validation {len(idx_val)}")

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # Create the data object
    theia_graph = Data(
        x=nodes,
        y=labels,
        edge_index=edge_index,
        adj=adj,  # Include adjacency matrix
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    
    print(f"****Done - Execution of function get_graph_dataset")
    return theia_graph
