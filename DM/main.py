import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from utils.utils_graph import *
from utils.utils import *
from gcdm import GCDM
from apt_dataset import APT_Dummy
from graph_prep_util import prepare_graph
from dataset_prep_util import prep_dataframe
from tracing_logging_util import w2v_model_save_file
from w2v_util import PositionalEncoder, load_w2v_model, infer
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_scipy_sparse_matrix


def main():

    parser = argparse.ArgumentParser(
        description="Parameters for GCBM-node classification"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config JSON file"
    )
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--section", type=str, default="")
    parser.add_argument("--wandb", type=int, default=0, help="Use wandb")
    parser.add_argument("--method", type=str, default="GCDM", help="Method")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--sgc", type=int, default=1)
    parser.add_argument("--reduction_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0, help="Regularization term")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr_adj", type=float, default=1e-3)
    parser.add_argument("--lr_feat", type=float, default=1e-3)
    parser.add_argument("--lr_model", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--outer", type=int, default=20)
    parser.add_argument("--transductive", type=int, default=1)
    parser.add_argument("--one_step", type=int, default=0)
    parser.add_argument("--init_way", type=str, default="Random_real")
    parser.add_argument("--label_rate", type=float, default=1)

    args = parser.parse_args()
    if os.path.exists(args.config_dir + "/" + args.config):
        with open(args.config_dir + "/" + args.config, "r") as config_file:
            config = json.load(config_file)

        if args.section in config:
            section_config = config[args.section]

        for key, value in section_config.items():
            setattr(args, key, value)

    torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(args)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(f"{args.save_dir}/{args.method}"):
        os.makedirs(f"{args.save_dir}/{args.method}")

    data_pyg = [
        "cora",
        "citeseer",
        "pubmed",
        "cornell",
        "texas",
        "wisconsin",
        "chameleon",
        "squirrel",
    ]
    if args.dataset in data_pyg:
        data_full = get_dataset(args.dataset, args.normalize_features, args.data_dir)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    elif args.dataset == "apt_dummy":
        num_nodes = 500
        num_features = 30
        num_classes = 2
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

        apt_graph = Data(
            x=nodes,
            y=labels,
            edge_index=edge_index,
            adj=adj,  # Include adjacency matrix
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        apt_graph = Pyg2Dpr(apt_graph, dataset_name="apt_dummy")
        data = Transd2Ind(apt_graph, keep_ratio=args.keep_ratio)
    elif args.dataset == "theia":
        dataset_name = "theia"
        test_file_processed_txt = "theia_test.txt"
        file_for_adding_attributes_from_json = "ta1-theia-e3-official-6r.json.8"
        
        data_frame = prep_dataframe(dataset_name, test_file_processed_txt, file_for_adding_attributes_from_json)
        phrases,labels,edges,mapp = prepare_graph(dataset_name, data_frame)
        
        encoder = PositionalEncoder(30)
        w2v_model_file = w2v_model_save_file(dataset_name)
        w2v_model = load_w2v_model(w2v_model_file)

        nodes = [infer(x, w2v_model, encoder) for x in phrases]
        nodes = np.array(nodes)
        
        num_nodes = len(nodes)
        
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
        
        # # Split indices into train, val, and test sets
        # all_indices = torch.arange(num_nodes)
        # idx_train, idx_test = train_test_split(
        #     all_indices, test_size=0.2, random_state=42
        # )
        # idx_train, idx_val = train_test_split(
        #     idx_train, test_size=0.25, random_state=42
        # )
        
        # Stratified split for train, test, and validation
        all_indices = torch.arange(num_nodes)
        print(f"all_indices:{all_indices.shape}")
        idx_train, idx_test = train_test_split(
            all_indices, test_size=0.2, random_state=42, stratify=labels
        )
        idx_train, idx_val = train_test_split(
            idx_train, test_size=0.25, random_state=42, stratify=labels[idx_train]
        )

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
        
        
        theia_graph = Pyg2Dpr(theia_graph, dataset_name="theia_dt")
        data = Transd2Ind(theia_graph, keep_ratio=args.keep_ratio)
    else:
        if args.transductive:
            data = DataGraph(args.dataset, data_dir=args.data_dir)
        else:
            data = DataGraph(
                args.dataset, label_rate=args.label_rate, data_dir=args.data_dir
            )
        data_full = data.data_full

    agent = GCDM(data, args, device="cuda")

    agent.train()


if __name__ == "__main__":
    main()
