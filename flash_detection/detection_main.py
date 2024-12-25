import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import json
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.loader import NeighborLoader
from sklearn.utils import class_weight
from torch.nn import CrossEntropyLoss
from torch_geometric import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils_apt.util_file_path import get_distillion_saved_file_path, gnn_model_save_file_path, get_ground_truth_file_path, gnn_model_save_dir_path
from utils_apt.flash_params_helper import get_gnn_training_epochs, get_gnn_training_batch_size, get_gnn_triaing_conf_score, get_gnn_testing_conf_score
from utils_apt.gnn_models import GCN
from utils_apt.flash_evaluation_helper import eval_helper
from utils_apt.util_file_path import get_save_evaluation_graph_data_file_for_detection_input
from utils_apt.graph_prep_util import graph_number_of_classes

def _print_separotor_line():
    print("--------------------")

def main_train_mode(dataset_name: str, model:torch.nn.Module, optimizer:torch.optim.Optimizer, distillation_method: str, distillation_ratio: int, distillation_seed: int):
    
    print(f"****Executing function main_train_mode({dataset_name}, {type(model)}, {type(optimizer)}, {distillation_method}, {distillation_ratio}):")
    
    adj_path, feature_path, label_path = get_distillion_saved_file_path(dataset_name, distillation_method, distillation_ratio, distillation_seed)
    
    print(f"Distilled graph will be loaded from: adjacency ({adj_path}), features ({feature_path}), labels ({label_path})")
    
    adj = torch.load(adj_path)
    feats = torch.load(feature_path)
    labels = torch.load(label_path)
    
    print(f"Distilled graph is loaded. Adj:{adj.shape}, features(nodes):{feats.shape}, labels:{labels.shape}")

    if torch.is_tensor(labels):
        nodes = feats
        labels = labels.cpu()
        # print(type(adj)) # <class 'torch.Tensor'>
        edges, _ = dense_to_sparse(adj)
    else:
        nodes = feats
        labels = labels
        # print(type(adj)) # <class 'scipy.sparse._csr.csr_matrix'>
        adj = torch.from_numpy(adj.toarray())
        edges, _ = dense_to_sparse(adj)
    
    graph = Data(
        x=torch.tensor(nodes, dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
    )
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=device)
    
    l = np.array(labels)
    class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(l), y=l)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    print("Ready for training.")
    
    EPOCHS = get_gnn_training_epochs(dataset_name)
    BATCH_SIZE = get_gnn_training_batch_size()
    CONF_SCORE = get_gnn_triaing_conf_score()
    
    os.makedirs(gnn_model_save_dir_path(dataset_name), exist_ok=True)
    for m_n in range(EPOCHS):
        loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=BATCH_SIZE, input_nodes=mask)
        total_loss = 0
        for subg in loader:
            subg=subg.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(subg.x, subg.edge_index)
            loss = criterion(out, subg.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * subg.batch_size
        if (mask.sum().item() != 0):
            print(total_loss / mask.sum().item())
        
        loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=BATCH_SIZE,input_nodes=mask)
        for subg in loader:
            subg=subg.to(device)
            model.eval()
            out = model(subg.x, subg.edge_index)
            
            sorted, indices = out.sort(dim=1,descending=True)
            conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
            conf = (conf - conf.min()) / conf.max()
            
            pred = indices[:,0]
            cond = (pred == subg.y) | (conf >= CONF_SCORE)
            subg_n_id_cond = subg.n_id[cond].to(device)
            mask[subg_n_id_cond] = False

        torch.save(model.state_dict(), gnn_model_save_file_path(dataset_name, m_n, distillation_method, distillation_ratio))
        print(f'Model# {m_n}. {mask.sum().item()} nodes still misclassified \n')
    
    print("****Done - Execution of function main_train_mode")


def _load_apt_graph_from_file(dataset_name: str):
    save_nodes_file, save_labels_file, save_edges_file, save_mapp_file, save_allids_file = get_save_evaluation_graph_data_file_for_detection_input(dataset_name)
    
    nodes = np.load(save_nodes_file)
    with open(save_labels_file, 'r') as f:
        labels = json.load(f)
    with open(save_edges_file, 'r') as f:
        edges = json.load(f)
    with open(save_mapp_file, 'r') as f:
        mapp = json.load(f)
    with open(save_allids_file, 'r') as f:
        all_ids = set(json.load(f))
    
    return nodes, labels, edges, mapp, all_ids

def main_test_mode(dataset_name: str, model:torch.nn.Module, distillation_method: str, distillation_rate:float):
    
    print(f"****Executing function main_test_mode({dataset_name}, {type(model)}):")
    
    nodes, labels, edges, mapp, all_ids = _load_apt_graph_from_file(dataset_name)
    
    gt_path = get_ground_truth_file_path(dataset_name)
    with open(gt_path, "r") as gt_json_file:
        ground_truth_malicious = set(json.load(gt_json_file))
    
    graph = Data(
        x=torch.tensor(nodes,dtype=torch.float).to(device),
        y=torch.tensor(labels,dtype=torch.long).to(device),
        edge_index=torch.tensor(edges,dtype=torch.long).to(device)
    )
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=device)
    
    print("Ready for testing.")
    
    EPOCHS = get_gnn_training_epochs(dataset_name)
    BATCH_SIZE = get_gnn_training_batch_size()
    CONF_SCORE = get_gnn_testing_conf_score(dataset_name)
    
    for m_n in range(EPOCHS):
        model.load_state_dict(
            torch.load(gnn_model_save_file_path(dataset_name, m_n, distillation_method, distillation_rate), map_location=torch.device('cpu'))
        )
        
        loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=BATCH_SIZE)
        for subg in loader:
            subg=subg.to(device)
            model.eval()
            out = model(subg.x, subg.edge_index)
            
            sorted, indices = out.sort(dim=1,descending=True)
            conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
            conf = (conf - conf.min()) / conf.max()
            
            pred = indices[:,0]
            cond = (pred == subg.y) & (conf > CONF_SCORE)
            subg_n_id_cond = subg.n_id[cond].to(device)
            flag[subg_n_id_cond] = torch.logical_and(
                flag[subg_n_id_cond], torch.tensor([False] * len(flag[subg_n_id_cond]), dtype=torch.bool, device=device)
            )

    index = utils.mask_to_index(flag).tolist()
    ids = set([mapp[x] for x in index])
    eval_helper(set(ids), set(all_ids), ground_truth_malicious, edges, mapp)
    
    print(f"****Done - Execution of function main_test_mode")

def main():

    parser = argparse.ArgumentParser(description="Parameters for APT detection")
    parser.add_argument(
        "--dataset", type=str, help="Dataset", required=True,
        choices=["theia", "cadets", "trace", "fivedirections"]
    )
    parser.add_argument("--mode", type=str, help="Detection model mode", default="train", choices=["train", "test"], required=True)
    
    parser.add_argument(
        "--dist_method", type=str, help="Distillation Method", required=True,
        choices=["random", "herding", "kcenter", "GCDM", "GCond", "SGDD"]
    )
    parser.add_argument("--dist_ratio", type=float, default=0.01, help="Reduction ratio at time of distillation")
    parser.add_argument("--dist_seed", type=int, default=15, help="Seed at the time of distillation")
    # parser.add_argument("--dist_mode", type=str, default="transductive", help="Distilltion mode (only for GCond and SGDD)", choices=["inductive", "transductive"])
    parser.add_argument("--dist_mode", type=str, default="inductive", help="Distilltion mode (only for GCond and SGDD)", choices=["inductive", "transductive"])

    args = parser.parse_args()

    _print_separotor_line()
    print(args)
    _print_separotor_line()
    
    gnn_out_channel = graph_number_of_classes(args.dataset)
    dtc_model = GCN(30,gnn_out_channel).to(device)
    dtc_optimizer = torch.optim.Adam(dtc_model.parameters(), lr=0.01, weight_decay=5e-4)

    distillation_method = f"{args.dist_method}_{args.dist_mode}" if (args.dist_method in ["GCond", "SGDD"]) else args.dist_method
    if args.mode == "train":
        # distillation_method = f"{args.dist_method}_{args.dist_mode}" if (args.dist_method in ["GCond", "SGDD"]) else args.dist_method
        main_train_mode(args.dataset, dtc_model, dtc_optimizer, distillation_method, args.dist_ratio, args.dist_seed)
    elif args.mode == "test":
        main_test_mode(args.dataset, dtc_model, distillation_method, args.dist_ratio)
    else:
        print(f"--mode \"{args.mode}\" is not implemented. Supported options are [train, test]")

    _print_separotor_line()


if __name__ == "__main__":
    main()
