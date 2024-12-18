import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
import os
# import torch.nn.functional as F
import json
import warnings
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')
from torch_geometric.loader import NeighborLoader
import multiprocessing
# from pprint import pprint
# import gzip
# from sklearn.manifold import TSNE
import json
# import copy
from sklearn.utils import class_weight
# import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset

from utils_apt.util_file_path import get_distillion_saved_file_path, gnn_model_save_file_path, get_names_of_test_data_files, get_ground_truth_file_path, w2v_model_save_file
from utils_apt.distillation_constants import is_modern_distillation
from utils_apt.flash_params_helper import get_gnn_training_epochs, get_gnn_training_batch_size, get_gnn_triaing_conf_score, get_gnn_testing_conf_score
from utils_apt.gnn_models import GCN
from utils_apt.dataset_prep_util import prep_dataframe
from utils_apt.graph_prep_util import prepare_graph
from utils_apt.w2v_util import PositionalEncoder, FLASH_W2V_DIMENSION, load_w2v_model, w2v_infer
from utils_apt.flash_evaluation_helper import eval_helper

def _print_separotor_line():
    print("--------------------")

def main_train_mode(dataset_name: str, model:torch.nn.Module, optimizer:torch.optim.Optimizer, distillation_method: str, distillation_ratio: int):
    
    print(f"****Executing function main_train_mode({dataset_name}, {type(model)}, {type(optimizer)}, {distillation_method}, {distillation_ratio}):")
    
    if is_modern_distillation(distillation_method):
        adj_path, feature_path, label_path = get_distillion_saved_file_path(dataset_name, distillation_method, distillation_ratio)
        
        print(f"Distilled graph will be loaded from: adjacency ({adj_path}), features ({feature_path}), labels ({label_path})")
        
        adj = torch.load(adj_path)
        feats = torch.load(feature_path)
        labels = torch.load(label_path)
        
        print(f"Distilled graph is loaded. Adj:{adj.shape}, features(nodes):{feats.shape}, labels:{labels.shape}")

        nodes = feats
        labels = labels.cpu()
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
        CONF_SCORE = get_gnn_triaing_conf_score(dataset_name)
        
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

            torch.save(model.state_dict(), gnn_model_save_file_path(dataset_name, m_n))
            print(f'Model# {m_n}. {mask.sum().item()} nodes still misclassified \n')
    
    print("****Done - Execution of function main_train_mode")

def main_test_mode(dataset_name: str, model:torch.nn.Module):
    
    print(f"****Executing function main_test_mode({dataset_name}, {type(model)}):")
    
    txt_processed_source, json_attribute_source = get_names_of_test_data_files(dataset_name)
    print(f"source data files: {txt_processed_source}, {json_attribute_source}")
    
    data_frame = prep_dataframe(dataset_name, txt_processed_source, json_attribute_source)
    node_features, labels, edges, mapp = prepare_graph(dataset_name, data_frame)
    
    encoder = PositionalEncoder(FLASH_W2V_DIMENSION)
    w2v_model_file = w2v_model_save_file(dataset_name)
    w2v_model = load_w2v_model(w2v_model_file)
    print(f"w2v model loaded from {w2v_model_file}")
    
    nodes = [w2v_infer(x, w2v_model, encoder) for x in node_features]
    nodes = np.array(nodes)
    print(f"Number of nodes: {len(nodes)}")
    
    all_ids = list(data_frame['actorID']) + list(data_frame['objectID'])
    all_ids = set(all_ids)
    print(f"Length of all_ids: {len(all_ids)}")
    
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
            torch.load(gnn_model_save_file_path(dataset_name, m_n), map_location=torch.device('cpu'))
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
    parser.add_argument("--dataset", type=str, default="theia", help="Dataset")
    parser.add_argument("--mode", type=str, default="train", help="Options: [train, test]")
    
    parser.add_argument("--dist_method", type=str, default="GCDM", help="Distillation Method")
    parser.add_argument("--dist_ratio", type=float, default=0.01, help="Reduction ratio at time of distillation")

    args = parser.parse_args()

    _print_separotor_line()
    print(args)
    _print_separotor_line()
    
    # dtc_model = GCN(30,5).to(device)###########################################################
    dtc_model = GCN(30,4).to(device)
    dtc_optimizer = torch.optim.Adam(dtc_model.parameters(), lr=0.01, weight_decay=5e-4)

    if args.mode == "train":
        main_train_mode(args.dataset, dtc_model, dtc_optimizer, args.dist_method, args.dist_ratio)
    elif args.mode == "test":
        main_test_mode(args.dataset, dtc_model)
    else:
        print(f"--mode \"{args.mode}\" is not implemented. Supported options are [train, test]")

    _print_separotor_line()


if __name__ == "__main__":
    main()
