import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import random
import numpy as np
import torch
from utils.utils_graph import *
from utils.utils import *
from DM.gcdm import GCDM
import deeprobust.graph.utils as utils
import torch.nn.functional as F
from networks_nc.gcn import GCN
from coreset import KCenter, Herding, Random
from GM.agent_induct import GCond
from utils_apt.dataset_constants import is_supported_dataset, raise_unsupported_dataset
from utils_apt.distillation_constants import SupportedDistillationMethods, is_supported_method, raise_unsupported_distillation_method
from utils_apt.gcbench_args_helper import set_args_coreset, set_args_DM, set_args_GM_NC
from utils_apt.util_file_path import get_save_graph_data_file_for_distillation_input

def _print_separotor_line():
    print("--------------------")

def _load_apt_graph_from_file(dataset_name: str):
    apt_graph = torch.load(get_save_graph_data_file_for_distillation_input(dataset_name))
    return apt_graph

def main():

    parser = argparse.ArgumentParser(description="Parameters for graph distillation")
    #### Required args ####
    parser.add_argument(
        "--method", type=str, help="Distillation method: Compulsory argument", required=True,
        choices=["random", "herding", "kcenter", "GCDM", "GCond", "SGDD"]
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset: Compulsory argument", required=True,
        choices=["theia"]
    )
    parser.add_argument("--reduction_rate", type=float, help="Reduction rate as a floating point number", required=True)
    #### Common args for all methods ####
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--wandb", type=int, default=0, help="Use wandb")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--debug", type=int, default=0)
    #### Other args - should be adapted for each method ####
    parser.add_argument("--config", type=str, help="Path to the config JSON file")
    parser.add_argument("--config_dir", type=str)
    parser.add_argument("--section", type=str)
    parser.add_argument("--wandb_id", type=str)
    parser.add_argument("--sgc", type=int)
    parser.add_argument("--alpha", type=float, help="Regularization term")
    parser.add_argument("--nlayers", type=int, help="Number of layers")
    parser.add_argument("--hidden", type=int, help="Hidden layer size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_adj", type=float)
    parser.add_argument("--lr_feat", type=float)
    parser.add_argument("--lr_model", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--normalize_features", type=bool)
    parser.add_argument("--gt", type=int)
    parser.add_argument("--inner", type=int)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--transductive", type=int, default=0)
    parser.add_argument("--inductive", type=int, default=1)
    parser.add_argument("--mlp", type=int)
    parser.add_argument("--one_step", type=int)
    parser.add_argument("--init_way", type=str)
    parser.add_argument("--label_rate", type=float, default=1)
    parser.add_argument("--dis_metric", type=str, help="Distance metric")
    parser.add_argument("--beta", type=float, help="coefficient for eculidean distance")
    parser.add_argument("--early_stopping", type=int)
    parser.add_argument("--max_epochs_without_improvement", type=int)
    parser.add_argument("--prune", type=float)
    parser.add_argument("--mining", type=float)
    parser.add_argument("--circulation", type=int)
    parser.add_argument("--mx_size", type=int)
    parser.add_argument("--ep_ratio",type=float,help="control the ratio of direct edges predict term in the graph.")
    parser.add_argument("--sinkhorn_iter",type=int,help="use sinkhorn iteration to warm-up the transport plan.")
    parser.add_argument("--opt_scale", type=float, help="control the scale of the opt loss")
    parser.add_argument("--coreset_method",type=str,choices=["kcenter", "herding", "random"])
    parser.add_argument("--option",type=int)

    args = parser.parse_args()
    
    dataset_name = args.dataset
    if not is_supported_dataset(dataset_name):
        raise_unsupported_dataset(dataset_name)
    distillation_method = args.method
    if not is_supported_method(distillation_method):
        raise_unsupported_distillation_method(distillation_method)
    
    if distillation_method in [
        SupportedDistillationMethods.RANDOM.value,
        SupportedDistillationMethods.KCENTER.value,
        SupportedDistillationMethods.HERDING.value
    ]:
        set_args_coreset(args)
    elif distillation_method in [
        SupportedDistillationMethods.GCDM.value
    ]:
        set_args_DM(args)
        if os.path.exists(args.config_dir + "/" + args.config):
            with open(args.config_dir + "/" + args.config, "r") as config_file:
                config = json.load(config_file)

            if args.section in config:
                section_config = config[args.section]

            for key, value in section_config.items():
                setattr(args, key, value)
    elif distillation_method in [
        SupportedDistillationMethods.GCOND.value,
        SupportedDistillationMethods.SGDD.value
    ]:
        set_args_GM_NC(args)
        if os.path.exists(args.config_dir + "/" + args.config):
            with open(args.config_dir + "/" + args.config, "r") as config_file:
                config = json.load(config_file)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(f"{args.save_dir}/{args.method}"):
        os.makedirs(f"{args.save_dir}/{args.method}")

    torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_separotor_line()
    print(args)
    print(f"Torch device: id {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    _print_separotor_line()
    
    apt_graph = _load_apt_graph_from_file(dataset_name)
    
    apt_graph = Pyg2Dpr(apt_graph, dataset_name=f"{dataset_name}_dt")
    data = Transd2Ind(apt_graph, keep_ratio=args.keep_ratio)

    _print_separotor_line()
    
    if distillation_method in [
        SupportedDistillationMethods.RANDOM.value,
        SupportedDistillationMethods.KCENTER.value,
        SupportedDistillationMethods.HERDING.value
    ]:
        # Extracted from GC-Bench/coreset/train_coreset_induct.py
        feat_train, adj_train, labels_train = data.feat_train, data.adj_train, data.labels_train
        
        model = GCN(nfeat=feat_train.shape[1], nhid=256, nclass=labels_train.max() + 1, device=device, weight_decay=args.weight_decay,)
        model = model.to(device)
        model.fit_with_val(feat_train, adj_train, labels_train, data, train_iters=600, normalize=True, verbose=False,)
        
        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()
        feat_test, adj_test = data.feat_test, data.adj_test

        embeds = model.predict().detach()

        output = model.predict(feat_test, adj_test)
        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        print(
            "FUll: Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )
        
        match distillation_method:
            case SupportedDistillationMethods.RANDOM.value:
                agent = Random(data, args, device="cuda")
            case SupportedDistillationMethods.KCENTER.value:
                agent = KCenter(data, args, device="cuda")
            case SupportedDistillationMethods.HERDING.value:
                agent = Herding(data, args, device="cuda")
        
        idx_selected = agent.select(embeds, inductive=True)
        
        feat_train = feat_train[idx_selected]
        adj_train = adj_train[np.ix_(idx_selected, idx_selected)]
        labels_train = labels_train[idx_selected]
        
        torch.save(adj_train,f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",)
        torch.save(feat_train,f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",)
        torch.save(labels_train,f"{args.save_dir}/{args.method}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",)
    elif distillation_method == SupportedDistillationMethods.GCDM.value:
        agent = GCDM(data, args, device="cuda")
        agent.train()
    elif distillation_method in [
        SupportedDistillationMethods.GCOND.value,
        SupportedDistillationMethods.SGDD.value
    ]:
        # data: <class 'utils.utils.Transd2Ind'>, data_full: <class 'utils.utils.Pyg2Dpr'>
        data_full = apt_graph
        
        # Extracted from GC-Bench/GM/main_nc.py
        if data_full.adj.shape[0] < args.mx_size:
            args.mx_size = data_full.adj.shape[0]
        elif args.transductive:
            data.adj_mx = data_full.adj[: args.mx_size, : args.mx_size]
        else:
            while True:  # exclude the subgraph with all zero edges
                subgraph_nodes = np.random.choice(
                    data_full.adj.shape[0], args.mx_size, replace=False
                )
                subgraph = data_full.adj[np.ix_(subgraph_nodes, subgraph_nodes)]
                if subgraph.sum() > 0:
                    break
            data.adj_mx = subgraph
        
        agent = GCond(data, args, device="cuda")
        
        agent.train()
    
    _print_separotor_line()

if __name__ == "__main__":
    main()
