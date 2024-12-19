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
from utils_apt.util_dataset import get_graph_dataset
from utils_apt.dataset_constants import is_supported_dataset, raise_unsupported_dataset
from utils_apt.distillation_constants import SupportedDistillationMethods, raise_unsupported_distillation_method

import deeprobust.graph.utils as utils
import torch.nn.functional as F
from networks_nc.gcn import GCN
from coreset import KCenter, Herding, Random

def _print_separotor_line():
    print("--------------------")

def main():

    parser = argparse.ArgumentParser(description="Parameters for graph distillation")
    parser.add_argument(
        "--method", type=str, help="Method: Compulsory argument",
        choices=["random", "herding", "kcenter", "GCDM"]
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset: Compulsory argument",
        choices=["theia"]
    )
    parser.add_argument("--reduction_rate", type=float, default=1.0)
    
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--section", type=str, default="")
    parser.add_argument("--wandb", type=int, default=0, help="Use wandb")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--sgc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0, help="Regularization term")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    # parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs") # GCDM
    # parser.add_argument("--epochs", type=int, default=400, help="Number of epochs") # Coreset
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=0.01) # Coreset
    parser.add_argument("--lr_adj", type=float, default=1e-3)
    parser.add_argument("--lr_feat", type=float, default=1e-3)
    parser.add_argument("--lr_model", type=float, default=1e-2)
    # parser.add_argument("--weight_decay", type=float, default=0.0) # GCDM
    parser.add_argument("--weight_decay", type=float, default=5e-4) # Coreset
    # parser.add_argument("--dropout", type=float, default=0.0) # GCDM
    parser.add_argument("--dropout", type=float, default=0.5) # Coreset
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--outer", type=int, default=20)
    parser.add_argument("--transductive", type=int, default=1) # GCDM
    parser.add_argument("--inductive", type=int, default=1) # Coreset
    parser.add_argument("--mlp", type=int, default=0) # Coreset
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_separotor_line()
    print(args)
    print(f"Torch device: id {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    _print_separotor_line()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(f"{args.save_dir}/{args.method}"):
        os.makedirs(f"{args.save_dir}/{args.method}")
    
    dataset_name = args.dataset
    if not is_supported_dataset(dataset_name):
        raise_unsupported_dataset(dataset_name)
    apt_graph = get_graph_dataset(dataset_name)
    apt_graph = Pyg2Dpr(apt_graph, dataset_name=f"{dataset_name}_dt")
    data = Transd2Ind(apt_graph, keep_ratio=args.keep_ratio)

    _print_separotor_line()
    
    distillation_method = args.method
    
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
        
        
        torch.save(
            adj_train,
            f"{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
        torch.save(
            feat_train,
            f"{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
        torch.save(
            labels_train,
            f"{args.save_dir}/{args.method}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
    elif distillation_method == SupportedDistillationMethods.GCDM.value:
        agent = GCDM(data, args, device="cuda")
        agent.train()
    else:
        raise_unsupported_distillation_method(distillation_method)
    
    _print_separotor_line()




def main1():

    parser = argparse.ArgumentParser(description="Parameters for graph distillation")
    parser.add_argument(
        "--method", type=str, help="Method: Compulsory argument",
        choices=["random", "GCDM"]
    )
    # parser.add_argument("--method", type=str, choices=["kcenter", "herding", "random"])
    parser.add_argument(
        "--dataset", type=str, help="Dataset: Compulsory argument",
        choices=["theia"]
    )
    parser.add_argument("--reduction_rate", type=float, default=1.0)
    
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--section", type=str, default="")
    parser.add_argument("--wandb", type=int, default=0, help="Use wandb")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--sgc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0, help="Regularization term")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    # parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs") # GCDM
    # parser.add_argument("--epochs", type=int, default=400, help="Number of epochs") # Coreset
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=0.01) # Coreset
    parser.add_argument("--lr_adj", type=float, default=1e-3)
    parser.add_argument("--lr_feat", type=float, default=1e-3)
    parser.add_argument("--lr_model", type=float, default=1e-2)
    # parser.add_argument("--weight_decay", type=float, default=0.0) # GCDM
    parser.add_argument("--weight_decay", type=float, default=5e-4) # Coreset
    # parser.add_argument("--dropout", type=float, default=0.0) # GCDM
    parser.add_argument("--dropout", type=float, default=0.5) # Coreset
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--outer", type=int, default=20)
    parser.add_argument("--transductive", type=int, default=1) # GCDM
    parser.add_argument("--inductive", type=int, default=1) # Coreset
    parser.add_argument("--mlp", type=int, default=0) # Coreset
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

    _print_separotor_line()
    
    distillation_method = args.method
    
    if distillation_method in [
        SupportedDistillationMethods.RANDOM.value,
        SupportedDistillationMethods.KCENTER.value,
        SupportedDistillationMethods.HERDING.value
    ]:
        match distillation_method:
            case SupportedDistillationMethods.RANDOM.value:
                print("Random")
            case SupportedDistillationMethods.KCENTER.value:
                print("KCenter")
            case SupportedDistillationMethods.HERDING.value:
                print("Herding")
    elif distillation_method == SupportedDistillationMethods.GCDM.value:
        print("GCDM")
    else:
        raise_unsupported_distillation_method(distillation_method)
    
    _print_separotor_line()

if __name__ == "__main__":
    main()
