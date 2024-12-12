from itertools import compress
from torch_geometric import utils
import json
from torch_geometric.loader import NeighborLoader
import torch
import random
import numpy as np


def helper(MP, all_pids=None, GP=None, edges=None, mapp=None, dataset_name="theia"):
    with open(
        f"data_files/{dataset_name}.json", "r"
    ) as json_file:  # change the dataset_name in the def parameters
        GT_mal = set(json.load(json_file))

    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, 2)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, 2)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR, acc = calculate_metrics(TP, FP, FN, TN)
    # print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
    print(
        f"True Positives/False Positives/False Negatives/True Negatives: {TP}/{FP}/{FN}/{TN}"
    )
    print(
        f"Accuracy: {round(acc, 2)}, Precision: {round(prec, 2)}, Recall: {round(rec, 2)}, Fscore: {round(fscore, 2)}"
    )

    return TPL, FPL


def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    acc = (TN + TP) / (TN + TP + FN + FP) if TN + TP + FN + FP > 0 else 0

    return prec, rec, fscore, FPR, TPR, acc


def Get_Adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()

    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))

    return neighbors


def downstream_training(graph, model, dataset="theia", device="cpu"):
    flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=device)
    for m_n in range(1):
        # model.load_state_dict(
        #     torch.load(
        #         f"trained_weights/{dataset}/lword2vec_gnn_{dataset}{m_n}_E3.pth",
        #         map_location=torch.device("cpu"),
        #     )
        # )
        loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=5000)
        for subg in loader:
            subg = subg.to(device)
            model.eval()
            out = model(subg.x, subg.edge_index)

            sorted, indices = out.sort(dim=1, descending=True)
            conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
            conf = (conf - conf.min()) / conf.max()

            pred = indices[:, 0]
            cond = (pred == subg.y) & (conf > 0.53)
            subg_n_id_cond = subg.n_id[cond].to(device)
            flag[subg_n_id_cond] = torch.logical_and(
                flag[subg_n_id_cond],
                torch.tensor(
                    [False] * len(subg_n_id_cond), dtype=torch.bool, device=device
                ),
            )

    index = utils.mask_to_index(flag).tolist()
    ids = set(index)
    alerts = helper(set(ids), set(all_pids), GP, edges, mapp)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
