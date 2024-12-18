from torchvision import datasets, transforms
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import torch_geometric.transforms as T

# from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import get_train_val_test
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from deeprobust.graph.utils import *
from torch_geometric.data import NeighborSampler, HeteroData
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.datasets import (
    Planetoid,
    WebKB,
    WikipediaNetwork,
    IMDB,
    HGBDataset,
)
import math
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import os
import logging


def get_dataset(
    name, normalize_features=False, dataset_dir=None, transform=None, if_dpr=True
):
    if dataset_dir is None:
        path = osp.join(osp.dirname(osp.realpath(__file__)), "data", name)
    else:
        path = osp.join(dataset_dir)
    if name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(path, name)
    elif name in ["IMDB"]:
        dataset = IMDB(root=f"{path}/{name}")
    elif name in ["imdb"]:
        dataset = HGBDataset(root=f"{path}/{name}", name=name)
    # elif name in ["ogbn-arxiv"]:
    #     dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    elif name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(path, name)
    elif name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(path, name)
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    dpr_data = Pyg2Dpr(dataset, dataset_name=name)
    if name in ["ogbn-arxiv"]:
        # the features are different from the features provided by GraphSAINT
        # normalize features, following graphsaint
        feat, idx_train = dpr_data.features, dpr_data.idx_train
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)
        dpr_data.features = feat

    return dpr_data


class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, dataset_name=None, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass
        if dataset_name is None:
            dataset_name = pyg_data.name
        # pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if isinstance(pyg_data, HeteroData):
            features = []
            edge_index_list = []
            labels = []
            node_offset = 0
            for node_type in pyg_data.node_types:
                if hasattr(pyg_data[node_type], "x"):
                    features.append(pyg_data[node_type].x.numpy())
                else:
                    features.append(np.eye(pyg_data[node_type].num_nodes))
                if "y" in pyg_data[node_type]:
                    n_target = pyg_data[node_type].num_nodes
                    dim_target = pyg_data[node_type].x.shape[1]
                    labels.append(pyg_data[node_type].y.numpy())
                    self.idx_train = mask_to_index(
                        pyg_data[node_type].train_mask, n_target
                    )
                    if hasattr(pyg_data[node_type], "val_mask"):
                        self.idx_val = mask_to_index(
                            pyg_data[node_type].val_mask, n_target
                        )
                    else:
                        self.idx_val = np.array([], dtype=np.int64)
                    self.idx_test = mask_to_index(
                        pyg_data[node_type].test_mask, n_target
                    )
                    self.name = "Pyg2Dpr"
                # else:
                #     labels.append(-1 * np.ones(pyg_data[node_type].x.shape[0], dtype=int))
                num_nodes = pyg_data[node_type].num_nodes
                node_offset += num_nodes
            # max_size = max(feature.shape[1] for feature in features)
            padded_features = [
                np.pad(
                    feature[:, :dim_target],
                    ((0, 0), (0, max(0, dim_target - feature.shape[1]))),
                    mode="constant",
                )
                for feature in features
            ]
            self.features = np.concatenate(padded_features, axis=0)
            adj_data = []
            for edge_type in pyg_data.edge_types:
                edge_index = pyg_data[edge_type].edge_index
                adj_data.append(
                    (
                        np.ones(edge_index.size(1)),
                        (edge_index[0].numpy(), edge_index[1].numpy()),
                    )
                )
            row = np.concatenate([data[1][0] for data in adj_data])
            col = np.concatenate([data[1][1] for data in adj_data])
            data = np.concatenate([data[0] for data in adj_data])
            self.labels = np.concatenate(labels)
            self.adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
        else:
            if dataset_name == "ogbn-arxiv":  # symmetrization
                pyg_data.edge_index = to_undirected(
                    pyg_data.edge_index, pyg_data.num_nodes
                )

            self.adj = sp.csr_matrix(
                (
                    np.ones(pyg_data.edge_index.shape[1]),
                    (pyg_data.edge_index[0], pyg_data.edge_index[1]),
                ),
                shape=(n, n),
            )

            self.features = pyg_data.x.numpy()
            self.labels = pyg_data.y.numpy()

            if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
                self.labels = self.labels.reshape(-1)  # ogb-arxiv needs to reshape

            if hasattr(pyg_data, "train_mask"):
                # for fixed split
                self.idx_train = mask_to_index(pyg_data.train_mask, n)
                self.idx_val = mask_to_index(pyg_data.val_mask, n)
                self.idx_test = mask_to_index(pyg_data.test_mask, n)
                self.name = "Pyg2Dpr"
            else:
                try:
                    # for ogb
                    self.idx_train = splits["train"]
                    self.idx_val = splits["valid"]
                    self.idx_test = splits["test"]
                    self.name = "Pyg2Dpr"
                except:
                    # for other datasets
                    self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                        nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels
                    )


def mask_to_index(index, size, split_choice=0):
    if len(index.shape) == 2:
        index = index[:, split_choice]
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


class Transd2Ind:
    # transductive setting to inductive setting

    def __init__(self, dpr_data, keep_ratio=1.0):
        idx_train, idx_val, idx_test = (
            dpr_data.idx_train,
            dpr_data.idx_val,
            dpr_data.idx_test,
        )
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.nclass = labels.max() + 1
        self.adj_full, self.feat_full, self.labels_full = adj, features, labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)

        if keep_ratio < 1:
            idx_train, _ = train_test_split(
                idx_train,
                random_state=None,
                train_size=keep_ratio,
                test_size=1 - keep_ratio,
                stratify=labels[idx_train],
            )

        self.adj_train = adj[np.ix_(idx_train, idx_train)]
        self.adj_val = adj[np.ix_(idx_val, idx_val)]
        self.adj_test = adj[np.ix_(idx_test, idx_test)]
        print("size of adj_train:", self.adj_train.shape)
        print("#edges in adj_train:", self.adj_train.sum())

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.feat_train = features[idx_train]
        self.feat_val = features[idx_val]
        self.feat_test = features[idx_test]

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict["class_%s" % i] = self.labels_train == i
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict["class_%s" % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(
                    NeighborSampler(
                        adj,
                        node_idx=node_idx,
                        sizes=sizes,
                        batch_size=num,
                        num_workers=12,
                        return_e_id=False,
                        num_nodes=adj.size(0),
                        shuffle=True,
                    )
                )
        batch = np.random.permutation(self.class_dict2[c])[:num]
        batch = torch.tensor(batch, dtype=torch.long)
        out = self.samplers[c].sample(batch)
        return out

    def sampling(self, ids_per_cls_train, budget, vecs, d, using_half=True):
        budget_dist_compute = 1000
        """
        if using_half:
            vecs = vecs.half()
        """
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)
        vecs = vecs.half()
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            class_ = list(budget.keys())[i]
            other_cls_ids = list(range(len(ids_per_cls_train)))
            other_cls_ids.pop(i)
            ids_selected0 = (
                ids_per_cls_train[i]
                if len(ids_per_cls_train[i]) < budget_dist_compute
                else random.choices(ids_per_cls_train[i], k=budget_dist_compute)
            )

            dist = []
            vecs_0 = vecs[ids_selected0]
            for j in other_cls_ids:
                chosen_ids = random.choices(
                    ids_per_cls_train[j],
                    k=min(budget_dist_compute, len(ids_per_cls_train[j])),
                )
                vecs_1 = vecs[chosen_ids]
                if len(chosen_ids) < 26 or len(ids_selected0) < 26:
                    # torch.cdist throws error for tensor smaller than 26
                    dist.append(torch.cdist(vecs_0.float(), vecs_1.float()).half())
                else:
                    dist.append(torch.cdist(vecs_0, vecs_1))

            # dist = [torch.cdist(vecs[ids_selected0], vecs[random.choices(ids_per_cls_train[j], k=min(budget_dist_compute,len(ids_per_cls_train[j])))]) for j in other_cls_ids]
            dist_ = torch.cat(dist, dim=-1)  # include distance to all the other classes
            n_selected = (dist_ < d).sum(dim=-1)
            rank = n_selected.sort()[1].tolist()
            current_ids_selected = (
                rank[: budget[class_]]
                if len(rank) > budget[class_]
                else random.choices(rank, k=budget[class_])
            )
            ids_selected.extend([ids_per_cls_train[i][j] for j in current_ids_selected])
        return ids_selected

    def retrieve_class_multi_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i in range(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[i])
                    layer_samplers.append(
                        NeighborSampler(
                            adj,
                            node_idx=node_idx,
                            sizes=sizes,
                            batch_size=num,
                            num_workers=12,
                            return_e_id=False,
                            num_nodes=adj.size(0),
                            shuffle=True,
                        )
                    )
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[args.nlayers - 1][c].sample(batch)
        return out

    def retrieve_class_sampler_val(self, transductive, num_per_class=64):
        # num = num_per_class*self.nclass
        if self.class_dict2 is None:
            self.class_dict2 = {}
            node_idx = []
            for i in range(self.nclass):
                if transductive:
                    idx_val = np.array(self.idx_val)
                    idx = idx_val[self.labels_val == i]
                else:
                    idx = np.arange(len(self.labels_val))[self.labels_val == i]
                self.class_dict2[i] = idx
                # node_idx.append(np.random.permutation(self.class_dict2[i])[:num_per_class])
                node_idx += np.random.permutation(self.class_dict2[i])[
                    :num_per_class
                ].tolist()
            self.class_dict2 = None
            return np.array(node_idx).reshape(-1)


def init_feat(feat_syn, args, data, syn_class_indices, transductive=True):
    feature_init = {}
    print(syn_class_indices)
    for c in range(data.nclass):
        features_c = data.feat_train[data.labels_train == c]
        ind = syn_class_indices[c]
        feat_init = init_feat_c(ind[1] - ind[0], features_c, args.init_way)
        feature_init[c] = feat_init
        if feat_init is None:
            feature_init[c] = feat_syn[ind[0] : ind[1]]
        else:
            feat_syn[ind[0] : ind[1]] = torch.tensor(feature_init[c])

    feat_syn = nn.Parameter(feat_syn)

    return feat_syn, feature_init


def init_feat_c(nnodes_syn, feat_real, method):
    if method == "Center":
        kmeans_init = KMeans(n_clusters=1, random_state=42, n_init="auto", verbose=1)
        labels_init = kmeans_init.fit_predict(feat_real)
        feature_init = kmeans_init.cluster_centers_
    elif method == "K-Center":
        kmeans_init = KMeans(
            n_clusters=nnodes_syn, random_state=42, n_init="auto", verbose=1
        )
        labels_init = kmeans_init.fit_predict(feat_real)
        feature_init = kmeans_init.cluster_centers_
    elif method == "Random_real":
        feature_init = np.random.permutation(feat_real)[:nnodes_syn]
    elif method == "K-means":
        kmeans_init = KMeans(nnodes_syn, random_state=42, n_init="auto", verbose=0)
        labels_init = kmeans_init.fit_predict(feat_real)
        random_sample_indices = [
            np.random.choice(np.where(labels_init == i)[0]) for i in range(nnodes_syn)
        ]
        feature_init = feat_real[random_sample_indices]
    else:
        feature_init = None
    return feature_init


def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == "ours" or args.dis_metric == "ctrl":

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(args, gwr, gws)

    elif args.dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("DC error: unknown distance function")

    return dis


def reshape_gw(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = "do nothing"
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])

    return gwr, gws


def distance_wb(args, gwr, gws):
    shape = gwr.shape
    gwr, gws = reshape_gw(gwr, gws)

    if len(shape) == 1:
        return 0

    if args.dis_metric == "ctrl":
        alpha = 1 - args.beta
        beta = args.beta
        if args.dataset in ["ogbn-arxiv"]:
            gradient_sum = torch.sum(torch.abs(gwr))
            threshold = 50
            if gradient_sum < threshold:
                distance = alpha * (
                    1 - F.cosine_similarity(gwr, gws, dim=-1)
                ) + beta * torch.norm(gwr - gws, dim=-1)
                return torch.sum(distance)
            else:
                dis_weight = torch.sum(
                    1
                    - torch.sum(gwr * gws, dim=-1)
                    / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
                )
                return torch.sum(dis_weight)
        elif args.dataset in ["reddit"]:
            gradient_sum = torch.sum(torch.abs(gwr))
            threshold = 50
            if gradient_sum < threshold:
                distance = alpha * (
                    1 - F.cosine_similarity(gwr, gws, dim=-1)
                ) + beta * torch.norm(gwr - gws, dim=-1)
                return torch.sum(distance)
            else:
                dis_weight = torch.sum(
                    1
                    - torch.sum(gwr * gws, dim=-1)
                    / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
                )
                return torch.sum(dis_weight)
        else:
            cosine_similarity = F.cosine_similarity(gwr, gws, dim=-1)
            euclidean_distance = torch.norm(gwr - gws, dim=-1)

            distance = alpha * (1 - cosine_similarity) + beta * euclidean_distance
    else:
        distance = 1 - torch.sum(gwr * gws, dim=-1) / (
            torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001
        )
    return torch.sum(distance)


def calc_f1(y_true, y_pred, is_sigmoid):
    if not is_sigmoid:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(
        y_true, y_pred, average="macro"
    )


def evaluate(output, labels, args):
    data_graphsaint = ["yelp", "ppi", "ppi-large", "flickr", "reddit", "amazon"]
    if args.dataset in data_graphsaint:
        labels = labels.cpu().numpy()
        output = output.cpu().numpy()
        if len(labels.shape) > 1:
            micro, macro = calc_f1(labels, output, is_sigmoid=True)
        else:
            micro, macro = calc_f1(labels, output, is_sigmoid=False)
        print(
            "Test set results:",
            "F1-micro= {:.4f}".format(micro),
            "F1-macro= {:.4f}".format(macro),
        )
    else:
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )
    return


def get_mnist(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    dst_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )  # no        augmentation
    dst_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )
    class_names = [str(c) for c in range(num_classes)]

    labels = []
    feat = []
    for x, y in dst_train:
        feat.append(x.view(1, -1))
        labels.append(y)
    feat = torch.cat(feat, axis=0).numpy()
    from utils.utils_graph import GraphData

    adj = sp.eye(len(feat))
    idx = np.arange(len(feat))
    dpr_data = GraphData(adj - adj, feat, labels, idx, idx, idx)
    from deeprobust.graph.data import Dpr2Pyg

    return Dpr2Pyg(dpr_data)


def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss


def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1)) / n - 0.5)


def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = -torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro


def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum() - thresh)
    # return F.relu(adj.sum()-thresh) / n**2


def feature_smoothing(adj, X):
    adj = (adj.t() + adj) / 2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv + 1e-8
    r_inv = r_inv.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat


def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == "M":  # multiple architectures
        model_eval_pool = [model, "GAT", "MLP", "APPNP", "GraphSage", "Cheby", "GCN"]
    elif eval_mode == "S":  # itself
        model_eval_pool = [model[: model.index("BN")]] if "BN" in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


# neighborhood-based difficulty measurer
def neighborhood_difficulty_measurer(data, adj, label, device="cuda"):
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(
        neighbor_label, 0, 1
    )  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = neighbor_class[data.idx_train]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = (
        -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))
    )  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    print("done")
    return local_difficulty.to(device)


def neighborhood_difficulty_measurer_in(data, adj, label, device="cuda"):
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(
        neighbor_label, 0, 1
    )  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = (
        -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))
    )  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    print("done")
    return local_difficulty.to(device)


def difficulty_measurer(data, adj, label, device="cuda"):
    local_difficulty = neighborhood_difficulty_measurer(data, adj, label, device=device)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes(data, adj, label, device="cuda"):
    node_difficulty = difficulty_measurer(data, adj, label, device=device)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()

    sorted_trainset = data.idx_train[indices]
    return sorted_trainset


def difficulty_measurer_in(data, adj, label):
    local_difficulty = neighborhood_difficulty_measurer_in(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes_in(data, adj, label):
    node_difficulty = difficulty_measurer_in(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()

    return indices


def training_scheduler(lam, t, T, scheduler="geom"):
    if scheduler == "linear":
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == "root":
        return min(1, math.sqrt(lam**2 + (1 - lam**2) * t / T))
    elif scheduler == "geom":
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset == "ogbn-arxiv":
            return 5, 0
        return 1, 0
    if args.dataset in ["ogbn-arxiv"]:
        return args.outer, args.inner
    if args.dataset in ["cora"]:
        return 20, 15  # sgc
    if args.dataset in ["citeseer"]:
        return 20, 15
    if args.dataset in ["physics"]:
        return 20, 10
    else:
        return 20, 10


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    labels = labels.cpu().numpy()
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(
        sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1)
    )
    return parity.item(), equality.item()
