import torch
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS


# ref. [1]
class GCN(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = SAGEConv(in_channel, 32, normalize=True)
        self.conv2 = SAGEConv(32, out_channel, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x
