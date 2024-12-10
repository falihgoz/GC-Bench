####################

# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution

####################

from itertools import compress
from torch_geometric import utils

from dataset_prep_util import SupportedDataset, raise_unsupported_dataset

####################

# ref. [1]
def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)

# ref. [1]
def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

# ref. [1], [2]
def prepare_graph(dataset_name: str, df):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            nodes, labels, edges = {}, {}, []
            dummies = {"SUBJECT_PROCESS":	0, "MemoryObject":	1, "FILE_OBJECT_BLOCK":	2,
                    "NetFlowObject":	3,"PRINCIPAL_REMOTE":	4,'PRINCIPAL_LOCAL':5}

            for _, row in df.iterrows():
                action = row["action"]
                properties = [row['exec'], action] + ([row['path']] if row['path'] else [])
                
                actor_id = row["actorID"]
                add_node_properties(nodes, actor_id, properties)
                labels[actor_id] = dummies[row['actor_type']]

                object_id = row["objectID"]
                add_node_properties(nodes, object_id, properties)
                labels[object_id] = dummies[row['object']]

                edges.append((actor_id, object_id))

            features, feat_labels, edge_index, index_map = [], [], [[], []], {}
            for node_id, props in nodes.items():
                features.append(props)
                feat_labels.append(labels[node_id])
                index_map[node_id] = len(features) - 1

            update_edge_index(edges, edge_index, index_map)

            print("At the end of function prepare_graph, |nodes|:", len(nodes), ", |labels|:", len(labels), ", |edges|:", len(edges))

            return features, feat_labels, edge_index, list(index_map.keys())
        
        case _:
            raise_unsupported_dataset(dataset_name)

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



