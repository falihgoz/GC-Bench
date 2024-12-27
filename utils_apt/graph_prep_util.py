####################

# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution

####################

from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset

####################

# ref. [1]
def _add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)

# ref. [1]
def _update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

def _get_dummies_theia():
    dummies = {
        "SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2,
        "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, 'PRINCIPAL_LOCAL': 5
    } # Consistent with FLASH [1]
    dummies = {
        "SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2, "NetFlowObject": 3
    }
    
    return dummies

def _get_dummies_cadets():
    dummies = {
        'SUBJECT_PROCESS': 0, 'FILE_OBJECT_FILE': 1, 'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3, 'NetFlowObject': 4, 'FILE_OBJECT_DIR': 5
    } # Consistent with FLASH [1]
    dummies = {
        'SUBJECT_PROCESS': 0, 'FILE_OBJECT_FILE': 1, 'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3, 'NetFlowObject': 4
    }
    
    return dummies

def _get_dummies_trace():
    dummies = {
        "SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_CHAR": 2, "FILE_OBJECT_FILE": 3,
        "FILE_OBJECT_DIR": 4, "SUBJECT_UNIT": 5, "UnnamedPipeObject": 6, "FILE_OBJECT_UNIX_SOCKET": 7,
        "SRCSINK_UNKNOWN": 8, "FILE_OBJECT_LINK": 9, "NetFlowObject": 10, "FILE_OBJECT_BLOCK": 11
    } # Consistent with FLASH [1]
    dummies = {
        "MemoryObject": 1, "FILE_OBJECT_FILE": 3, "FILE_OBJECT_DIR": 4, "SUBJECT_UNIT": 5,
        "FILE_OBJECT_UNIX_SOCKET": 7, "SRCSINK_UNKNOWN": 8, "NetFlowObject": 10
    } # outputs of experiment 'distill_training_epoch_50
    dummies = {
        "MemoryObject": 0, "FILE_OBJECT_DIR": 1, "SUBJECT_UNIT": 2,
        "SRCSINK_UNKNOWN": 3, "NetFlowObject": 4
    }
    
    return dummies

def _get_dummies_fivedirections():
    dummies = {
        'SUBJECT_PROCESS': 0, 'FILE_OBJECT_CHAR': 1, 'VALUE_TYPE_SRC': 2, 'SRCSINK_DATABASE': 3,
        'FILE_OBJECT_UNIX_SOCKET': 4, 'FILE_OBJECT_BLOCK': 5, 'NetFlowObject': 6,
        'SRCSINK_PROCESS_MANAGEMENT': 7, 'SUBJECT_THREAD': 8
    } # Consistent with FLASH [1]
    dummies = {
        'SUBJECT_PROCESS': 0, 'FILE_OBJECT_CHAR': 1, 'NetFlowObject': 6, 'SUBJECT_THREAD': 8
    }
    dummies = {
        'SUBJECT_PROCESS': 0, 'VALUE_TYPE_SRC': 1, 'NetFlowObject': 2, 'SUBJECT_THREAD': 3
    }
    
    return dummies

def _get_properties_from_row(row, action) -> list:
    properties = [row['exec'], action] + ([row['path']] if row['path'] else [])
    
    return properties
def _get_properties_from_row_fivedirections(row, action) -> list:
    properties = []
    if row['exec'] != '':
        properties.append(row['exec'])
    properties.append(action)
    if row['path'] != '':
        properties.append(row['path'])
    
    return properties

# ref. [1], [2]
def prepare_graph(dataset_name: str, df):
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            dummies = _get_dummies_theia()
            _get_properties_function = _get_properties_from_row
        case SupportedDataset.CADETS3.value:
            dummies = _get_dummies_cadets()
            _get_properties_function = _get_properties_from_row
        case SupportedDataset.TRACE3.value:
            dummies = _get_dummies_trace()
            _get_properties_function = _get_properties_from_row
        case SupportedDataset.FIVEDIRECTIONS3.value:
            dummies = _get_dummies_fivedirections()
            _get_properties_function = _get_properties_from_row_fivedirections
        case _:
            raise_unsupported_dataset(dataset_name)
    
    nodes, labels, edges = {}, {}, []
    for _, row in df.iterrows():
        if row['actor_type'] in dummies:
            actor_class_dummy = dummies[row['actor_type']]
        else:
            continue
        if row['object'] in dummies:
            object_class_dummy = dummies[row['object']]
        else:
            continue
        
        action = row["action"]
        properties = _get_properties_function(row, action)
        
        actor_id = row["actorID"]
        _add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = actor_class_dummy
        
        object_id = row["objectID"]
        _add_node_properties(nodes, object_id, properties)
        labels[object_id] = object_class_dummy
        
        # if dataset_name in [SupportedDataset.THEIA3.value, SupportedDataset.CADETS3.value, SupportedDataset.TRACE3.value]:
        #     edges.append((actor_id, object_id))
        # elif dataset_name in [SupportedDataset.FIVEDIRECTIONS3.value]:
        #     edges.append((actor_id, object_id, action))
        edges.append((actor_id, object_id))
    
    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1
    
    _update_edge_index(edges, edge_index, index_map)
    
    print("At the end of function prepare_graph, |nodes|:", len(nodes), ", |labels|:", len(labels), ", |edges|:", len(edges))
    
    mapp = list(index_map.keys())
    return features, feat_labels, edge_index, mapp

# NOTE: Tailor the output in a way it matches with dummy classes.
def graph_number_of_classes(dataset_name: str) -> int:
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            number_of_classes = 4
        case SupportedDataset.CADETS3.value:
            number_of_classes = 5
        case SupportedDataset.TRACE3.value:
            number_of_classes = 5
        case SupportedDataset.FIVEDIRECTIONS3.value:
            number_of_classes = 4
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return number_of_classes
