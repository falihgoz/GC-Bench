## Keep consistent with FLASH [https://github.com/DART-Laboratory/Flash-IDS] and GC-Bench [https://github.com/RingBDStack/GC-Bench] 's folder structure

from typing import Tuple
from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset

# For distillation phase --> using 'train' files
def get_names_of_related_data_files(dataset_name: str) -> Tuple[str, str]:
    txt_processed_source, json_attribute_source = "", ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            txt_processed_source = "theia_train.txt"
            json_attribute_source = "ta1-theia-e3-official-1r.json"
        case SupportedDataset.CADETS3.value:
            txt_processed_source = "cadets_train.txt"
            json_attribute_source = "ta1-cadets-e3-official.json.1"
        case SupportedDataset.TRACE3.value:
            txt_processed_source = "trace_train.txt"
            json_attribute_source = "ta1-trace-e3-official-1.json"
        case SupportedDataset.FIVEDIRECTIONS3.value:
            txt_processed_source = "fivedirections_train.txt"
            json_attribute_source = "ta1-fivedirections-e3-official-2.json"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return txt_processed_source, json_attribute_source

# For evaluating detection phase --> using 'test' files
def get_names_of_test_data_files(dataset_name: str) -> Tuple[str, str]:
    txt_processed_source, json_attribute_source = "", ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            txt_processed_source = "theia_test.txt"
            json_attribute_source = "ta1-theia-e3-official-6r.json.8"
        case SupportedDataset.CADETS3.value:
            txt_processed_source = "cadets_test.txt"
            json_attribute_source = "ta1-cadets-e3-official-2.json"
        case SupportedDataset.TRACE3.value:
            txt_processed_source = "trace_test.txt"
            json_attribute_source = "ta1-trace-e3-official-1.json.4"
        case SupportedDataset.FIVEDIRECTIONS3.value:
            txt_processed_source = "fivedirections_test.txt"
            json_attribute_source = "ta1-fivedirections-e3-official-2.json.23"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return txt_processed_source, json_attribute_source

# For evaluating detection phase
def get_ground_truth_file_path(dataset_name: str) -> str:
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            gt_path = "theia.json"
        case SupportedDataset.CADETS3.value:
            gt_path = "cadets.json"
        case SupportedDataset.TRACE3.value:
            gt_path = "trace.json"
        case SupportedDataset.FIVEDIRECTIONS3.value:
            gt_path = "fivedirections.json"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    GROUND_TRUTH_DIRECTORY_PATH = "ground_truth"
    return f"{GROUND_TRUTH_DIRECTORY_PATH}/{gt_path}"

def w2v_model_save_file(dataset_name: str) -> str:
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            save_model_file = "word2vec_theia_E3.model"
        case SupportedDataset.CADETS3.value:
            save_model_file = "word2vec_cadets_E3.model"
        case SupportedDataset.TRACE3.value:
            save_model_file = "word2vec_trace_E3.model"
        case SupportedDataset.FIVEDIRECTIONS3.value:
            save_model_file = "word2vec_five_E3.model"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    TRAINED_WEIGHTS_DIRECTORY_PATH = "trained_weights"
    return f"{TRAINED_WEIGHTS_DIRECTORY_PATH}/{save_model_file}"

def get_distillion_saved_file_path(dataset_name:str, distillation_method:str, reduction_ratio: int, dist_seed: int) -> (Tuple[str, str, str] | str):
    adj_path     = f"save/{distillation_method}/adj_{dataset_name}_{reduction_ratio}_{dist_seed}.pt"
    feature_path = f"save/{distillation_method}/feat_{dataset_name}_{reduction_ratio}_{dist_seed}.pt"
    label_path   = f"save/{distillation_method}/label_{dataset_name}_{reduction_ratio}_{dist_seed}.pt"
    
    return adj_path, feature_path, label_path

def gnn_model_save_dir_path(dataset_name: str) -> str:
    return f"trained_weights/gnn/{dataset_name}"

def gnn_model_save_file_path(dataset_name: str, epoch: int, distillation_method: str, distillation_rate: float) -> str:
    save_model_file = f"lword2vec_{distillation_method}_{distillation_rate}_gnn_{epoch}_E3.pth"
    
    TRAINED_WEIGHTS_DIRECTORY_PATH = gnn_model_save_dir_path(dataset_name)
    
    return f"{TRAINED_WEIGHTS_DIRECTORY_PATH}/{save_model_file}"

#### Consistent with our project:

def get_save_directory_of_processed_graph_data(dataset_name: str) -> str:
    return f"data/{dataset_name}"

def get_save_graph_data_file_for_distillation_input(dataset_name: str) -> str:
    SAVE_GRAPH_DIRECTORY_PATH = get_save_directory_of_processed_graph_data(dataset_name)
    save_graph_file = "src_processed_graph.pt"
    
    return f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_graph_file}"

def get_save_evaluation_graph_data_file_for_detection_input(dataset_name: str) -> Tuple[str, str, str, str, str]:
    SAVE_GRAPH_DIRECTORY_PATH = get_save_directory_of_processed_graph_data(dataset_name)
    
    save_nodes_file = "test_processed_nodes.npy"
    save_labels_file = "test_processed_labels.json"
    save_edges_file = "test_processed_edges.json"
    save_mapp_file = "test_processed_mapp.json"
    save_allids_file = "test_processed_allids.json"
    
    save_nodes_file = f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_nodes_file}"
    save_labels_file = f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_labels_file}"
    save_edges_file = f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_edges_file}"
    save_mapp_file = f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_mapp_file}"
    save_allids_file = f"{SAVE_GRAPH_DIRECTORY_PATH}/{save_allids_file}"
    
    return save_nodes_file, save_labels_file, save_edges_file, save_mapp_file, save_allids_file
