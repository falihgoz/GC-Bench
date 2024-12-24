import os
import sys
import torch
import numpy as np
import json

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from utils_apt.util_dataset import get_graph_dataset
from utils_apt.dataset_prep_util import prep_dataframe
from utils_apt.graph_prep_util import prepare_graph
from utils_apt.w2v_util import PositionalEncoder, FLASH_W2V_DIMENSION, load_w2v_model, w2v_infer
from utils_apt.util_file_path import get_names_of_test_data_files, w2v_model_save_file, get_save_directory_of_processed_graph_data, get_save_graph_data_file_for_distillation_input, get_save_evaluation_graph_data_file_for_detection_input
from utils_apt.dataset_constants import SupportedDataset

def _get_test_graph_data(dataset_name: str):
    txt_processed_source, json_attribute_source = get_names_of_test_data_files(dataset_name)
    print(f"source data files: {txt_processed_source}, {json_attribute_source}")
    
    # data_frame: <class 'pandas.core.frame.DataFrame'>
    data_frame = prep_dataframe(dataset_name, txt_processed_source, json_attribute_source)
    # node_features: <class 'list'>, labels: <class 'list'>, edges: <class 'list'>, mapp: <class 'list'>
    node_features, labels, edges, mapp = prepare_graph(dataset_name, data_frame)
    
    encoder = PositionalEncoder(FLASH_W2V_DIMENSION)
    w2v_model_file = w2v_model_save_file(dataset_name)
    w2v_model = load_w2v_model(w2v_model_file)
    print(f"w2v model loaded from {w2v_model_file}")
    
    nodes = [w2v_infer(x, w2v_model, encoder) for x in node_features] # <class 'list'>
    nodes = np.array(nodes) # <class 'numpy.ndarray'>
    print(f"Number of nodes: {len(nodes)}")
    
    all_ids = list(data_frame['actorID']) + list(data_frame['objectID'])
    all_ids = set(all_ids)
    print(f"Length of all_ids: {len(all_ids)}")
    
    return nodes, labels, edges, mapp, all_ids


def _save_original_apt_graph_to_file_for_distillation(dataset_name: str, apt_graph):
    save_dir = get_save_directory_of_processed_graph_data(dataset_name)
    save_file = get_save_graph_data_file_for_distillation_input(dataset_name)
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(apt_graph, save_file)

def _save_original_apt_graph_to_file_for_detection_eval(
    dataset_name: str, nodes:np.ndarray, labels: list, edges: list, mapp: list, all_ids: set
):
    save_dir = get_save_directory_of_processed_graph_data(dataset_name)
    save_nodes_file, save_labels_file, save_edges_file, save_mapp_file, save_allids_file = get_save_evaluation_graph_data_file_for_detection_input(dataset_name)
    
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(save_nodes_file, nodes)
    with open(save_labels_file, 'w') as f:
        json.dump(labels, f)
    with open(save_edges_file, 'w') as f:
        json.dump(edges, f)
    with open(save_mapp_file, 'w') as f:
        json.dump(mapp, f)
    with open(save_allids_file, 'w') as f:
        json.dump(list(all_ids), f)

def main():
    supported_datasets = [dataset.value for dataset in SupportedDataset]
    for dataset in supported_datasets:
        print(f"**** Processing dataset {dataset}: START ****")
        apt_graph = get_graph_dataset(dataset)
        _save_original_apt_graph_to_file_for_distillation(dataset, apt_graph)
        
        nodes, labels, edges, mapp, all_ids = _get_test_graph_data(dataset)
        _save_original_apt_graph_to_file_for_detection_eval(dataset, nodes, labels, edges, mapp, all_ids)
        print(f"**** Processing dataset {dataset}: DONE ****")


main()
