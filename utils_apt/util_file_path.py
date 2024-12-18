## Keep consistent with FLASH [https://github.com/DART-Laboratory/Flash-IDS] and GC-Bench's folder structure

from typing import Tuple
from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset
from utils_apt.distillation_constants import raise_unsupported_distillation_method, is_modern_distillation


def get_names_of_related_data_files(dataset_name: str) -> Tuple[str, str]:
    txt_processed_source, json_attribute_source = "", ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            txt_processed_source = "theia_train.txt"
            json_attribute_source = "ta1-theia-e3-official-1r.json"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return txt_processed_source, json_attribute_source

def get_names_of_test_data_files(dataset_name: str) -> Tuple[str, str]:
    txt_processed_source, json_attribute_source = "", ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            txt_processed_source = "theia_test.txt"
            json_attribute_source = "ta1-theia-e3-official-6r.json.8"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return txt_processed_source, json_attribute_source

def get_ground_truth_file_path(dataset_name: str) -> str:
    gt_path = ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            gt_path = "ground_truth/theia.json"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return gt_path

def w2v_model_save_file(dataset_name: str) -> str:
    save_model_file = ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            save_model_file = "trained_weights/theia/word2vec_theia_E3.model"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return save_model_file

def get_distillion_saved_file_path(dataset_name:str, distillation_method:str, reduction_ratio: int) -> (Tuple[str, str, str] | str):
    magic_number = 15 # Let's figure it out later. It's based on observation that all saved files end with *_15.pt!
    if is_modern_distillation(distillation_method):
        adj_path     = f"save/{distillation_method}/adj_{dataset_name}_{reduction_ratio}_{magic_number}.pt"
        feature_path = f"save/{distillation_method}/feat_{dataset_name}_{reduction_ratio}_{magic_number}.pt"
        label_path   = f"save/{distillation_method}/label_{dataset_name}_{reduction_ratio}_{magic_number}.pt"
        
        return adj_path, feature_path, label_path
    # elif <traditional methods> return selecet_idx_path
    else:
        raise_unsupported_distillation_method()

def gnn_model_save_file_path(dataset_name: str, epoch: int) -> str:
    save_model_file = ""
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            save_model_file = f"trained_weights/theia/lword2vec_gnn_theia{epoch}_E3.pth"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return save_model_file
