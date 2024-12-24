## Keep consistent with FLASH [https://github.com/DART-Laboratory/Flash-IDS]

from utils_apt.dataset_constants import SupportedDataset, raise_unsupported_dataset

def get_gnn_training_epochs(dataset_name: str) -> int:
    epochs = 0
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            epochs = 20
        case SupportedDataset.CADETS3.value:
            epochs = 22
        case SupportedDataset.TRACE3.value:
            epochs = 20
        case SupportedDataset.FIVEDIRECTIONS3.value:
            epochs = 13
        case _:
            raise_unsupported_dataset()
    
    return epochs

def get_gnn_training_batch_size() -> int:
    return 5000

def get_gnn_triaing_conf_score() -> float:
    return 0.9

def get_gnn_testing_conf_score(dataset_name: str) -> float:
    conf = float('-inf')
    
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            conf = 0.53
        case SupportedDataset.CADETS3.value:
            pass
        case SupportedDataset.TRACE3.value:
            pass
        case SupportedDataset.FIVEDIRECTIONS3.value:
            conf = 0.94
        case _:
            raise_unsupported_dataset()
    
    return conf
