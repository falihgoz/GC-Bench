from enum import Enum

#### Supported datasets ####

class SupportedDataset(Enum):
    THEIA3 = "theia"
    CADETS3 = "cadets"
    TRACE3 = "trace"
    FIVEDIRECTIONS3 = "fivedirections"

def raise_unsupported_dataset(dataset_name: str):
    raise ValueError(f"Dataset {dataset_name} is not supported currently.")

def is_supported_dataset(dataset_name: str) -> bool:
    return (dataset_name in [supported_dataset.value for supported_dataset in SupportedDataset])

####################
