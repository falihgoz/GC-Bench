from enum import Enum

#### Supported datasets ####

class SupportedDataset(Enum):
    THEIA3 = "theia"

def raise_unsupported_dataset(dataset_name: str):
    raise ValueError(f"Dataset {dataset_name} is not supported currently.")

####################
