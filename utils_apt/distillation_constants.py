from enum import Enum

#### Supported datasets ####

class SupportedDistillationMethods(Enum):
    SGDD = "SGDD"
    GCOND = "GCond"
    GCDM = "GCDM"
    HERDING = "herding"
    KCENTER = "kcenter"
    RANDOM = "random"

def raise_unsupported_distillation_method(distillation_method: str):
    raise ValueError(f"Distillation method {distillation_method} is not supported currently.")

def is_supported_method(distillation_method: str) -> bool:
    return (distillation_method in [method.value for method in SupportedDistillationMethods])

####################
