from enum import Enum

#### Supported datasets ####

class SupportedDistillationMethods(Enum):
    GCDM = "GCDM"

def raise_unsupported_distillation_method(distillation_method: str):
    raise ValueError(f"Distillation method {distillation_method} is not supported currently.")

def is_modern_distillation(distillation_method: str) -> bool:
    return (
        distillation_method in [
            SupportedDistillationMethods.GCDM.value
        ]
    )

####################
