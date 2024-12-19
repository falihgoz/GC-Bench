import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import subprocess

from utils_apt.dataset_constants import SupportedDataset
from utils_apt.distillation_constants import SupportedDistillationMethods

SAVE_OUTPUT_DIR = "exp_out"

def run_command(command: str, output_file: str):
    try:
        with open(output_file, 'w') as out_file:
            subprocess.run(command, shell=True, stdout=out_file, stderr=out_file)
        
        print("Successfully finished.")
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}")

def main_exp1(reduction_rates : list[float]):
    supported_datasets = [dataset.value for dataset in SupportedDataset]
    supported_distillation_methods = [dist_method.value for dist_method in SupportedDistillationMethods]
    
    for dataset in supported_datasets:
        for method in supported_distillation_methods:
            for reduction_rate in reduction_rates:
                # Distillation:
                command = f"python flash_detection/distillation_main.py --method {method} --dataset {dataset} --reduction_rate {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_1GC_{dataset}_{method}_{reduction_rate}_.txt"
                print(f"Will run command {command} > {out_file}")
                run_command(command, out_file)
                
                # Detector - training:
                command = f"python flash_detection/main.py --dataset {dataset} --mode train --dist_method {method} --dist_ratio {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_2DTC_train_{dataset}_{method}_{reduction_rate}_.txt"
                print(f"Will run command {command} > {out_file}")
                run_command(command, out_file)
                
                # Detector - training:
                command = f"python flash_detection/main.py --dataset {dataset} --mode test --dist_method {method} --dist_ratio {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_3DTC_test_{dataset}_{method}_{reduction_rate}_.txt"
                print(f"Will run command {command} > {out_file}")
                run_command(command, out_file)

main_exp1([0.01])

