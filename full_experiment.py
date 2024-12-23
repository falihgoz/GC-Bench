import sys
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import subprocess
import datetime
import shutil

from utils_apt.dataset_constants import SupportedDataset
from utils_apt.distillation_constants import SupportedDistillationMethods

SAVE_OUTPUT_DIR = "exp_out"

def _delete_dir_and_its_contents(directory_path: str):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
def _clean_previous_exp_outputs():
    exp_out_dir = f"{SAVE_OUTPUT_DIR}"
    distillation_model_save_dir = "save"
    detection_model_save_dir = "trained_weights/gnn"
    
    _delete_dir_and_its_contents(exp_out_dir)
    _delete_dir_and_its_contents(distillation_model_save_dir)
    _delete_dir_and_its_contents(detection_model_save_dir)

def _run_command(command: str, output_file: str):
    try:
        with open(output_file, 'w') as out_file:
            subprocess.run(command, shell=True, stdout=out_file, stderr=out_file)
        
        print(f"Successfull: {command}")
    except subprocess.CalledProcessError as e:
        with open(output_file, 'w') as out_file:
            out_file.write(f"Command {command} failed due to the following: {e}")
        
        print(f"Failed: {command}")

def _print_time_now():
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"-------------------- {time_now} --------------------")

def _print_separotor_line():
    print("------------------------------------------------------------")

def main_exp1(reduction_rates : list[float]):
    supported_datasets = [dataset.value for dataset in SupportedDataset]
    supported_distillation_methods = [dist_method.value for dist_method in SupportedDistillationMethods]
    
    _clean_previous_exp_outputs()
    
    os.makedirs(f'{SAVE_OUTPUT_DIR}/', exist_ok=True)
    
    for dataset in supported_datasets:
        for method in supported_distillation_methods:
            for reduction_rate in reduction_rates:
                _print_separotor_line()
                
                # Distillation:
                command = f"python flash_detection/distillation_main.py --method {method} --dataset {dataset} --reduction_rate {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_1GC_{dataset}_{method}_{reduction_rate}_.txt"
                _print_time_now()
                print(f"Will run command {command} > {out_file}")
                _run_command(command, out_file)
                _print_time_now()
                
                _print_separotor_line()
                
                # Detector - training:
                command = f"python flash_detection/detection_main.py --dataset {dataset} --mode train --dist_method {method} --dist_ratio {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_2DTC_train_{dataset}_{method}_{reduction_rate}_.txt"
                _print_time_now()
                print(f"Will run command {command} > {out_file}")
                _run_command(command, out_file)
                _print_time_now()
                
                _print_separotor_line()
                
                # Detector - training:
                command = f"python flash_detection/detection_main.py --dataset {dataset} --mode test --dist_method {method} --dist_ratio {reduction_rate}"
                out_file = f"{SAVE_OUTPUT_DIR}/run_3DTC_test_{dataset}_{method}_{reduction_rate}_.txt"
                _print_time_now()
                print(f"Will run command {command} > {out_file}")
                _run_command(command, out_file)
                _print_time_now()
                
                _print_separotor_line()

main_exp1([0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])

