# -*- coding: utf-8 -*-
import yaml
import argparse

from processing.pipeline_kfold import run_crossval_pipeline
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run symbolic eye-tracking pipeline")
     
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use the ETRA dataset with binary task"
    )
    parser.add_argument(
        "--ternary",
        action="store_true",
        help="Use the CLDrive dataset with ternary task"
    )
   
    args = parser.parse_args()
     
    if args.binary:
        task = 'binary' 
    elif args.ternary:
        task = 'ternary' 
    else:
        raise ValueError("Please specify one task using --binary or --ternary")
     
    with open('configuration/analysis_cldrive.yaml', 'r') as file:
        config = yaml.safe_load(file)
    path = 'input/CLDrive/features/'   
        
    run_crossval_pipeline(config, path, task)
    
    
    
    
    
    