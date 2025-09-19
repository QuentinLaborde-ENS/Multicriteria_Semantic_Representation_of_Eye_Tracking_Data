# -*- coding: utf-8 -*-

import yaml
import os
import time
import logging
from typing import Optional, Dict, List
from pathlib import Path
from enum import Enum
import traceback
 
import processing.normalization as nm
import processing.segmentation as sg
import processing.symbolization as sy 
import processing.clustering as cl

 

class Dataset(Enum):
    """Supported datasets.""" 
    CLDRIVE = "CLDrive"
    ETRA = "ETRA"
    GAZEBASE = "GazeBase"

class SymbolizationMethod(Enum):
    """Supported symbolization methods.""" 
    KPCA = "kpca"  

class PipelineConfig:
    """Configuration for pipeline paths and modules."""
    def __init__(self, dataset: Dataset):
        """
        Initialize pipeline configuration.

        Args:
            dataset: The dataset to process (e.g, CLDRIVE).
        """
        self.dataset = dataset
        self.config_path = Path(f'configurations/analysis_{dataset.value.lower()}.yaml')
        self.paths = { 
            'features': Path(f'input/{dataset.value}/features'),
            'normalized': Path(f'output/{dataset.value}/normalized_features'),
            'symbolized': Path(f'output/{dataset.value}/symbolization_kpca')
        }
        self.modules = { 
            'normalization': nm,
            'segmentation': sg,
            'symbolization': sy,
            'clustering': cl
            # 'clustering': cl  # Uncomment if clustering module is available
        }
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary output directories if they don’t exist."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class Pipeline:
    """Pipeline for processing eye-tracking data."""
    def __init__(self, dataset: Dataset, method: SymbolizationMethod):
        """
        Initialize the pipeline.

        Args:
            dataset: Dataset to process.
            method: Symbolization method to use.
        """
        self.config = PipelineConfig(dataset)
        self.method = method
        self.steps = {
            'feature_normalization': True,
            'segmentation': True,
            'symbolization':  True,
            'clustering': True
        }
        self._config_cache = None

    def _get_config(self) -> Dict:
        """Load or retrieve cached configuration."""
        if self._config_cache is None:
            self._config_cache = load_config(self.config.config_path)
        return self._config_cache

    def _get_records(self, path: Path, extension: str) -> List[str]:
        """Get list of files with the specified extension from a directory."""
        records = [f for f in os.listdir(path) if f.endswith(extension)]
        if not records:
            raise ValueError(f"No {extension} files found in {path}")
        return records

    def run(self) -> None:
        """Execute the pipeline steps."""
        start_time = time.time()
        config = self._get_config()

        try: 
            if self.steps['feature_normalization']: 
                self._run_feature_normalization(config)

            if self.steps['segmentation']: 
                self._run_segmentation(config)

            if self.steps['symbolization']: 
                self._run_symbolization(config)

            if self.steps['clustering']: 
                self._run_clustering(config)

        except Exception as e: 
            raise

        execution_time = time.time() - start_time
        print(f"Pipeline for {self.config.dataset.value} completed in {execution_time:.2f} seconds")
 
    def _run_feature_normalization(self, config: Dict) -> None:
        """Run feature normalization step."""
        path = self.config.paths['features']
        records = self._get_records(path, '.csv') 
        self.config.modules['normalization'].process(
            config, str(path), records, self.config.dataset.value
        )

    def _run_segmentation(self, config: Dict) -> None:
        """Run segmentation step."""
        path = self.config.paths['normalized']
        records = self._get_records(path, '.csv')
        self.config.modules['segmentation'].process(
            config, str(path), records, self.config.dataset.value
        )

    def _run_symbolization(self, config: Dict) -> None:
        """Run symbolization step."""
        path = self.config.paths['normalized']
        records = self._get_records(path, '.csv')
        self.config.modules['symbolization'].process(
            config, str(path), records, self.config.dataset.value, self.method.value
        )

    def _run_clustering(self, config: Dict) -> None:
        """Run clustering step (placeholder)."""
        path = self.config.paths['symbolized']
        records = self._get_records(path, '.pkl')  
        cl = self.config.modules['clustering'].process(
            config, str(path), records, self.config.dataset.value)
         
        
def main(dataset: Dataset, method: SymbolizationMethod) -> None:
    """Execute the processing pipeline."""
    pipeline = Pipeline(dataset, method)
    pipeline.run()
    
    
    

if __name__ == '__main__':
    # Configuration
    selected_dataset = Dataset.ETRA #Or Dataset.CLDRIVE or Dataset.GAZEBASE
    selected_method = SymbolizationMethod.KPCA
    
    main(selected_dataset, selected_method)
    
  
    
    
    
    
    
    