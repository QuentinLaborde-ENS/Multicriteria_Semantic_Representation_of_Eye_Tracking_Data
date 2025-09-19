# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

 
class Dataset(Enum):
    """Enum for supported datasets.""" 
    CLDRIVE = "CLDrive"
    ETRA = "ETRA"
    GAZEBASE = "GazeBase"

class Segmentation:
    """Class for segmenting eye-tracking feature data across multiple datasets."""
    
    def __init__(self, config: Dict, path: str, records: List[str], dataset: str = "CLDrive"):
        """
        Initialize Segmentation with configuration, path, records, and dataset type.

        Args:
            config: Configuration dictionary with segmentation parameters
            path: Directory path containing feature data files
            records: List of feature record filenames
            dataset: Dataset identifier (e.g., "CLDrive", "ETRA", "GazeBase")
        
        Raises:
            ValueError: If dataset is not supported
        """
        self.config = config
        self.path = Path(path)
        self.records = records
        try:
            self.dataset = Dataset(dataset)
        except ValueError as e: 
            raise ValueError(f"Unsupported dataset: {dataset}. Must be one of {', '.join(d.value for d in Dataset)}")
        
        # Feature processing flags
        self.process_oculomotor = config.get('process_oculomotor', True)
        self.process_scanpath = config.get('process_scanpath', True)
        self.process_aoi = config.get('process_aoi', True)

    def process(self) -> None:
        """Process and segment features for each type."""
         
        if self.process_oculomotor:
            print("Segmenting oculomotor feature series...")
            oculomotor_records = [r for r in self.records if r.split('.')[0].split('_')[-1] == 'oculomotor']
            self._segment_oculomotor(oculomotor_records) 
        
        if self.process_scanpath:
            print("Segmenting scanpath feature series...")
            scanpath_records = [r for r in self.records if r.split('.')[0].split('_')[-1] == 'scanpath']
            self._segment_scanpath_aoi(scanpath_records, 'scanpath') 
        
        if self.process_aoi:
            print("Segmenting AoI feature series...")
            aoi_records = [r for r in self.records if r.split('.')[0].split('_')[-1] == 'aoi']
           
            self._segment_scanpath_aoi(aoi_records, 'aoi') 

    def _segment_oculomotor(self, feature_records: List[str], display: bool = False) -> None:
        """Segment oculomotor features into fixation and saccade components."""
         
        outpath = f'output/{self.dataset.value}/segmentation/'
        Path(outpath).mkdir(parents=True, exist_ok=True)

        for record in feature_records:
            if self._should_process_record(record):
                try:
                    df = pd.read_csv(self.path / record)
                    name = record.split('.')[0]
                    
                    # Segment fixation features
                    df_fix = df[[col for col in df.columns if col.startswith('fix')]]
                    signal_fix = df_fix.to_numpy()
               
                    bkps_fix = self._signal_segmentation(signal_fix, None)
                    if display:
                        self._display_segmentation(signal_fix, bkps_fix, f"{name}_fixationFeatures")
                    bkps_fix.insert(0, 0)
                    np.save(f"{outpath}{name}Fixation.npy", np.array(bkps_fix))
                    
                    # Segment saccade features
                    df_sac = df[[col for col in df.columns if col.startswith('sac')]]
                    signal_sac = df_sac.to_numpy()
                
                    bkps_sac = self._signal_segmentation(signal_sac, None)
                    if display:
                        self._display_segmentation(signal_sac, bkps_sac, f"{name}_saccadeFeatures")
                    bkps_sac.insert(0, 0)
                    np.save(f"{outpath}{name}Saccade.npy", np.array(bkps_sac))  
                except Exception as e:
                    print(f"Error: {e}")

    def _segment_scanpath_aoi(self, feature_records: List[str], type_: str, display: bool = False) -> None:
        """Segment scanpath or AoI features."""
         
        outpath = f'output/{self.dataset.value}/segmentation/'
        Path(outpath).mkdir(parents=True, exist_ok=True)

        for record in feature_records:
            if self._should_process_record(record):
                    
                try:
                    df = pd.read_csv(self.path / record)
                    name = record.split('.')[0]
                    
                    # Prepare signal (skip startTime(s) for full signal) 
                    signal = df.to_numpy()[:, 1:] if type_ == 'aoi' else df[[col for col in df.columns if col.startswith('Sp')]].to_numpy()  
                    bkps = self._signal_segmentation(signal, None)
               
                    if display:
                        self._display_segmentation(signal, bkps, record)
                   
                    bkps.insert(0, 0)
                    np.save(f"{outpath}{name}.npy", np.array(bkps))
                     
                except Exception as e:
                    print(f"Error: {e}")
        return 

    def _should_process_record(self, record: str) -> bool:
        """Check if a record should be processed based on dataset-specific conditions."""
        parts = record.split('.')[0].split('_')
        try: 
            if self.dataset == Dataset.CLDRIVE:
                subject, label = parts[:2]
                return label in self.config['data'].get('label_set', [])
            elif self.dataset == Dataset.ETRA:
                subject, trial, task, condition, stimulus = parts[:5]
                return (subject in self.config['data'].get('subject_set', []) and 
                        task in self.config['data'].get('task_set', []) and 
                        condition in self.config['data'].get('condition_set', []))
            elif self.dataset == Dataset.GAZEBASE:
                subject, trial, task = parts[:3]
                return (task in self.config['data'].get('label_set', []) and 
                        trial in self.config['data'].get('session', []))
        except Exception as e: 
            return False
        return False

    @staticmethod
    def _signal_segmentation(signal: np.ndarray, nb_bkps: Optional[int] = None) -> List[int]:
        """
        Perform signal segmentation using ruptures.

        Args:
            signal: Input signal array
            nb_bkps: Number of breakpoints (if None, use penalty-based method)

        Returns:
            List of breakpoint indices
        """
        try:
            if nb_bkps is not None:
                algo = rpt.KernelCPD(kernel="linear", jump=1).fit(signal)
                return algo.predict(n_bkps=nb_bkps)
            else:
                pen = np.log(signal.shape[0]) / 10
                algo = rpt.Pelt(model="l2", jump=1).fit(signal)
                return algo.predict(pen=pen)
        except Exception as e: 
            return []

    @staticmethod
    def _display_segmentation(signal: np.ndarray, bkps: List[int], name: Optional[str] = None) -> None:
        """Display segmentation results with plots."""
        plt.style.use("seaborn-v0_8")
        
        # Plot signal with breakpoints
        plt.plot(signal)
        for x in bkps[:-1]:
            plt.axvline(x=x-1, color='indianred', linewidth=5, linestyle='dashed')
        if name:
            plt.title(name)
        plt.show()
        plt.clf()
        print((signal)) 
        # Heatmap without breakpoints
        fig, ax = plt.subplots()
        ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
        ax.grid(None)
        ax.set_xlabel("Time windows", fontsize=15)
        ax.set_ylabel("Features", fontsize=15)
        plt.yticks([])
         
        plt.show()
        plt.clf()
        
        # Heatmap with breakpoints
        fig, ax = plt.subplots()
        ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
        ax.grid(None)
        for x in bkps[:-1]:
            ax.axvline(x=x-0.5, color='red', linewidth=3, linestyle='dashed')
        ax.set_xlabel("Time windows", fontsize=22)
        ax.set_ylabel("Features", fontsize=22)
        plt.yticks([])
        plt.xticks(fontsize=16)
        plt.tight_layout()
        if name:
            plt.title(name)
        plt.show()
        plt.clf()

def process(config: Dict, path: str, records: List[str], dataset: str = "CLDrive") -> None:
    """Functional wrapper for segmentation."""
    segmentation = Segmentation(config, path, records, dataset)
    segmentation.process()