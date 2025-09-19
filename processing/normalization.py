# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
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

class Normalization:
    """Class for normalizing eye-tracking feature data across multiple datasets."""
    
    def __init__(self, config: Dict, path: str, records: List[str], dataset: str = "CLDrive"):
        """
        Initialize Normalization with configuration, path, records, and dataset type.

        Args:
            config: Configuration dictionary with normalization parameters
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
        """Process and filter records, then normalize features for each type."""
        
        # Filter records based on dataset-specific logic
        to_keep = self._filter_records()
        
        # Process each feature type if enabled
        if self.process_oculomotor:
            oculomotor_records = [r + '_oculomotor.csv' for r in to_keep]
            self._process_normalization(oculomotor_records, 'oculomotor')
        if self.process_scanpath:
            scanpath_records = [r + '_scanpath.csv' for r in to_keep]
            self._process_normalization(scanpath_records, 'scanpath')
        if self.process_aoi:
            aoi_records = [r + '_aoi.csv' for r in to_keep]
            self._process_normalization(aoi_records, 'aoi')

    def _filter_records(self) -> List[str]:
        """Filter records based on availability and segment proportion threshold."""
        to_keep = []
        thr = self.config['general'].get('available_segment_prop', 0.5)
 
        if self.dataset == Dataset.CLDRIVE:
            for subject in self.config['data'].get('subjects', []):
                for label in self.config['data'].get('label_set', []):
                    self._check_and_append(subject, label, to_keep, thr)
        elif self.dataset == Dataset.ETRA: 
            records_ = ["_".join(r.split('_')[:5]) for r in self.records]
            for record in records_:
                self._check_and_append_etra(record, to_keep, thr) 
        elif self.dataset == Dataset.GAZEBASE:
            for subject in range(323):  # GazeBase subject range
                for session in self.config['data'].get('session', []):
                    for label in self.config['data'].get('label_set', []):
                        self._check_and_append(f"{subject}_{session}", label, to_keep, thr)
        
        return to_keep

    def _check_and_append_etra(self, label: str, to_keep: List[str], thr: float) -> None:
        """Check if a record meets the segment proportion threshold and append if valid."""
        try:
            df_o = pd.read_csv(self.path / f'{label}_oculomotor.csv')
            l_o = np.count_nonzero(~np.isnan(df_o.iloc[:, 1].to_numpy())) / len(df_o)
            df_s = pd.read_csv(self.path / f'{label}_scanpath.csv')
            l_s = np.count_nonzero(~np.isnan(df_s.iloc[:, 1].to_numpy())) / len(df_s)
            df_a = pd.read_csv(self.path / f'{label}_AoI.csv')
            l_a = np.count_nonzero(~np.isnan(df_a.iloc[:, 1].to_numpy())) / len(df_a)
            
            if l_o >= thr and l_s >= thr and l_a >= thr:
                to_keep.append(f'{label}') 
        except Exception as e:
            print(f"Error: {e}")
            
    def _check_and_append(self, subject: str, label: str, to_keep: List[str], thr: float) -> None:
        """Check if a record meets the segment proportion threshold and append if valid."""
        try:
            df_o = pd.read_csv(self.path / f'{subject}_{label}_oculomotor.csv')
            l_o = np.count_nonzero(~np.isnan(df_o.iloc[:, 1].to_numpy())) / len(df_o)
            df_s = pd.read_csv(self.path / f'{subject}_{label}_scanpath.csv')
            l_s = np.count_nonzero(~np.isnan(df_s.iloc[:, 1].to_numpy())) / len(df_s)
            df_a = pd.read_csv(self.path / f'{subject}_{label}_AoI.csv')
            l_a = np.count_nonzero(~np.isnan(df_a.iloc[:, 1].to_numpy())) / len(df_a)
            
            if l_o >= thr and l_s >= thr and l_a >= thr:
                to_keep.append(f'{subject}_{label}') 
        except Exception as e:
            print(f"Error: {e}")

    def _process_normalization(self, feature_records: List[str], type_: str) -> None:
        """Normalize features for a given type (oculomotor, scanpath, or AoI)."""
        dict_methods = {
            'log_normal': self.lognormal_uniformization,
            'empirical': self.empirical_cdf
        }
        
        data = {}
        dict_norm = {}
        print(f'Normalizing {type_} features...')
        features = self.config['data'][f'{type_}_features']

        # Load and preprocess data
        for record in feature_records:
            if self._should_process_record(record):
                try:
                    df = pd.read_csv(self.path / record)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df = df.interpolate(axis=0).ffill().bfill()
                    data[record.split('.')[0]] = df
                except Exception as e:
                    print(f"Error: {e}")

        # Compute normalization parameters
        norm_method = self.config['symbolization']['normalization']
        norm_type = self.config['symbolization']['normalization_method']

        if norm_method == 'longitudinal':
            for subject in self._get_subjects():
                dict_norm[subject] = {}
                for feature in features:
                    if feature != 'startTime(s)':
                        ts = self._concatenate_feature(data, feature, subject)
                        if ts:
                            name = f"{subject}_{feature}"
                            feat_params = dict_methods[norm_type](ts, name)
                            dict_norm[subject][feature] = feat_params
        elif norm_method == 'all':
            for feature in features:
                if feature != 'startTime(s)':
                    ts = self._concatenate_feature(data, feature)
                    if ts:
                        name = feature
                        feat_params = dict_methods[norm_type](ts, name)
                        dict_norm[feature] = feat_params

        # Normalize and save
        for file in data.keys(): 
            l_data = data[file]
            new_data = {}
            
            for feature in features:
                ts = l_data[feature].values
                if feature != 'startTime(s)':
                    subject = file.split('_')[0]
                    if norm_method == 'longitudinal' and subject in dict_norm:
                        params = dict_norm[subject][feature]
                    elif norm_method == 'all' and feature in dict_norm:
                        params = dict_norm[feature]
                   
                    if norm_type == 'log_normal':
                        ts_n = sp.stats.lognorm.cdf(ts, params[0], loc=params[1], scale=params[2])
                    elif norm_type == 'empirical':
                        ts_n = params.evaluate(ts)
                    new_data[feature] = ts_n
                else:
                    new_data[feature] = ts
            
            new_df = pd.DataFrame.from_dict(new_data)
            filename = f'output/{self.dataset.value}/normalized_features/{file}.csv'
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(filename, index=False) 

    def _should_process_record(self, record: str) -> bool:
        """Check if a record should be processed based on dataset-specific conditions."""
        parts = record.split('.')[0].split('_') 
        if self.dataset == Dataset.CLDRIVE:
            subject, label = parts[:2]
            return label in self.config['data'].get('label_set', [])
        elif self.dataset == Dataset.ETRA:
            subject, trial, task, condition, stimulus = parts[:5]
            return (subject in self.config['data'].get('subject_set', []) and 
                    task in self.config['data'].get('task_set', []) and 
                    condition in self.config['data'].get('condition_set', []))
        elif self.dataset == Dataset.GAZEBASE:
            subject, session, task = parts[:3] 
            return (task in self.config['data'].get('label_set', []) and 
                    session in self.config['data'].get('session', []))
        return False

    def _get_subjects(self) -> List[str]:
        """Get list of subjects based on dataset.""" 
        if self.dataset == Dataset.CLDRIVE:
            return self.config['data'].get('subjects', [])
        elif self.dataset == Dataset.ETRA:
            return self.config['data'].get('subject_set', [])
        elif self.dataset == Dataset.GAZEBASE:
            return [str(i) for i in range(323)]  # GazeBase subject range

    def _concatenate_feature(self, data: Dict, feature: str, subject: Optional[str] = None) -> List[float]:
        """Concatenate feature data across files, optionally for a specific subject."""
        ts = []
        for file in data.keys():
            if subject is None or file.split('_')[0] == subject:
                l_data = data[file]
                ts.extend(l_data[feature].values)
        return ts

    @staticmethod
    def lognormal_uniformization(time_series: List[float], name: Optional[str] = None) -> tuple:
        """Fit a log-normal distribution and return parameters."""
        try:
            param = sp.stats.lognorm.fit(time_series)
            x = np.linspace(0, max(time_series), 250)
            pdf_fitted = sp.stats.lognorm.pdf(x, param[0], loc=param[1], scale=param[2])
            
            plt.style.use("seaborn-v0_8")
            plt.hist(time_series, bins=50, alpha=0.3, density=True)
            plt.plot(x, pdf_fitted, 'r-')
            if name:
                plt.title(name.split('_')[-1])
                fig = plt.gcf()
                path = f'output/{Dataset(name.split("_")[0]).value}/figures/normalization/'
                Path(path).mkdir(parents=True, exist_ok=True)
                fig.savefig(path + name)
            plt.show()
            plt.clf()
        except Exception as e:
            print(f"Error: {e}")
        return param

    @staticmethod
    def empirical_cdf(time_series: List[float], name: Optional[str] = None, display: bool = False) -> object:
        """Compute empirical CDF for a time series."""
        res = stats.ecdf(time_series)
        ecdf = res.cdf
        
        if display:
            plt.style.use("seaborn-v0_8")
            plt.hist(time_series, bins=50, alpha=0.3, density=True)
            if name:
                plt.title(name.split('_')[-1])
                fig = plt.gcf()
                path = f'output/{Dataset(name.split("_")[0]).value}/figures/normalization/'
                Path(path).mkdir(parents=True, exist_ok=True)
                fig.savefig(path + name)
            plt.show()
            plt.clf()
        return ecdf

def process(config: Dict, path: str, records: List[str], dataset: str = "CLDrive") -> None:
    """Functional wrapper for normalization.""" 
    normalization = Normalization(config, path, records, dataset)
    normalization.process()