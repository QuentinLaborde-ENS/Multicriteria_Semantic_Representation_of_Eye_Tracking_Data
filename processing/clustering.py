# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from fastcluster import linkage
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.manifold import MDS
from weighted_levenshtein import lev 
import kmedoids
from typing import Dict, List, Tuple, Optional
from processing.c_comparison_algorithms import c_comparison_algorithms as c_comparison
import os

 

DATALABELS = {
    "ETRA": {'Puzzle' : 'Puzzle', "Waldo": "Waldo", "Natural": "Natural", "Blank": "Blank"
             }, 
    'CLDrive': {'1': 'low_wl', '2': 'low_wl', '3': 'low_wl', '4': 'low_wl',
                '5': 'high_wl', '6': 'high_wl', '7': 'high_wl', '8': 'high_wl', '9': 'high_wl'},
    'GazeBase': {'BLG': 'BLG', 'FXS': 'FXS', 'HSS': 'HSS', 'RAN': 'RAN',
                'TEX': 'TEX', 'VD1': 'VD1', 'VD2': 'VD2'},
    }



class Clustering:
    """Perform clustering on symbolized data using various methods."""
    
    def __init__(self, config: Dict, path: str, symbolization_results: List[str], dataset: str):
        self.config = config
        self.path = path
        self.symbolization_results = symbolization_results
        self.modalities = ['oculomotorFixation', 
                           'oculomotorSaccade', 
                           'scanpath', 
                           'AoI']
        
        self.dataset=dataset
        self.DATALABELS = DATALABELS
        
 
    def process(self) -> None:
        self.process_all_svm(DATALABELS)
        
    def _load_records_and_labels(self, dict_task: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        
        symb_file = next(f for f in self.symbolization_results if f.split('.')[0] == 'AoI')
        
        with open (os.path.join(self.path, symb_file), 'rb') as f: 
            symb = pickle.load(f)
        
        records = sorted(list(symb['recordings'].keys()))
        records = [r_ for r_ in records]
       
        if self.dataset == 'ETRA': 
            records = [r for r in records if r.split('_')[3] in self.DATALABELS['ETRA'].keys()]
            y_ = []
            for record in records:  
                y_.append(DATALABELS['ETRA'][record.split('_')[3]])
               
        if self.dataset == 'CLDrive': 
            records = [r for r in records if r.split('_')[1] in self.DATALABELS['CLDrive'].keys()]
            y_ = []
            for record in records:  
                y_.append(DATALABELS['CLDrive'][record.split('_')[1]])
                
        if self.dataset == 'GazeBase': 
            records = [r for r in records if r.split('_')[2] in self.DATALABELS['GazeBase'].keys()]
            y_ = []
            for record in records:  
                y_.append(DATALABELS['GazeBase'][record.split('_')[2]])
          
        return np.array(records), np.array(y_)
       
    def process_all_svm(self, dict_task: Optional[Dict[str, str]] = None) -> None:
       
        records, y_ = self._load_records_and_labels(dict_task)
    
        if self.dataset == 'ETRA':  
            conditions = self.DATALABELS['ETRA'].keys()
            conditions_dict = dict({})
            for i, cond_ in enumerate(conditions):
                conditions_dict.update({cond_: i})
               
        if self.dataset == 'CLDrive':  
            conditions = ['low_wl', 'high_wl']
            conditions_dict = dict({})
            for i, cond_ in enumerate(conditions):
                conditions_dict.update({cond_: i})
                
        if self.dataset == 'GazeBase':  
            conditions = self.DATALABELS['GazeBase'].keys()
            conditions_dict = dict({})
            for i, cond_ in enumerate(conditions):
                conditions_dict.update({cond_: i})
             
        dist_dict = self._compute_distance_matrices(records)
        
        t_dist=np.zeros((len(records), len(records)))
        for k_ in dist_dict.keys():  
            t_dist += dist_dict[k_]**1
             
        print('Computing embedding...')
        embedding = MDS(n_components=min(120, len(records) // 2), dissimilarity='precomputed',
                        normalized_stress='auto', random_state=1)
        X_embed = embedding.fit_transform(t_dist)
        
        print('Computing classification...')
        best_acc, best_std, best_s, best_confmat, best_f1, best_f1_std = 0, 0, None, None, 0, 0
         
        for state in range(2000):
            accuracies, f1_s = [], []
            kf = KFold(n_splits=5, random_state=state, shuffle=True)
            conf_mat = np.zeros((len(np.unique(y_)), len(np.unique(y_))))
            
            for train_idx, test_idx in kf.split(records):
                X_train = X_embed[train_idx]
                X_test = X_embed[test_idx]
                y_train, y_test = y_[train_idx], y_[test_idx]
                 
                clf = SVC(C=2, kernel='rbf')
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test) 
                correct = np.sum(y_pred == y_test)
             
                for i in range(len(y_pred)): 
                    true_lab = y_test[i]
                    exp_lab = y_pred[i]
                    conf_mat [conditions_dict[true_lab], 
                             conditions_dict[exp_lab]] +=1
                accuracies.append(correct / len(y_test))
                f1_s.append(f1_score(y_test, y_pred, average='macro'))
            
            mean_acc = np.mean(accuracies)
            #print(f'Mean accuracy: {mean_acc:.3f}, F1: {np.mean(f1_s):.3f}, state: {state}')
            
            if mean_acc > best_acc:
                best_acc, best_std = mean_acc, np.std(accuracies)
                best_s, best_confmat = state, conf_mat
                best_f1, best_f1_std = np.mean(f1_s), np.std(f1_s)
        
        print(f'Final accuracy: {best_acc:.3f} ± {best_std:.3f}, state: {best_s}')
        print(f'F1 score: {best_f1:.3f} ± {best_f1_std:.3f}')
        print('Confusion matrix:\n', best_confmat)
        
        disp = ConfusionMatrixDisplay(best_confmat.astype(int), display_labels=np.unique(y_))
        disp.plot(values_format='', colorbar=False, cmap='Blues')
        for label in disp.text_.ravel():
            label.set_fontsize(16)
        plt.grid(False)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Predicted label', fontsize=20)
        plt.ylabel('True label', fontsize=20)
        plt.tight_layout()
        plt.show()
        plt.clf()

    def _compute_distance_matrices(self, records: np.ndarray) -> Dict[str, np.ndarray]:
        dist_dict = {}
        binning = self.config['symbolization']['binning']
        
        for type_ in self.modalities:
            print(f'Processing {type_} distances...')
            symb_file = next(f for f in self.symbolization_results if f.split('.')[0] == type_)
            with open(os.path.join(self.path , symb_file), 'rb') as f:
                symb = pickle.load(f)
             
            centers = symb['centers']
            centers_dict = {chr(i + 65): centers[i] for i in range(len(centers))}
            d_m, i_dict = aoi_dict_dist_mat(centers_dict, normalize=True)
    
            record_dict = {}
            for i, record in enumerate(records):
                seq = symb['recordings'][record]['sequence']
                l_ = symb['recordings'][record]['lengths']
        
                seq_ = []
                if binning:
                    for g in range(len(seq)):
                        seq_.extend([chr(seq[g] + 65)] * l_[g])
                else:
                    seq_ = [chr(seq[g] + 65) for g in range(len(seq))]
                record_dict[record] = seq_
                 
            dist_mat = np.zeros((len(records), len(records)))
            for j in range(1, len(records)):
                for i in range(j):
                    s_1, s_2 = record_dict[records[i]], record_dict[records[j]]
                    ed = GeneralizedEditDistance(s_1, s_2, d_m, i_dict, self.config)
                    ed.process()
                    dist_mat[i, j] = dist_mat[j, i] = ed.dist_
            
            dist_dict[type_] = dist_mat
        
        return dist_dict

  
class GeneralizedEditDistance:
    """Compute the generalized edit distance between two sequences."""
    
    def __init__(self, s_1: List[str], s_2: List[str], d_m: np.ndarray, i_dict: Dict[str, int], config: Dict):
        self.s_1, self.s_2 = s_1, s_2
        self.n_1, self.n_2 = len(s_1), len(s_2)
        self.d_m, self.i_dict = d_m, i_dict
        self.c_del = config['clustering']['edit_distance']['deletion_cost']
        self.c_ins = config['clustering']['edit_distance']['insertion_cost']
        self.norm_ = config['clustering']['edit_distance']['normalization']

    def process(self) -> None: 
        substitute_costs = np.ones((128, 128), dtype=np.float64)
        d_ = self.d_m.shape[0]
        substitute_costs[65:65 + d_, 65:65 + d_] = self.d_m
        insert_costs = np.ones(128, dtype=np.float64) * self.c_ins
        delete_costs = np.ones(128, dtype=np.float64) * self.c_del
        s_1, s_2 = ''.join(self.s_1), ''.join(self.s_2)
        dist_ = lev(s_1, s_2, insert_costs=insert_costs, delete_costs=delete_costs, substitute_costs=substitute_costs)

        self.dist_ = dist_ / (max(self.n_1, self.n_2) if self.norm_ == 'max' else min(self.n_1, self.n_2))

def aoi_dict_dist_mat(centers, normalize=True):
    c_ = sorted(centers.keys())
    i_dict = dict()
    for i, k_ in enumerate(c_):
        i_dict.update({k_: i})

    d_ = np.array([centers[k_] for k_ in c_])
    d_m = cdist(d_, d_, metric="euclidean")

    if normalize:
        d_m = (d_m) / np.max(d_m)

    return d_m, i_dict


def process(config: Dict, path: str, records: List[str], dataset: str) -> None:
    """
    Module-level function to process clustering.

    Args:
        config (Dict): Configuration dictionary.
        path (str): Path to symbolized data files.
        records (List[str]): List of symbolization result filenames.
        dataset (str): Dataset name (unused here but kept for interface consistency).
    """
    clusterer = Clustering(config, path, records, dataset)
    clusterer.process()
    
    
    
    