# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, squareform
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from weighted_levenshtein import lev
import ruptures as rpt
from fastcluster import linkage
import matplotlib.pyplot as plt
import yaml

from joblib import Parallel, delayed


# ---------------------------------------------------------------------
# --------------------- Common small utilities ------------------------
# ---------------------------------------------------------------------

def seriation(Z, N, cur_index):
    """
    Recover the leaf ordering from a hierarchical clustering tree (dendrogram).
    Depth-first traversal.
    """
    if cur_index < N:
        return [cur_index]
    left = int(Z[cur_index - N, 0])
    right = int(Z[cur_index - N, 1])
    return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """
    Reorder cluster centers according to hierarchical structure.

    dist_mat: pairwise distances between cluster centers
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)

    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[
        [res_order[i] for i in a],
        [res_order[j] for j in b]
    ]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage


def generalized_edit_distance(seq1, seq2, d_m, deletion_cost, insertion_cost):
    """
    Weighted Levenshtein distance between two symbolic sequences.

    - seq1, seq2: lists like ['A','A','B','C',...]
    - d_m: (KxK) substitution cost matrix between cluster centroids
    - insertion_cost / deletion_cost: scalars

    We embed d_m in a 128x128 substitution cost matrix (ASCII A=65,...),
    then call weighted_levenshtein. Result is normalized by max len.
    """
    n1, n2 = len(seq1), len(seq2)

    sub_costs = np.ones((128, 128), dtype=np.float64)
    K = d_m.shape[0]
    sub_costs[65:65+K, 65:65+K] = d_m

    ins_costs = np.ones(128, dtype=np.float64) * insertion_cost
    del_costs = np.ones(128, dtype=np.float64) * deletion_cost

    s1 = ''.join(seq1)
    s2 = ''.join(seq2)

    dist_ = lev(
        s1,
        s2,
        insert_costs=ins_costs,
        delete_costs=del_costs,
        substitute_costs=sub_costs
    )
    return dist_ / max(n1, n2)


def _normalize_distance_train(D):
    """
    Median-based scaling on the TRAIN×TRAIN block.
    Returns normalized matrix + the scale so we can reuse on test.
    """
    tri = D[np.triu_indices_from(D, 1)]
    m = np.median(tri)
    scale = m if m > 0 else 1.0
    return D / scale, scale


def _apply_scale_to_test_block(D_te_tr, scale):
    """Apply train-derived scale to TEST×TRAIN block."""
    return D_te_tr / (scale if scale > 0 else 1.0)


def classical_mds_fit(D_tr, k):
    """
    Classical MDS (Torgerson) fit on TRAIN distances.
    Returns:
      - X_tr   : embedding of train samples
      - cache  : Nyström projection info for test
    """
    n = D_tr.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D_tr ** 2
    B = -0.5 * (J @ D2 @ J)

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    pos = eigvals > 1e-12
    eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]

    k_eff = min(k, eigvals.size) if eigvals.size > 0 else 2
    eigvals, eigvecs = eigvals[:k_eff], eigvecs[:, :k_eff]

    X_tr = eigvecs * np.sqrt(eigvals)

    row_mean = D2.mean(axis=1)
    grand = D2.mean()

    cache = {
        'V': eigvecs,
        'L': eigvals,
        'row_mean': row_mean,
        'grand': grand
    }
    return X_tr, cache


def classical_mds_transform(D_te_tr, cache):
    """
    Nyström projection: embed TEST samples in the TRAIN MDS space.
    """
    V = cache['V']
    L = cache['L']
    row_mean = cache['row_mean']
    grand = cache['grand']

    d2 = D_te_tr ** 2                  # (Nte x Ntr)
    mean_t = d2.mean(axis=1, keepdims=True)  # (Nte x 1)
    B_t = -0.5 * (d2 - row_mean[None, :] - mean_t + grand)
    X_te = B_t @ (V / (np.sqrt(L)[None, :] + 1e-12))
    return X_te


# ---------------------------------------------------------------------
# 1. Load raw features (oculomotor / scanpath / AoI ONLY)
# ---------------------------------------------------------------------

def load_raw_features(config, path):
    """
    Load per-modality CSVs and keep only (subject,label) pairs that
    satisfy availability for ALL required modalities:
      - 'oculomotor'
      - 'scanpath'
      - 'AoI'
    (EDA/ECG removed)
    """
    thr = config['general']['available_segment_prop']

    # Here we keep your original hardcoded subject list and label_set,
    # since your code was doing that instead of reading config['data'].
    subjects = [
        '1030', '1105', '1106', '1241', '1271', '1314', '1323',
        '1337', '1372', '1417', '1434', '1544', '1547', '1595',
        '1629', '1716', '1717', '1744', '1868', '1892', '1953',
    ]
    label_set = ['1','2','3','4','5','6','7','8','9']

    kept_records = []
    for subject in subjects:
        for label in label_set:
            ok = True
            modality_files = {
                'oculomotor': f"{subject}_{label}_oculomotor.csv",
                'scanpath':   f"{subject}_{label}_scanpath.csv",
                'AoI':        f"{subject}_{label}_AoI.csv",
            }
            for mod, fname in modality_files.items():
                try:
                    df_mod = pd.read_csv(path + fname)
                    df_mod = df_mod.interpolate(axis=0).ffill().bfill()
                    prop_avail = (
                        np.count_nonzero(~np.isnan(df_mod.iloc[:, 1].to_numpy()))
                        / len(df_mod)
                    )
                    if prop_avail < thr:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break

            if ok:
                kept_records.append(f"{subject}_{label}")

    # Reload clean dataframes for the kept records
    raw_data = {}
    for rec in kept_records:
        subject, label = rec.split('_')
        for modality in ['oculomotor', 'scanpath', 'AoI']:
            fname = f"{subject}_{label}_{modality}.csv"
            try:
                df_mod = pd.read_csv(path + fname)
                df_mod = df_mod.interpolate(axis=0).ffill().bfill()
                raw_data[(subject, label, modality)] = df_mod
            except Exception:
                pass

    return raw_data


# ---------------------------------------------------------------------
# 2. ECDF normalizer (train-only fit, apply to train/test)
# ---------------------------------------------------------------------

class ECDFNormalizer:
    """
    Learn monotonic CDF transforms per feature, per modality,
    from TRAIN recordings only. Then apply to train and test.
    """

    def __init__(self, config):
        self.config = config
        self.norm_map = {}

    def _feature_list_for_modality(self, modality):
        if modality == 'oculomotor':
            return self.config['data']['oculomotor_features']
        elif modality == 'scanpath':
            return self.config['data']['scanpath_features']
        elif modality == 'AoI':
            return self.config['data']['aoi_features']
        else:
            raise ValueError(f"Unknown modality {modality}")

    def fit(self, raw_data, train_records):
        """
        Build an ECDF for each feature of each modality by pooling values
        across TRAIN recordings only.
        """
        self.norm_map = {}
        for modality in ['oculomotor', 'scanpath', 'AoI']:
            feats = self._feature_list_for_modality(modality)
            for feat in feats:
                if feat == 'startTime(s)':
                    continue
                vals_all = []
                for (subj, lab) in train_records:
                    key = (subj, lab, modality)
                    if key not in raw_data:
                        continue
                    arr = raw_data[key][feat].to_numpy()
                    vals_all.extend(list(arr))
                if len(vals_all) == 0:
                    continue
                ecdf_obj = stats.ecdf(vals_all).cdf
                self.norm_map[(modality, feat)] = ecdf_obj

    def transform(self, raw_data, records_subset):
        """
        Apply learned ECDFs to produce normalized trajectories.
        Returns dict[(subj, lab, modality)] = normalized DataFrame
        """
        norm_data = {}
        for (subj, lab) in records_subset:
            for modality in ['oculomotor', 'scanpath', 'AoI']:
                key = (subj, lab, modality)
                if key not in raw_data:
                    continue

                df_in = raw_data[key]
                feats = self._feature_list_for_modality(modality)

                new_cols = {}
                for feat in feats:
                    if feat == 'startTime(s)':
                        # keep time axis raw
                        new_cols[feat] = df_in[feat].to_numpy()
                    else:
                        x = df_in[feat].to_numpy()
                        ecdf_fun = self.norm_map.get((modality, feat), None)
                        if ecdf_fun is None:
                            new_cols[feat] = x
                        else:
                            new_cols[feat] = ecdf_fun.evaluate(x)

                df_out = pd.DataFrame(new_cols)
                norm_data[(subj, lab, modality)] = df_out
        return norm_data


# ---------------------------------------------------------------------
# 3. Segmentation (per recording, no leakage)
# ---------------------------------------------------------------------

class Segmenter:
    """
    Ruptures/PELT segmentation on each (already normalized) recording,
    independently for each modality.
    """

    def __init__(self):
        pass

    def _segment_signal(self, X):
        pen = np.log(X.shape[0]) / 10.0
        algo = rpt.Pelt(model='l2', jump=1).fit(X)
        bkps = algo.predict(pen=pen)
        bkps.insert(0, 0)
        return bkps

    def segment_records(self, norm_data):
        """
        norm_data[(subj, lab, modality)] = normalized df

        Returns:
            segments[(subj, lab, submodality)] = breakpoints list
        where submodality ∈ {
            'oculomotorFixation', 'oculomotorSaccade',
            'scanpath', 'AoI'
        }
        """
        segments = {}
        for (subj, lab, modality), df_mod in norm_data.items():
            if modality == 'oculomotor':
                # fixation-only columns
                fix_cols = [c for c in df_mod.columns if c.startswith('fix')]
                if len(fix_cols) > 0:
                    X_fix = df_mod[fix_cols].to_numpy()
                    bkps_fix = self._segment_signal(X_fix)
                    segments[(subj, lab, 'oculomotorFixation')] = bkps_fix

                # saccade-only columns
                sac_cols = [c for c in df_mod.columns if c.startswith('sac')]
                if len(sac_cols) > 0:
                    X_sac = df_mod[sac_cols].to_numpy()
                    bkps_sac = self._segment_signal(X_sac)
                    segments[(subj, lab, 'oculomotorSaccade')] = bkps_sac

            else:
                # single segmentation on all non-time cols
                feat_cols = [c for c in df_mod.columns if c != 'startTime(s)']
                if len(feat_cols) == 0:
                    continue
                X_all = df_mod[feat_cols].to_numpy()
                bkps_all = self._segment_signal(X_all)
                segments[(subj, lab, modality)] = bkps_all

        return segments


# ---------------------------------------------------------------------
# 4. Symbolizer (KernelPCA + KMeans on TRAIN only)
# ---------------------------------------------------------------------

class Symbolizer:
    """
    Convert segmented trajectories into sequences of discrete symbols.
    Fit is TRAIN-only (KernelPCA+KMeans+cluster ordering).
    """

    def __init__(self, config):
        self.config = config
        self.models = {}

    def _n_centers(self, submodality):
        if submodality == 'scanpath':
            return self.config['symbolization']['nb_clusters']['scanpath']
        elif submodality == 'AoI':
            return self.config['symbolization']['nb_clusters']['aoi']
        else:
            # 'oculomotorFixation' / 'oculomotorSaccade'
            return self.config['symbolization']['nb_clusters']['oculomotor']

    def _features_for_submodality(self, submodality, df_mod):
        """
        Return feature columns relevant to that sub-modality.
        """
        if submodality == 'oculomotorFixation':
            cols = [c for c in df_mod.columns if c.startswith('fix')]
        elif submodality == 'oculomotorSaccade':
            cols = [c for c in df_mod.columns if c.startswith('sac')]
        elif submodality == 'scanpath':
            cols = [c for c in df_mod.columns if c.startswith('Sp')]
        elif submodality == 'AoI':
            cols = [c for c in df_mod.columns if c.startswith('AoI')]
        else:
            cols = [c for c in df_mod.columns if c != 'startTime(s)']
        return cols

    def _collect_segment_means(self, norm_data, segments, records_subset, submodality):
        """
        For each (subj,lab) in records_subset:
         - take each segment (between bkps[i-1], bkps[i])
         - compute mean feature vector for that segment
        Return all_means: (total_segments, D)
        """
        all_means = []
        for (subj, lab) in records_subset:
            seg_key = (subj, lab, submodality)
            if seg_key not in segments:
                continue
            bkps = segments[seg_key]

            # map submodality -> modality name in norm_data
            if submodality.startswith('oculomotor'):
                modality = 'oculomotor'
            else:
                modality = submodality

            data_key = (subj, lab, modality)
            if data_key not in norm_data:
                continue

            df_mod = norm_data[data_key]
            cols = self._features_for_submodality(submodality, df_mod)
            if len(cols) == 0:
                continue

            X = df_mod[cols].to_numpy()
            for i in range(1, len(bkps)):
                start = bkps[i - 1]
                end = bkps[i]
                seg = X[start:end, :]
                seg_mean = np.mean(seg, axis=0)
                all_means.append(seg_mean)

        return np.array(all_means)

    def fit(self, norm_data_train, segments_train, train_records):
        """
        Fit KernelPCA + KMeans (+ Ward reordering) separately for each
        submodality, using TRAIN data only.
        """
        self.models = {}
        submodalities = [
            'oculomotorFixation', 'oculomotorSaccade',
            'scanpath', 'AoI'
        ]

        for sm in submodalities:
            K = self._n_centers(sm)
            all_means = self._collect_segment_means(
                norm_data_train, segments_train, train_records, sm
            )
            if all_means.size == 0:
                continue

            kpca = KernelPCA(n_components=10, kernel='rbf', n_jobs=-1)
            Z = kpca.fit_transform(all_means)

            kmeans = KMeans(n_clusters=K, n_init=100, random_state=0)
            kmeans.fit(Z)
            centers = kmeans.cluster_centers_  # (K, latent_dim)

            # reorder clusters via Ward linkage for stable labels A,B,C,...
            dist_mat = cdist(centers, centers)
            _, res_order, _ = compute_serial_matrix(dist_mat, 'ward')

            inv_res_order = np.zeros(len(res_order), dtype=int)
            for new_idx, old_idx in enumerate(res_order):
                inv_res_order[old_idx] = new_idx

            centers_reordered = np.zeros_like(centers)
            for new_idx, old_idx in enumerate(res_order):
                centers_reordered[new_idx] = centers[old_idx]

            self.models[sm] = {
                'kpca': kpca,
                'kmeans': kmeans,
                'inv_reorder': inv_res_order,
                'centers': centers_reordered,
            }

    def transform(self, norm_data, segments, records_subset):
        """
        Apply TRAIN-fitted (kpca+kmeans+relabel) to any set of records
        (TRAIN or TEST). Output discrete sequences + segment lengths.
        """
        symb_out = {}
        for sm, model in self.models.items():
            kpca = model['kpca']
            kmeans = model['kmeans']
            inv_reorder = model['inv_reorder']
            centers_reordered = model['centers']

            recordings_dict = {}
            for (subj, lab) in records_subset:
                seg_key = (subj, lab, sm)
                if seg_key not in segments:
                    continue
                bkps = segments[seg_key]

                modality = 'oculomotor' if sm.startswith('oculomotor') else sm
                data_key = (subj, lab, modality)
                if data_key not in norm_data:
                    continue

                df_mod = norm_data[data_key]
                cols = self._features_for_submodality(sm, df_mod)
                if len(cols) == 0:
                    continue

                X = df_mod[cols].to_numpy()
                seg_means, seg_lens = [], []

                for i in range(1, len(bkps)):
                    start = bkps[i - 1]
                    end = bkps[i]
                    seg_arr = X[start:end, :]
                    seg_mean = np.mean(seg_arr, axis=0)
                    seg_means.append(seg_mean)
                    seg_lens.append(end - start)

                if len(seg_means) == 0:
                    continue

                Z = kpca.transform(np.array(seg_means))
                raw_labels = kmeans.predict(Z)  # cluster IDs [0..K-1]
                remapped = [int(inv_reorder[l]) for l in raw_labels]

                rec_name = f"{subj}_{lab}"
                recordings_dict[rec_name] = {
                    'sequence': remapped,
                    'lengths': seg_lens
                }

            symb_out[sm] = {
                'centers': centers_reordered,
                'recordings': recordings_dict
            }

        return symb_out


# ---------------------------------------------------------------------
# 5. Build per-modality distance matrices
# ---------------------------------------------------------------------

def build_distance_matrices_per_modality(symb_results, records, config, binning=True):
    """
    For each submodality, build an NxN distance matrix using
    the weighted Levenshtein between symbolic sequences.
    """
    dist_dict = {}
    N = len(records)

    del_cost = config['clustering']['edit_distance']['deletion_cost']
    ins_cost = config['clustering']['edit_distance']['insertion_cost']

    for sm, pack in symb_results.items():
        centers = pack['centers']         # (K, latent_dim)
        recs    = pack['recordings']      # dict rec_name -> {sequence, lengths}

        # compute normalized centroid distance matrix for substitutions
        d_m = cdist(centers, centers)
        if np.max(d_m) > 0:
            d_m = d_m / np.max(d_m)

        # build expanded seq for each record
        seq_map = {}
        for rec in records:
            if rec in recs:
                labs = recs[rec]['sequence']
                lens = recs[rec]['lengths']
                if binning:
                    expanded = []
                    for lab_id, seg_len in zip(labs, lens):
                        expanded.extend([chr(lab_id + 65)] * seg_len)
                else:
                    expanded = [chr(lab_id + 65) for lab_id in labs]
                seq_map[rec] = expanded
            else:
                seq_map[rec] = []

        # fill pairwise distance matrix
        D = np.zeros((N, N), dtype=float)
        for j in range(1, N):
            for i in range(j):
                s1 = seq_map[records[i]]
                s2 = seq_map[records[j]]
                dij = generalized_edit_distance(
                    s1, s2, d_m,
                    deletion_cost=del_cost,
                    insertion_cost=ins_cost
                )
                D[i, j] = D[j, i] = dij

        dist_dict[sm] = D

    return dist_dict


def fuse_modalities_simple(dist_dict):
    """
    Simple late fusion = unweighted sum of all available modality
    distance matrices (oculomotorFixation, oculomotorSaccade, scanpath, AoI).
    """
    fused_all = None
    for _, D in dist_dict.items():
        if fused_all is None:
            fused_all = D.copy()
        else:
            fused_all += D
    return fused_all


# ---------------------------------------------------------------------
# 6. Single-state CV execution (parallelized over states)
# ---------------------------------------------------------------------

def run_one_state_cv(
    state,
    config,
    raw_data,
    all_records,
    y_all,
    conditions,
    conditions_dict
):
    """
    Run full KFold CV for a single random seed/state.
    Returns metrics aggregated over folds.
    """

    # force KFold mode only (LOSO removed)
    n_splits  = config['clustering'].get('n_splits', 10)
    svc_C     = config['clustering'].get('svc_C', 2.0)
    k_mds_def = config['clustering'].get('mds_components', None)
    binning   = config['symbolization'].get('binning', True)

    splitter = KFold(n_splits=n_splits, random_state=state, shuffle=True)
    cv_iter = splitter.split(all_records)

    fold_accuracies = []
    fold_f1s = []
    conf_mat_accum = np.zeros((len(conditions), len(conditions)))

    for train_index, test_index in cv_iter:
        train_records = [all_records[i] for i in train_index]
        test_records  = [all_records[i] for i in test_index]

        y_train = y_all[train_index]
        y_test  = y_all[test_index]

        # convert "1234_5" -> (subject, label)
        train_pairs = [(r.split('_')[0], r.split('_')[1]) for r in train_records]
        test_pairs  = [(r.split('_')[0], r.split('_')[1]) for r in test_records]

        # 1. ECDF normalization
        ecdf_norm = ECDFNormalizer(config)
        ecdf_norm.fit(raw_data, train_pairs)
        norm_train = ecdf_norm.transform(raw_data, train_pairs)
        norm_test  = ecdf_norm.transform(raw_data, test_pairs)

        # 2. segmentation
        segm = Segmenter()
        segments_train = segm.segment_records(norm_train)
        segments_test  = segm.segment_records(norm_test)

        # 3. symbolization
        symb = Symbolizer(config)
        symb.fit(norm_train, segments_train, train_pairs)
        symb_train = symb.transform(norm_train, segments_train, train_pairs)
        symb_test  = symb.transform(norm_test, segments_test, test_pairs)

        # 4. distances per modality on union(train,test)
        union_records = train_records + test_records

        # merge symb info from train + test
        merged_symb = {}
        for sm in symb_train.keys():
            merged_symb[sm] = {
                'centers': symb_train[sm]['centers'],
                'recordings': {}
            }
            merged_symb[sm]['recordings'].update(symb_train[sm]['recordings'])
            if sm in symb_test:
                merged_symb[sm]['recordings'].update(symb_test[sm]['recordings'])

        dist_all = build_distance_matrices_per_modality(
            merged_symb,
            union_records,
            config,
            binning=binning
        )

        fused_all = fuse_modalities_simple(dist_all)

        # 5. slice TRAIN×TRAIN and TEST×TRAIN blocks
        Ntr = len(train_records)
        Nte = len(test_records)
        idx_train = np.arange(Ntr)
        idx_test  = np.arange(Ntr, Ntr + Nte)

        D_tr = fused_all[np.ix_(idx_train, idx_train)]
        D_te_tr = fused_all[np.ix_(idx_test, idx_train)]

        # 6. classical MDS fit on train only, Nyström on test
        D_tr_norm, scale = _normalize_distance_train(D_tr)
        D_te_tr_norm = _apply_scale_to_test_block(D_te_tr, scale)

        k_mds = k_mds_def if k_mds_def is not None else max(2, Ntr // 2)
        X_tr, cache = classical_mds_fit(D_tr_norm, k=k_mds)
        X_te = classical_mds_transform(D_te_tr_norm, cache)

        # 7. final SVM
        clf = SVC(C=svc_C, kernel='rbf', probability=True)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc_fold = np.mean(y_pred == y_test)
        f1_fold = f1_score(y_test, y_pred, average='macro')
        fold_accuracies.append(acc_fold)
        fold_f1s.append(f1_fold)

        # update confusion matrix aggregate
        for ii in range(len(y_pred)):
            conf_mat_accum[
                conditions_dict[y_test[ii]],
                conditions_dict[y_pred[ii]]
            ] += 1

    # aggregate for this state
    result = {
        "state": state,
        "acc_mean": float(np.mean(fold_accuracies)),
        "acc_std": float(np.std(fold_accuracies)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "confmat": conf_mat_accum,
    }
    return result


# ---------------------------------------------------------------------
# 7. High-level driver: parallelize over states
# ---------------------------------------------------------------------

def run_crossval_pipeline(config, path, task):
    """
    Orchestrates:
      - load data
      - build label arrays
      - run multiple random seeds in parallel (KFold only)
      - pick best state
      - plot confusion matrix of the best state
    """

    # 1. Load raw data (oculomotor / scanpath / AoI only)
    raw_data = load_raw_features(config, path)

    # 2. All usable recordings (subject_label strings)
    all_records = sorted(list({
        f"{subj}_{lab}"
        for (subj, lab, _) in raw_data.keys()
        if lab in ['1','2','3','4','5','6','7','8','9']
    }))

    # 3. Map numeric label → workload class
    if task == 'binary':
        dict_task = {
            '1': 'low_wl', '2': 'low_wl', '3': 'low_wl',
            '4': 'low_wl', '5': 'high_wl', '6': 'high_wl',
            '7': 'high_wl', '8': 'high_wl', '9': 'high_wl'
        }
        conditions = ['low_wl', 'high_wl']
    elif task == 'ternary':
        dict_task = {
            '1': 'low_wl', '2': 'low_wl', '3': 'low_wl',
            '4': 'medium_wl', '5': 'medium_wl', '6': 'medium_wl',
            '7': 'high_wl', '8': 'high_wl', '9': 'high_wl'
        }
        conditions = ['low_wl', 'medium_wl', 'high_wl']
 
    y_all = np.array([dict_task[r.split('_')[1]] for r in all_records])

    # 4. Confusion-matrix helpers
    conditions_dict = {c: i for i, c in enumerate(conditions)}

    # 5. We'll run several random seeds (states) in parallel.
    #    No LOSO anymore, so we always do this.
    n_states = config['clustering'].get('n_states', 15)

    print(f"Running KFold CV for {task} task with n_states = {n_states}")

    # 6. Launch all states in parallel
    state_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_one_state_cv)(
            state,
            config,
            raw_data,
            all_records,
            y_all,
            conditions,
            conditions_dict
        )
        for state in range(n_states)
    )

    # 7. Pick best state by mean accuracy
    best = max(state_results, key=lambda r: r["acc_mean"])

    print(f"Final best accuracy={best['acc_mean']:.4f} (state={best['state']})")
    print("STD acc:", best["acc_std"])
    print("F1 score:", best["f1_mean"])
    print("STD F1:", best["f1_std"])
    print("Confusion matrix:")
    print(best["confmat"])

    # 8. Plot confusion matrix for best state
    disp = ConfusionMatrixDisplay(
        best["confmat"].astype(int),
        display_labels=conditions
    )
    disp.plot(values_format='', colorbar=False, cmap='Blues')
    plt.tight_layout()
    plt.show()

 
