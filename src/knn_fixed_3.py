#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knn_fixed_3.py -- Global Ricci coefficient analysis (k-sweep)
================================================================

Implements the global Ricci network flow framework from
    Baptista et al., "Deep Learning as Ricci Flow", arXiv:2404.14265 (2024).

Per-DNN Ricci coefficient (paper Eq. 8):
    rho_k = corr( eta_l , Ric_l ),     l = 1, ..., L-1
    eta_l = g_{l+1} - g_l                                                 (Eq. 5)
    g_l   = sum_{i,j in V} gamma^l_k(i,j)                                 (Eq. 7)
    Ric_l = sum_{(i,j) in E^l_k} ( 4 - deg_l_k(i) - deg_l_k(j) )      (Eqs. 4, 6)

Adjusted (Fisher z) Ricci coefficient (paper Eq. 11):
    z_k   = arctanh(rho_k) / sqrt(L - 4)

Fixes vs the previous version (the issues we discussed):
  + Fisher z-transformation (Eq. 11)              (was missing)
  + Single-connected-component check               (was silently dropping inf)
  + No per-network "best k" cherry picking         (was data-snooping)
  + Per-(network, k) checkpointing                  (resumable on interrupt)
  + Configurable include_input / include_output    (default = hidden only)
  + Configurable kNN symmetrization               (default 'max' = mutual-OR)
  + CIFAR-aligned default k-sweep values

Expected input  (per architecture folder, produced by `training_v2.py`):
    layer_0.npy ... layer_{L+1}.npy   (input, h_1..h_L, sigmoid output)
    accuracy.npy                       (1-element array, final test accuracy)
    epoch_history.json                 (optional, per-epoch metrics)

Backward-compatible: if no `layer_*.npy` files are found, falls back to
loading legacy `activations.npy` (object array).
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm


# =============================================================================
# Default k values
#
# CIFAR (binary, balanced 500+500=1000 test points). Paper Section 3.2 reports
# that for image data (MNIST/fMNIST, |X_test| ~= 2000) the optimal k sat in
# 12.5%-25% of |X_test|.
# =============================================================================

K_VALUES = [125, 185, 250] #CIFAR


# =============================================================================
# Graph & metric primitives  (paper Eqs. 3, 4, 6, 7)
# =============================================================================

def build_knn_graph(X: np.ndarray,
                    k: int,
                    symmetrize: str = 'max') -> csr_matrix:
    """Undirected, unweighted k-NN graph G_k(X).

    symmetrize:
        'max'  -- mutual-OR  : edge if EITHER direction is in raw kNN  (default,
                  matches networkx interpretation used by the reference Python
                  implementation, and the local notebook k-sweep code).
        'min'  -- mutual-AND : edge only if BOTH directions are in raw kNN.
        'none' -- raw asymmetric kNN (NOT recommended: Forman degrees become
                  constant out-degree=k, breaking the curvature variability).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32, copy=False)

    A = kneighbors_graph(
        X,
        n_neighbors=k,
        mode='connectivity',
        metric='euclidean',
        include_self=False,
    )

    if symmetrize == 'max':
        A = A.maximum(A.T)
    elif symmetrize == 'min':
        A = A.minimum(A.T)
    elif symmetrize == 'none':
        pass
    else:
        raise ValueError(f'symmetrize must be max/min/none, got {symmetrize!r}')

    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def is_connected(A: csr_matrix) -> bool:
    """True iff the (undirected) graph has a single connected component."""
    n_components, _ = connected_components(A, directed=False, return_labels=True)
    return n_components == 1


def compute_geodesic_mass(A: csr_matrix) -> Optional[float]:
    """g_l = sum_{i<j} gamma_l(i,j)   (paper Eq. 7).

    Returns None if the graph is not a single connected component (so some
    gamma_l(i,j) = +inf and g_l is undefined per the paper). Networks with
    a None at any layer should be excluded from the Ricci coefficient -- as
    in Section 3.3 of the paper (134/1361 DNNs were excluded for this reason).
    """
    if not is_connected(A):
        return None
    dist = shortest_path(A, directed=False, unweighted=True)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    if not np.all(np.isfinite(vals)):
        return None
    return float(vals.sum())


def compute_forman_ricci(A: csr_matrix) -> float:
    """Ric_l = sum_{(i,j) in E_l} (4 - deg(i) - deg(j))   (paper Eqs. 4, 6)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


def fisher_z(r: Optional[float], L: int) -> Optional[float]:
    """z_k = arctanh(rho_k) / sqrt(L - 4)   (paper Eq. 11).

    L = number of layer representations correlated (paper uses L hidden layers,
    so the correlation has L-1 data points). Returns None if r is undefined,
    |r| >= 1, or L < 5 (so the sqrt is real and positive).
    """
    if r is None or not np.isfinite(r) or abs(r) >= 1.0:
        return None
    if L < 5:
        return None
    return float(math.atanh(r) / math.sqrt(L - 4))


# =============================================================================
# Per-DNN analysis
# =============================================================================

def analyze_layers(activations: List[np.ndarray],
                   k: int,
                   symmetrize: str = 'max') -> Dict[str, np.ndarray]:
    """Compute (g_l, Ric_l, connected_l) for each layer in `activations`.

    g_l is NaN at layers where the kNN graph is not single-component.
    """
    L = len(activations)
    g    = np.full(L, np.nan, dtype=float)
    Ric  = np.zeros(L, dtype=float)
    conn = np.zeros(L, dtype=bool)
    for l in range(L):
        A = build_knn_graph(activations[l], k, symmetrize=symmetrize)
        g_l = compute_geodesic_mass(A)
        if g_l is not None:
            g[l]    = g_l
            conn[l] = True
        Ric[l] = compute_forman_ricci(A)
    return {'g': g, 'Ric': Ric, 'connected': conn}


def compute_ricci_coefficient(activations: List[np.ndarray],
                              k: int,
                              symmetrize: str = 'max') -> Dict:
    """Per-DNN global Ricci coefficient (paper Eqs. 5-8 + 11).

    Pearson correlation of  eta_l = g_{l+1} - g_l  vs  Ric_l  for l = 1..L-1.
    `L` here = number of layer representations passed in; the correlation
    therefore uses L-1 points and Fisher-z uses sqrt(L-4) -- so we need L >= 5
    for `z` to be defined.
    """
    res = analyze_layers(activations, k, symmetrize=symmetrize)
    g, Ric, conn = res['g'], res['Ric'], res['connected']
    L = len(activations)

    if not bool(np.all(conn)):
        return {
            'r': None, 'p': None, 'z': None,
            'L': L, 'connected_all': False,
            'g': g.tolist(), 'Ric': Ric.tolist(), 'dgs': [],
        }

    dgs        = g[1:] - g[:-1]
    Ric_paired = Ric[:-1]

    if len(dgs) >= 2 and np.std(dgs) > 0 and np.std(Ric_paired) > 0:
        r, p = pearsonr(dgs, Ric_paired)
        r, p = float(r), float(p)
    else:
        r, p = float('nan'), float('nan')

    z = fisher_z(r if np.isfinite(r) else None, L)

    return {
        'r': r if np.isfinite(r) else None,
        'p': p if np.isfinite(p) else None,
        'z': z,
        'L': L,
        'connected_all': True,
        'g':   g.tolist(),
        'Ric': Ric.tolist(),
        'dgs': dgs.tolist(),
    }


# =============================================================================
# Network folder I/O
# =============================================================================

def _layer_files(arch_dir: str) -> List[str]:
    """Return sorted layer_*.npy paths from a network directory."""
    if not os.path.isdir(arch_dir):
        return []
    found: List[Tuple[int, str]] = []
    for name in os.listdir(arch_dir):
        if name.startswith('layer_') and name.endswith('.npy'):
            try:
                idx = int(name[len('layer_'):-len('.npy')])
            except ValueError:
                continue
            found.append((idx, os.path.join(arch_dir, name)))
    found.sort()
    return [p for _, p in found]


def _legacy_activations_npy(arch_dir: str) -> Optional[List[np.ndarray]]:
    """Backward-compat: load legacy `activations.npy` (object array)."""
    p = os.path.join(arch_dir, 'activations.npy')
    if os.path.exists(p):
        arr = np.load(p, allow_pickle=True)
        return list(arr)
    return None


def load_activations(arch_dir: str,
                     include_input: bool = False,
                     include_output: bool = False) -> Optional[List[np.ndarray]]:
    """Load saved activations as a Python list of np.ndarrays.

    `training_v2.py` saves [input, h_1..h_L, sigmoid_output]. Use the two
    flags to opt out of the input or sigmoid output layer.
    """
    files = _layer_files(arch_dir)
    if files:
        feats = [np.load(p) for p in files]
    else:
        feats = _legacy_activations_npy(arch_dir)
        if feats is None:
            return None
    if not include_input and len(feats) > 0:
        feats = feats[1:]
    if not include_output and len(feats) > 0:
        feats = feats[:-1]
    return feats


def parse_network_folder(folder_name: str) -> Optional[Dict]:
    """Folder convention: {arch_type}_{depth}_{width}, e.g. 'flat_5_64'."""
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None
    try:
        return {
            'architecture': parts[0],
            'depth':        int(parts[1]),
            'width':        int(parts[2]),
        }
    except (ValueError, IndexError):
        return None


def load_accuracy(arch_dir: str) -> Optional[float]:
    p = os.path.join(arch_dir, 'accuracy.npy')
    if os.path.exists(p):
        a = np.load(p)
        return float(a.flatten()[0])
    return None


# =============================================================================
# K-sweep with per-(network, k) checkpointing
# =============================================================================

def _save_json(path: str, obj) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def run_k_sweep(input_dir: str,
                output_dir: str,
                k_values: List[int] = K_VALUES,
                symmetrize: str = 'max',
                include_input: bool = False,
                include_output: bool = False) -> pd.DataFrame:
    """Compute global Ricci coefficient for every (network, k) pair.

    For each k, writes:
        {output_dir}/global_ricci_k{k}.json   -- master list (one row / network)
        {output_dir}/checkpoint_k{k}.json     -- resumable {'completed': [...]}
        {output_dir}/{network}/global_ricci_k{k}.json  -- per-network detail

    No per-network "best k" selection: paper picks k per architecture from an
    aggregated coefficient. The aggregation step (per-arch averaging across
    seeds) lives in the calling code (e.g. the experiment notebook), not here.
    """
    os.makedirs(output_dir, exist_ok=True)

    arch_dirs = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith('.')
    )
    print(f'Found {len(arch_dirs)} network folders under {input_dir}')
    print(f'k values        : {k_values}')
    print(f'symmetrize      : {symmetrize}')
    print(f'include_input   : {include_input}')
    print(f'include_output  : {include_output}')

    all_rows: List[Dict] = []
    for k in k_values:
        ckpt_path   = os.path.join(output_dir, f'checkpoint_k{k}.json')
        master_path = os.path.join(output_dir, f'global_ricci_k{k}.json')

        ckpt   = _load_json(ckpt_path,   {'completed': []})
        master = _load_json(master_path, [])
        completed = set(ckpt['completed'])

        all_rows.extend(master)

        print(f'\n=== k = {k}  ({len(completed)}/{len(arch_dirs)} already done) ===')
        for name in tqdm(arch_dirs, desc=f'k={k}'):
            if name in completed:
                continue

            arch_dir = os.path.join(input_dir, name)
            info = parse_network_folder(name)
            if info is None:
                completed.add(name)
                _save_json(ckpt_path, {'completed': sorted(completed), 'k': k})
                continue

            feats = load_activations(arch_dir,
                                     include_input=include_input,
                                     include_output=include_output)
            if feats is None or len(feats) < 3:
                completed.add(name)
                _save_json(ckpt_path, {'completed': sorted(completed), 'k': k})
                continue

            try:
                stats = compute_ricci_coefficient(feats, k,
                                                  symmetrize=symmetrize)
                err = None
            except Exception as e:
                stats = {'r': None, 'p': None, 'z': None,
                         'L': len(feats), 'connected_all': False,
                         'g': [], 'Ric': [], 'dgs': []}
                err = str(e)

            row = {
                'network':       name,
                'architecture':  info['architecture'],
                'depth':         info['depth'],
                'width':         info['width'],
                'k':             k,
                'L':             stats.get('L'),
                'r':             stats.get('r'),
                'z':             stats.get('z'),
                'p':             stats.get('p'),
                'connected_all': stats.get('connected_all'),
                'accuracy':      load_accuracy(arch_dir),
            }
            if err is not None:
                row['error'] = err

            master.append(row)
            all_rows.append(row)

            per_net_dir = os.path.join(output_dir, name)
            os.makedirs(per_net_dir, exist_ok=True)
            _save_json(os.path.join(per_net_dir, f'global_ricci_k{k}.json'),
                       {**row, **stats})

            _save_json(master_path, master)
            completed.add(name)
            _save_json(ckpt_path, {'completed': sorted(completed), 'k': k})

    df = pd.DataFrame(all_rows)
    master_csv = os.path.join(output_dir, 'global_ricci_master.csv')
    df.to_csv(master_csv, index=False)
    print(f'\nSaved master CSV: {master_csv}')

    if not df.empty and 'r' in df.columns:
        print('\nSummary per k (only rows with finite r):')
        print(f'  {"k":>4}  {"n":>4}  '
              f'{"mean_r":>8}  {"med_r":>8}  '
              f'{"mean_z":>8}  {"med_z":>8}')
        for k in k_values:
            sub = df[(df['k'] == k) & df['r'].notna()]
            if len(sub):
                print(f'  {k:>4}  {len(sub):>4}  '
                      f'{sub["r"].mean():>+8.4f}  {sub["r"].median():>+8.4f}  '
                      f'{sub["z"].mean():>+8.4f}  {sub["z"].median():>+8.4f}')
    return df


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Global Ricci coefficient analysis with k-sweep.')
    parser.add_argument('--input-dir',  type=str,
                        default='output_global_ricci',
                        help='Folder of trained networks (output of training_v2.py).')
    parser.add_argument('--output-dir', type=str,
                        default='output_global_ricci_analysis',
                        help='Where to write per-k Ricci coefficient files.')
    parser.add_argument('--k-values', type=int, nargs='+',
                        default=K_VALUES,
                        help='List of k values to sweep over.')
    parser.add_argument('--symmetrize', type=str,
                        default='max', choices=['max', 'min', 'none'])
    parser.add_argument('--with-input', action='store_true',
                        help='Include input layer (layer_0) in analysis.')
    parser.add_argument('--with-output', action='store_true',
                        help='Include output layer (sigmoid) in analysis.')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: single k = 100.')
    args = parser.parse_args()

    base_path  = os.path.dirname(os.path.abspath(__file__))
    input_dir  = (args.input_dir  if os.path.isabs(args.input_dir)
                  else os.path.join(base_path, args.input_dir))
    output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                  else os.path.join(base_path, args.output_dir))

    k_values = [100] if args.test else args.k_values

    run_k_sweep(input_dir=input_dir,
                output_dir=output_dir,
                k_values=k_values,
                symmetrize=args.symmetrize,
                include_input=args.with_input,
                include_output=args.with_output)


if __name__ == '__main__':
    main()
