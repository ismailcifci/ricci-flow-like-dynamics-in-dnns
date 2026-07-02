#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci-Flow K-Sweep Analysis (knn_fixed_2_1.py)
==============================================
This script performs Ricci flow analysis with k-sweep over multiple networks.

Based on the paper "Deep Learning as Ricci Flow" (arXiv:2404.14265v1)

Key features:
- Skips input layer (uses Hidden Layer 1 as baseline X0)
- K-sweep with multiple k values: [250, 350, 450, 500, 600]
- Outputs per-k and aggregated results

Expected input structure per network folder:
    activations.npy: Object array (shape: (L,)) where each entry is (n_samples, layer_dim)
    accuracy.npy: Single-element array with model accuracy

Output:
    Per network folder:
        - mfr_k{k}.csv, msc_k{k}.csv for each k
        - correlations_k_sweep.csv: Summary of all k values
    
    Root output folder:
        - K_SWEEP_MASTER_SUMMARY.csv: All networks x all k values
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
K_VALUES = [250, 350, 450, 500, 600, 750]  # K-sweep values

# =============================================================================
# GRAPH & METRIC FUNCTIONS (Matching paper Eq. 3, 4, 6, 7)
# =============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """
    Build undirected, unweighted k-NN graph.
    
    Per paper Section 2.2:
    - Symmetrize kNN graph using max (mutualization)
    - Zero diagonal (no self-loops)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)  # Symmetrization
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def compute_geodesic_mass(A: csr_matrix) -> float:
    """
    Compute geodesic mass g_l (Eq. 7 in paper):
    g_l = sum_{i<j} gamma_l(i,j)
    
    where gamma_l(i,j) is the shortest path distance in the k-NN graph.
    """
    dist = shortest_path(A, directed=False, unweighted=True)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    finite = np.isfinite(vals)
    if not np.all(finite):
        vals = vals[finite]
    return float(vals.sum())


def compute_forman_ricci(A: csr_matrix) -> float:
    """
    Compute global Forman-Ricci curvature Ric_l (Eq. 4, 6 in paper):
    
    For edge (i,j): R(i,j) = 4 - deg(i) - deg(j)  [unit weights]
    Ric_l = sum_{(i,j) in E_l} R(i,j)
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()  # Upper triangle for undirected edges
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_layers(activations: List[np.ndarray], k: int) -> Dict[str, np.ndarray]:
    """
    Analyze layer activations for a single model.
    
    IMPORTANT: We skip the input layer (as per knn_fixed_2.py).
    - X0 (baseline) = activations[0] (first hidden layer)
    - Sequence = activations[1:] (remaining hidden layers)
    
    Returns:
        g: array of geodesic masses (length = L)
        Ric: array of Ricci curvatures (length = L)
    
    where L = number of hidden layers
    """
    L = len(activations)
    g_list = []
    ric_list = []
    
    for l in range(L):
        A = build_knn_graph(activations[l], k)
        g_list.append(compute_geodesic_mass(A))
        ric_list.append(compute_forman_ricci(A))
    
    return {
        "g": np.array(g_list, dtype=float),
        "Ric": np.array(ric_list, dtype=float)
    }


def compute_correlation(activations: List[np.ndarray], k: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute Δg_l vs Ric_{l-1} correlation (matching paper Eq. 5).
    
    The paper states: g_{l+1} - g_l ≈ -α·Ric(g_l)
    So we correlate: Δg_l = g_l - g_{l-1} with Ric_{l-1}
    
    Since we skip input layer:
    - Layer 1 = first hidden layer (baseline)
    - Layer 2 = second hidden layer
    - etc.
    
    Returns:
        mfr: DataFrame with Ric values per layer
        msc: DataFrame with Δg values per layer  
        stats: dict with 'r' (Pearson correlation) and 'p' (p-value)
    """
    res = analyze_layers(activations, k)
    g = res["g"]
    Ric = res["Ric"]
    L = len(activations)
    
    # Δg_l = g_l - g_{l-1} for l = 2, 3, ..., L
    dgs = g[1:] - g[:-1]  # length L-1
    
    # Build DataFrames
    # msc: Δg values (layer 2 onwards)
    # mfr: Ric values (layer 1 onwards, will be paired with Δg of next layer)
    rows_msc = []
    rows_mfr = []
    
    for l in range(1, L):  # l = 1 to L-1
        # Δg between layer l+1 and layer l
        rows_msc.append({"layer": l + 1, "mod": 0, "ssr": float(dgs[l - 1])})
        # Ric at layer l (to correlate with Δg at layer l+1)
        rows_mfr.append({"layer": l, "mod": 0, "ssr": float(Ric[l - 1])})
    
    msc = pd.DataFrame(rows_msc, columns=["layer", "mod", "ssr"])
    mfr = pd.DataFrame(rows_mfr, columns=["layer", "mod", "ssr"])
    
    # Compute Pearson correlation
    # Align: mfr[layer=l] with msc[layer=l+1]
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    
    if len(merged) >= 2:
        r, p = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    else:
        r, p = np.nan, np.nan
    
    stats = {"r": float(r) if not np.isnan(r) else None, "p": float(p) if not np.isnan(p) else None}
    
    return mfr, msc, stats


# =============================================================================
# NETWORK FOLDER PROCESSING
# =============================================================================

def parse_network_folder(folder_name: str) -> Optional[Dict]:
    """
    Parse network folder name to extract architecture info.
    
    Format: {architecture}_{depth}_{width}
    Example: flat_5_64 -> {'architecture': 'flat', 'depth': 5, 'width': 64}
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None
    
    try:
        architecture = parts[0]
        depth = int(parts[1])
        width = int(parts[2])
        return {'architecture': architecture, 'depth': depth, 'width': width}
    except (ValueError, IndexError):
        return None


def process_single_network(network_path: str, k_values: List[int], output_path: str) -> List[Dict]:
    """
    Process a single network folder with k-sweep.
    
    Returns list of result dicts (one per k value).
    """
    folder_name = os.path.basename(network_path)
    info = parse_network_folder(folder_name)
    
    if info is None:
        print(f"  [SKIP] Cannot parse folder name: {folder_name}")
        return []
    
    # Load activations
    activations_path = os.path.join(network_path, "activations.npy")
    accuracy_path = os.path.join(network_path, "accuracy.npy")
    
    if not os.path.exists(activations_path):
        print(f"  [SKIP] No activations.npy: {folder_name}")
        return []
    
    activations = np.load(activations_path, allow_pickle=True)
    activations = list(activations)  # Convert to list
    
    if os.path.exists(accuracy_path):
        accuracy = float(np.load(accuracy_path)[0])
    else:
        accuracy = None
    
    # Create output folder
    os.makedirs(output_path, exist_ok=True)
    
    # K-sweep analysis
    results = []
    k_sweep_rows = []
    
    for k in k_values:
        try:
            mfr, msc, stats = compute_correlation(activations, k)
            
            # Save per-k outputs
            mfr.to_csv(os.path.join(output_path, f"mfr_k{k}.csv"), index=False)
            msc.to_csv(os.path.join(output_path, f"msc_k{k}.csv"), index=False)
            
            # Record result
            result = {
                "network": folder_name,
                "architecture": info['architecture'],
                "depth": info['depth'],
                "width": info['width'],
                "k": k,
                "r": stats['r'],
                "p": stats['p'],
                "accuracy": accuracy
            }
            results.append(result)
            k_sweep_rows.append({"k": k, "r": stats['r'], "p": stats['p']})
            
        except Exception as e:
            print(f"  [ERROR] k={k}: {e}")
            results.append({
                "network": folder_name,
                "architecture": info['architecture'],
                "depth": info['depth'],
                "width": info['width'],
                "k": k,
                "r": None,
                "p": None,
                "accuracy": accuracy,
                "error": str(e)
            })
    
    # Save k-sweep summary for this network
    if k_sweep_rows:
        pd.DataFrame(k_sweep_rows).to_csv(
            os.path.join(output_path, "correlations_k_sweep.csv"), index=False
        )
    
    return results


def run_k_sweep_analysis(input_dir: str, output_dir: str, k_values: List[int] = K_VALUES) -> pd.DataFrame:
    """
    Run k-sweep analysis over all network folders.
    
    Args:
        input_dir: Directory containing network folders with activations
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("K-SWEEP RICCI FLOW ANALYSIS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"K values: {k_values}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all network folders
    network_folders = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path) and not name.startswith('.'):
            network_folders.append((name, path))
    
    print(f"Found {len(network_folders)} network folders")
    print("-" * 70)
    
    all_results = []
    
    for folder_name, folder_path in tqdm(network_folders, desc="Processing networks"):
        output_path = os.path.join(output_dir, folder_name)
        results = process_single_network(folder_path, k_values, output_path)
        all_results.extend(results)
    
    # Save master summary
    results_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, "K_SWEEP_MASTER_SUMMARY.csv")
    results_df.to_csv(summary_path, index=False)
    
    # Generate BEST_K_SUMMARY.csv - find best k (most negative r) for each network
    best_k_rows = []
    if not results_df.empty and 'r' in results_df.columns:
        for network in results_df['network'].unique():
            net_data = results_df[results_df['network'] == network].copy()
            # Filter to valid r values only
            net_data = net_data[net_data['r'].notna()]
            if not net_data.empty:
                # Best k = most negative r (strongest Ricci flow-like behavior)
                best_idx = net_data['r'].idxmin()
                best_row = net_data.loc[best_idx]
                best_k_rows.append({
                    "network": best_row['network'],
                    "architecture": best_row['architecture'],
                    "depth": best_row['depth'],
                    "width": best_row['width'],
                    "best_k": best_row['k'],
                    "best_r": best_row['r'],
                    "best_p": best_row['p'],
                    "accuracy": best_row['accuracy']
                })
    
    best_k_df = pd.DataFrame(best_k_rows)
    best_k_path = os.path.join(output_dir, "BEST_K_SUMMARY.csv")
    best_k_df.to_csv(best_k_path, index=False)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Processed: {len(network_folders)} networks")
    print(f"Total results: {len(all_results)} (networks × k values)")
    print(f"Master summary: {summary_path}")
    print(f"Best K summary: {best_k_path}")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Sweep Ricci Flow Analysis')
    parser.add_argument('--input-dir', type=str, default='output_k_sweep',
                        help='Input directory containing network folders')
    parser.add_argument('--output-dir', type=str, default='output_k_sweep_analysis',
                        help='Output directory for results')
    parser.add_argument('--k-values', type=int, nargs='+', default=K_VALUES,
                        help='K values for sweep')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: single k value only')
    args = parser.parse_args()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_path, args.input_dir)
    output_dir = os.path.join(base_path, args.output_dir)
    
    k_values = [350] if args.test else args.k_values
    
    results = run_k_sweep_analysis(input_dir, output_dir, k_values)
    
    # Print summary statistics
    if not results.empty and 'r' in results.columns:
        print("\nSummary by K value:")
        for k in k_values:
            k_data = results[results['k'] == k]
            valid_r = k_data['r'].dropna()
            if len(valid_r) > 0:
                print(f"  k={k}: mean(r)={valid_r.mean():.4f}, median(r)={valid_r.median():.4f}, n={len(valid_r)}")


if __name__ == "__main__":
    main()
