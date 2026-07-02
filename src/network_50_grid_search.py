#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50-Network Grid Search for Ricci Flow Analysis

This script trains 50 unique DNN architectures on Fashion-MNIST (Sandals vs Boots)
and computes Ricci-flow analysis for each network.

Based on:
- training.py: Model training and activation extraction
- knn_fixed.py: Ricci curvature analysis

Output:
- NETWORK_GRID_SUMMARY.csv: Summary of all 50 networks
- Per-network folders with mfr.csv, msc.csv, per_layer_correlations.csv
"""

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, triu as sp_triu
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple
import time

from network_50_config import get_all_networks, format_structure

# =============================================================================
# CONFIGURATION
# =============================================================================
K = 350  # kNN neighbors
ACC_THRESHOLD = 0.90  # Minimum accuracy for analysis
OUTPUT_DIR = "output_50_networks"

# =============================================================================
# kNN GRAPH FUNCTIONS (from knn_fixed.py)
# =============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Return an undirected, unweighted kNN adjacency in CSR."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def sum_shortest_paths(A: csr_matrix) -> float:
    """Compute g = sum of all-pairs shortest-path distances, i<j."""
    dist = shortest_path(A, directed=False, unweighted=True)
    iu = np.triu_indices_from(dist, k=1)
    vals = dist[iu]
    finite = np.isfinite(vals)
    if not np.all(finite):
        vals = vals[finite]
    return float(vals.sum())


def global_forman_ricci(A: csr_matrix) -> float:
    """Global Ricci coefficient Ric_l = sum over edges of Forman-Ricci curvatures."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


# =============================================================================
# ANALYSIS FUNCTIONS (from knn_fixed.py)
# =============================================================================

def analyze_model_layers(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """For one model: build graphs, compute (g_l, Ric_l) for l=0..L."""
    A0 = build_knn_graph(X0, k)
    g0 = sum_shortest_paths(A0)
    Ric0 = global_forman_ricci(A0)

    g_list = [g0]
    ric_list = [Ric0]

    for l, Xl in enumerate(activations, start=1):
        A = build_knn_graph(np.asarray(Xl), k)
        g_list.append(sum_shortest_paths(A))
        ric_list.append(global_forman_ricci(A))

    return {"g": np.array(g_list, dtype=float), "Ric": np.array(ric_list, dtype=float)}


def compute_correlations(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute mfr, msc dataframes and correlation stats for a single model.
    """
    res = analyze_model_layers(activations, X0, k)
    g = res["g"]
    Ric = res["Ric"]
    L = len(activations)
    
    # Δg_l for l=1..L
    dgs = g[1:] - g[:-1]
    
    rows_fr = []
    rows_sc = []
    
    for l in range(1, L+1):
        rows_sc.append({"layer": l, "mod": 0, "ssr": float(dgs[l-1])})
        rows_fr.append({"layer": l-1, "mod": 0, "ssr": float(Ric[l-1])})
    
    msc = pd.DataFrame(rows_sc, columns=["layer", "mod", "ssr"])
    mfr = pd.DataFrame(rows_fr, columns=["layer", "mod", "ssr"])
    
    # Compute correlations
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=["mod", "layer"], how="inner", suffixes=("_dg", "_fr"))
    
    if len(merged) >= 2:
        r_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    else:
        r_all = (np.nan, np.nan)
    
    merged_skip = merged[merged["layer"] != 1]
    if len(merged_skip) >= 2:
        r_skip = pearsonr(merged_skip["ssr_dg"].values, merged_skip["ssr_fr"].values)
    else:
        r_skip = (np.nan, np.nan)
    
    stats = {
        "r_all": float(r_all[0]) if not np.isnan(r_all[0]) else None,
        "p_all": float(r_all[1]) if not np.isnan(r_all[1]) else None,
        "r_skip": float(r_skip[0]) if not np.isnan(r_skip[0]) else None,
        "p_skip": float(r_skip[1]) if not np.isnan(r_skip[1]) else None,
    }
    
    return mfr, msc, stats


def compute_per_layer_correlations(mfr: pd.DataFrame, msc: pd.DataFrame) -> pd.DataFrame:
    """Compute per-layer correlations."""
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    
    layers = sorted(merged['layer'].unique())
    rows = []
    
    for layer in layers:
        layer_data = merged[merged['layer'] == layer]
        n = len(layer_data)
        if n >= 2:
            r, p = pearsonr(layer_data['ssr_dg'], layer_data['ssr_fr'])
            rows.append({'layer': layer, 'correlation': r, 'p_value': p, 'n_samples': n})
        else:
            rows.append({'layer': layer, 'correlation': None, 'p_value': None, 'n_samples': n})
    
    return pd.DataFrame(rows)


def plot_summary(msc: pd.DataFrame, mfr: pd.DataFrame, out_png: str) -> None:
    """Generate summary plots."""
    fig = plt.figure(figsize=(8, 8))

    ax1 = plt.subplot(2, 2, 1)
    msc_grouped = msc.groupby('layer')['ssr'].apply(list)
    ax1.boxplot([msc_grouped[l] for l in sorted(msc_grouped.index)], labels=sorted(msc_grouped.index))
    ax1.set_xlabel('Layer l')
    ax1.set_ylabel('Δg_l = g_l - g_{l-1}')
    ax1.set_title('Geodesic Change per Layer')

    ax2 = plt.subplot(2, 2, 2)
    mfr_grouped = mfr.groupby('layer')['ssr'].apply(list)
    ax2.boxplot([mfr_grouped[l] for l in sorted(mfr_grouped.index)], labels=sorted(mfr_grouped.index))
    ax2.set_xlabel('Layer index (for Ric_{l-1})')
    ax2.set_ylabel('Global Forman-Ricci (Ric_{l-1})')
    ax2.set_title('Ricci Curvature per Layer')

    ax3 = plt.subplot(2, 2, 3)
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    sc = ax3.scatter(merged['ssr_dg'], merged['ssr_fr'], c=merged['layer'], marker='o')
    ax3.set_xlabel('Δg_l')
    ax3.set_ylabel('Ric_{l-1}')
    ax3.set_title('Correlation (Δg vs Ric)')
    plt.colorbar(sc, ax=ax3, label='Layer')

    if len(merged) >= 2:
        xs = merged['ssr_dg'].values
        ys = merged['ssr_fr'].values
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        xsu = np.unique(xs)
        ax3.plot(xsu, p(xsu), 'r--', label='Linear fit')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MODEL BUILDING & TRAINING
# =============================================================================

def build_model(layer_structure: List[int], input_dim: int) -> Sequential:
    """Build Keras Sequential model from layer structure."""
    model = Sequential()
    
    # First hidden layer with input shape
    model.add(Dense(units=layer_structure[0], activation='relu', input_shape=(input_dim,)))
    
    # Remaining hidden layers
    for neurons in layer_structure[1:]:
        model.add(Dense(units=neurons, activation='relu'))
    
    # Output layer (binary classification)
    model.add(Dense(units=1, activation='sigmoid'))
    
    return model


def train_model(model: Sequential, x_train: np.ndarray, y_train: np.ndarray, 
                x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, List[np.ndarray]]:
    """
    Train model and extract activations.
    
    Returns:
        accuracy: Test accuracy
        activations: List of layer activations for test data
    """
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    
    # Train
    model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    acc = model.evaluate(x_test, y_test, verbose=0)[1]
    
    # Extract activations
    activations = []
    current_input = x_test
    for layer in model.layers[:-1]:  # Exclude output layer
        current_output = layer(current_input)
        activations.append(current_output.numpy())
        current_input = current_output
    
    return acc, activations


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fmnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST data (Sandals vs Boots)."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'our_data_fmnist')
    
    print(f"Loading data from: {data_path}")
    
    x_test = pd.read_csv(os.path.join(data_path, "fashion-mnist_test.csv"))
    y_test = x_test['label']
    x_test = x_test.iloc[:, 1:]

    x_train = pd.read_csv(os.path.join(data_path, "fashion-mnist_train.csv"))
    y_train = x_train['label']
    x_train = x_train.iloc[:, 1:]

    # Restrict to labels 5 (Sandals) and 9 (Ankle Boots)
    labels = [5, 9]
    train_idx = np.concatenate([np.where(y_train == label)[0] for label in labels])
    test_idx = np.concatenate([np.where(y_test == label)[0] for label in labels])

    y_train = y_train.iloc[train_idx].values
    y_test = y_test.iloc[test_idx].values

    # Convert to binary (0 and 1)
    y_test[y_test == 5] = 0
    y_test[y_test == 9] = 1
    y_train[y_train == 5] = 0
    y_train[y_train == 9] = 1

    x_train = np.array(x_train.iloc[train_idx, :])
    x_test = np.array(x_test.iloc[test_idx, :])
    
    return x_train, y_train, x_test, y_test


# =============================================================================
# MAIN GRID SEARCH
# =============================================================================

def run_single_network(network: Dict, x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray, output_base: str) -> Dict:
    """
    Train and analyze a single network.
    
    Returns dict with results for summary CSV.
    """
    network_id = network['id']
    architecture = network['architecture']
    depth = network['depth']
    width = network['width']
    layer_structure = network['layer_structure']
    
    # Create output folder
    folder_name = f"{architecture}_{depth}_{width}"
    output_dir = os.path.join(output_base, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build and train model
    model = build_model(layer_structure, x_train.shape[1])
    accuracy, activations = train_model(model, x_train, y_train, x_test, y_test)
    
    # Run Ricci analysis
    mfr, msc, stats = compute_correlations(activations, x_test, K)
    
    # Compute per-layer correlations
    per_layer = compute_per_layer_correlations(mfr, msc)
    
    # Save outputs
    mfr.to_csv(os.path.join(output_dir, "mfr.csv"), index=False)
    msc.to_csv(os.path.join(output_dir, "msc.csv"), index=False)
    per_layer.to_csv(os.path.join(output_dir, "per_layer_correlations.csv"), index=False)
    plot_summary(msc, mfr, os.path.join(output_dir, "analysis_plot.png"))
    
    return {
        "network_id": network_id,
        "architecture": architecture,
        "depth": depth,
        "width": width,
        "layer_structure": format_structure(layer_structure),
        "r_all": stats['r_all'],
        "p_all": stats['p_all'],
        "r_skip": stats['r_skip'],
        "p_skip": stats['p_skip'],
        "accuracy": accuracy
    }


def main(test_mode: bool = False):
    """Run the full 50-network grid search."""
    print("=" * 80)
    print("50-NETWORK GRID SEARCH FOR RICCI FLOW ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/3] Loading Fashion-MNIST data (Sandals vs Boots)...")
    x_train, y_train, x_test, y_test = load_fmnist_data()
    print(f"  Training samples: {x_train.shape[0]}")
    print(f"  Test samples: {x_test.shape[0]}")
    
    # Get network configurations
    networks = get_all_networks()
    
    # Test mode: only run first of each architecture type
    if test_mode:
        print("\n[TEST MODE] Running only 5 networks (1 per architecture type)")
        networks = [networks[0], networks[10], networks[20], networks[30], networks[40]]
    
    print(f"\n[2/3] Running grid search over {len(networks)} networks...")
    print(f"  K = {K}")
    print("=" * 80)
    
    # Create output directory
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR)
    os.makedirs(output_base, exist_ok=True)
    
    # Run grid search
    results = []
    
    for network in tqdm(networks, desc="Networks"):
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Network #{network['id']}: {network['architecture']} | L={network['depth']} | W={network['width']}")
        tqdm.write(f"Structure: {format_structure(network['layer_structure'])}")
        
        start_time = time.time()
        result = run_single_network(network, x_train, y_train, x_test, y_test, output_base)
        elapsed = time.time() - start_time
        
        results.append(result)
        
        tqdm.write(f"  ✓ Accuracy: {result['accuracy']:.4f}")
        tqdm.write(f"  ✓ r_all: {result['r_all']:.4f}, r_skip: {result['r_skip']:.4f}")
        tqdm.write(f"  ✓ Time: {elapsed:.1f}s")
    
    # Save summary CSV
    print("\n[3/3] Saving summary results...")
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(output_base, "NETWORK_GRID_SUMMARY.csv")
    results_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE!")
    print("=" * 80)
    print(f"\nSummary saved to: {summary_path}")
    print(f"Output folder: {output_base}")
    
    # Display summary
    print("\nResults Summary:")
    print(results_df[['network_id', 'architecture', 'depth', 'width', 'accuracy', 'r_all', 'r_skip']].to_string(index=False))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='50-Network Grid Search for Ricci Flow Analysis')
    parser.add_argument('--test-mode', action='store_true', 
                        help='Run in test mode (only 5 networks)')
    args = parser.parse_args()
    
    main(test_mode=args.test_mode)
