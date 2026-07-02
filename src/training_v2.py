#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script v2 - DNN Training with Proper Activation Saving
================================================================
This script trains DNN models on Fashion-MNIST (Sandals vs Boots)
and saves activations in a format compatible with knn_fixed_2_1.py

Output per network folder:
    - activations.npy: Object array containing layer activations
    - accuracy.npy: Model test accuracy
"""

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from typing import List, Tuple
import argparse



def load_fmnist_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST data (Sandals vs Boots - labels 5 vs 9).
    
    If CSV files not found, automatically downloads using Keras.
    """
    csv_test = os.path.join(data_path, "fashion-mnist_test.csv")
    csv_train = os.path.join(data_path, "fashion-mnist_train.csv")
    
    # Check if CSV files exist, if not use Keras to download
    if not os.path.exists(csv_test) or not os.path.exists(csv_train):
        print("  CSV files not found, downloading via Keras...")
        from tensorflow.keras.datasets import fashion_mnist
        (x_train_full, y_train_full), (x_test_full, y_test_full) = fashion_mnist.load_data()
        
        # Flatten images (28x28 -> 784)
        x_train_full = x_train_full.reshape(-1, 784).astype(np.float32)
        x_test_full = x_test_full.reshape(-1, 784).astype(np.float32)
        
        # Filter to labels 5 (Sandals) and 9 (Ankle Boots)
        labels = [5, 9]
        train_idx = np.isin(y_train_full, labels)
        test_idx = np.isin(y_test_full, labels)
        
        x_train = x_train_full[train_idx]
        y_train = y_train_full[train_idx]
        x_test = x_test_full[test_idx]
        y_test = y_test_full[test_idx]
        
        # Convert to binary (5->0, 9->1)
        y_train = (y_train == 9).astype(np.int32)
        y_test = (y_test == 9).astype(np.int32)
        
        return x_train, y_train, x_test, y_test
    
    # Load from CSV files
    x_test = pd.read_csv(csv_test)
    y_test = x_test['label']
    x_test = x_test.iloc[:, 1:]

    x_train = pd.read_csv(csv_train)
    y_train = x_train['label']
    x_train = x_train.iloc[:, 1:]

    # Restrict to labels 5 (Sandals) and 9 (Ankle Boots)
    labels = [5, 9]
    train_idx = np.concatenate([np.where(y_train == label)[0] for label in labels])
    test_idx = np.concatenate([np.where(y_test == label)[0] for label in labels])

    y_train = y_train.iloc[train_idx].values
    y_test = y_test.iloc[test_idx].values

    # Convert to binary
    y_test[y_test == 5] = 0
    y_test[y_test == 9] = 1
    y_train[y_train == 5] = 0
    y_train[y_train == 9] = 1

    x_train = np.array(x_train.iloc[train_idx, :])
    x_test = np.array(x_test.iloc[test_idx, :])
    
    return x_train, y_train, x_test, y_test



def build_flat_model(depth: int, width: int, input_dim: int) -> Sequential:
    """Build flat architecture: all hidden layers have same width."""
    model = Sequential()
    model.add(Dense(units=width, activation='relu', input_shape=(input_dim,)))
    for _ in range(depth - 1):
        model.add(Dense(units=width, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def build_bottleneck_model(depth: int, width: int, input_dim: int) -> Sequential:
    """
    Build bottleneck architecture: 
    Expands from width, compresses to 32 in the middle, expands back.
    Example for depth=5, width=128: 128 -> 64 -> 32 -> 64 -> 128
    """
    model = Sequential()
    
    # Calculate layer sizes for symmetric bottleneck
    half_depth = depth // 2
    
    # Compression ratios (from width to 32)
    layers_sizes = []
    
    if depth <= 4:
        # Simpler bottleneck for shallow networks
        layers_sizes = [width, 32, 32, width][:depth]
    else:
        # Build symmetric bottleneck
        compression_layers = []
        current_size = width
        for i in range(half_depth):
            compression_layers.append(current_size)
            if current_size > 32:
                current_size = max(32, current_size // 2)
        
        # Add middle layer(s)
        middle_count = depth - 2 * len(compression_layers)
        middle_layers = [32] * max(1, middle_count)
        
        # Expansion is mirror of compression
        expansion_layers = compression_layers[::-1]
        
        layers_sizes = compression_layers + middle_layers + expansion_layers
        layers_sizes = layers_sizes[:depth]  # Trim to exact depth
    
    # Build model
    model.add(Dense(units=layers_sizes[0], activation='relu', input_shape=(input_dim,)))
    for size in layers_sizes[1:]:
        model.add(Dense(units=size, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    return model


def train_and_save(
    architecture: str,
    depth: int,
    width: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[float, str]:
    """
    Train a single model and save outputs.
    
    Returns:
        (accuracy, output_path)
    """
    input_dim = x_train.shape[1]
    
    # Build model
    if architecture == 'flat':
        model = build_flat_model(depth, width, input_dim)
    elif architecture == 'bottleneck':
        model = build_bottleneck_model(depth, width, input_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    
    # Train
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    
    # Extract activations from hidden layers (exclude output layer)
    activations = []
    current_input = x_test
    for layer in model.layers[:-1]:
        current_output = layer(current_input)
        activations.append(current_output.numpy())
        current_input = current_output
    
    # Create output folder
    folder_name = f"{architecture}_{depth}_{width}"
    network_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(network_output_dir, exist_ok=True)
    
    # Save outputs - use object array for varying shapes (bottleneck networks)
    activations_arr = np.empty(len(activations), dtype=object)
    for i, act in enumerate(activations):
        activations_arr[i] = act
    np.save(os.path.join(network_output_dir, "activations.npy"), activations_arr)
    np.save(os.path.join(network_output_dir, "accuracy.npy"), np.array([accuracy]))
    
    return accuracy, network_output_dir


def main():
    parser = argparse.ArgumentParser(description='Train DNN and save activations')
    parser.add_argument('--architecture', type=str, default='flat', choices=['flat', 'bottleneck'])
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='output_k_sweep')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--test', action='store_true', help='Quick test with 5 epochs')
    parser.add_argument('--force', action='store_true', help='Force retrain even if already exists')
    args = parser.parse_args()
    
    # Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'our_data_fmnist')
    output_dir = os.path.join(base_path, args.output_dir)
    
    # Check if already trained (checkpoint)
    folder_name = f"{args.architecture}_{args.depth}_{args.width}"
    network_output_dir = os.path.join(output_dir, folder_name)
    checkpoint_file = os.path.join(network_output_dir, "activations.npy")
    
    if os.path.exists(checkpoint_file) and not args.force:
        print(f"[SKIP] {folder_name} already trained (use --force to retrain)")
        return
    
    print("=" * 60)
    print("TRAINING DNN MODEL")
    print("=" * 60)
    print(f"Architecture: {args.architecture}")
    print(f"Depth: {args.depth}")
    print(f"Width: {args.width}")
    print(f"Output: {output_dir}")
    
    # Load data
    print("\n[1/3] Loading Fashion-MNIST data...")
    x_train, y_train, x_test, y_test = load_fmnist_data(data_path)
    print(f"  Training: {x_train.shape}, Test: {x_test.shape}")
    
    # Train
    epochs = 5 if args.test else args.epochs
    print(f"\n[2/3] Training model ({epochs} epochs)...")
    
    accuracy, output_path = train_and_save(
        architecture=args.architecture,
        depth=args.depth,
        width=args.width,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
        epochs=epochs
    )
    
    print(f"\n[3/3] Done!")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Saved to: {output_path}")
    
    # Verify saved files
    acts = np.load(os.path.join(output_path, "activations.npy"), allow_pickle=True)
    print(f"  Activations: {len(acts)} layers")
    for i, a in enumerate(acts):
        print(f"    Layer {i}: {a.shape}")


if __name__ == "__main__":
    main()
