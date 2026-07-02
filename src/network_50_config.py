#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Configuration for 50-Network Grid Search

This file defines all 50 unique DNN architectures for the Ricci-flow analysis experiment.

Architecture Types:
- Flat-16 (16 neurons): Networks 1-10 (layers 3-12)
- Flat-32 (32 neurons): Networks 11-20 (layers 3-12)
- Flat-64 (64 neurons): Networks 21-30 (layers 3-12)
- Flat-128 (128 neurons): Networks 31-40 (layers 3-12)
- Bottleneck-128 (128→64→32 hourglass): Networks 41-50 (layers 3-12)

Total: 50 networks
"""

from typing import List, Dict


def generate_flat_structure(num_layers: int, width: int) -> List[int]:
    """Generate flat (constant width) layer structure."""
    return [width] * num_layers


def generate_bottleneck_structure(num_layers: int, outer: int = 128, mid: int = 64, center: int = 32) -> List[int]:
    """
    Generate bottleneck (hourglass) layer structure.
    
    Pattern (for 128→64→32):
    - 3 layers: 128 → 32 → 128
    - 4 layers: 128 → 32 → 32 → 128
    - 5 layers: 128 → 64 → 32 → 64 → 128
    - 6 layers: 128 → 64 → 32 → 32 → 64 → 128
    - 7 layers: 128 → 128 → 64 → 32 → 64 → 128 → 128
    - etc.
    """
    if num_layers < 3:
        raise ValueError("Bottleneck requires at least 3 layers")
    
    is_even = (num_layers % 2 == 0)
    
    # Center layers
    if is_even:
        center_count = 2
    else:
        center_count = 1
    
    remaining = num_layers - center_count
    half = remaining // 2
    
    # Build left half (from outer to center)
    # We need: outer → mid → center
    # For short networks (3-4 layers), skip mid layer
    if half == 0:
        left_half = []
    elif half == 1:
        left_half = [outer]  # Just outer, then center
    elif half == 2:
        left_half = [outer, mid]  # outer → mid → center
    else:
        # More layers: fill with outer, then mid
        outer_count = half - 1
        left_half = [outer] * outer_count + [mid]
    
    # Build the full structure: left + center + right (mirror of left)
    structure = left_half + [center] * center_count + left_half[::-1]
    
    return structure


def get_all_networks() -> List[Dict]:
    """
    Generate all 50 network configurations.
    
    Returns:
        List of dicts with keys:
        - id: int (1-50)
        - architecture: str (flat/bottleneck)
        - depth: int (number of hidden layers)
        - width: int (main width parameter)
        - layer_structure: List[int] (neurons per layer)
    """
    networks = []
    network_id = 1
    
    # ==========================================================================
    # FLAT-16 (16 neurons) - Networks 1-10
    # ==========================================================================
    for num_layers in range(3, 13):  # 3 to 12 inclusive
        networks.append({
            "id": network_id,
            "architecture": "flat",
            "depth": num_layers,
            "width": 16,
            "layer_structure": generate_flat_structure(num_layers, 16)
        })
        network_id += 1
    
    # ==========================================================================
    # FLAT-32 (32 neurons) - Networks 11-20
    # ==========================================================================
    for num_layers in range(3, 13):  # 3 to 12 inclusive
        networks.append({
            "id": network_id,
            "architecture": "flat",
            "depth": num_layers,
            "width": 32,
            "layer_structure": generate_flat_structure(num_layers, 32)
        })
        network_id += 1
    
    # ==========================================================================
    # FLAT-64 (64 neurons) - Networks 21-30
    # ==========================================================================
    for num_layers in range(3, 13):  # 3 to 12 inclusive
        networks.append({
            "id": network_id,
            "architecture": "flat",
            "depth": num_layers,
            "width": 64,
            "layer_structure": generate_flat_structure(num_layers, 64)
        })
        network_id += 1
    
    # ==========================================================================
    # FLAT-128 (128 neurons) - Networks 31-40
    # ==========================================================================
    for num_layers in range(3, 13):  # 3 to 12 inclusive
        networks.append({
            "id": network_id,
            "architecture": "flat",
            "depth": num_layers,
            "width": 128,
            "layer_structure": generate_flat_structure(num_layers, 128)
        })
        network_id += 1
    
    # ==========================================================================
    # BOTTLENECK-128 (128→64→32) - Networks 41-50
    # ==========================================================================
    for num_layers in range(3, 13):  # 3 to 12 inclusive
        networks.append({
            "id": network_id,
            "architecture": "bottleneck",
            "depth": num_layers,
            "width": 128,
            "layer_structure": generate_bottleneck_structure(num_layers, outer=128, mid=64, center=32)
        })
        network_id += 1
    
    return networks


def format_structure(layer_structure: List[int]) -> str:
    """Format layer structure as arrow-separated string."""
    return "→".join(map(str, layer_structure))


# =============================================================================
# For testing / display
# =============================================================================
if __name__ == "__main__":
    networks = get_all_networks()
    
    print("=" * 80)
    print("50-NETWORK GRID SEARCH CONFIGURATION")
    print("=" * 80)
    print(f"\nTotal networks: {len(networks)}")
    
    print("\n" + "-" * 80)
    print("FLAT-16 (16 neurons) - Networks 1-10")
    print("-" * 80)
    for net in networks[:10]:
        print(f"#{net['id']:2d} | L={net['depth']:2d} | {format_structure(net['layer_structure'])}")
    
    print("\n" + "-" * 80)
    print("FLAT-32 (32 neurons) - Networks 11-20")
    print("-" * 80)
    for net in networks[10:20]:
        print(f"#{net['id']:2d} | L={net['depth']:2d} | {format_structure(net['layer_structure'])}")
    
    print("\n" + "-" * 80)
    print("FLAT-64 (64 neurons) - Networks 21-30")
    print("-" * 80)
    for net in networks[20:30]:
        print(f"#{net['id']:2d} | L={net['depth']:2d} | {format_structure(net['layer_structure'])}")
    
    print("\n" + "-" * 80)
    print("FLAT-128 (128 neurons) - Networks 31-40")
    print("-" * 80)
    for net in networks[30:40]:
        print(f"#{net['id']:2d} | L={net['depth']:2d} | {format_structure(net['layer_structure'])}")
    
    print("\n" + "-" * 80)
    print("BOTTLENECK-128 (128→64→32) - Networks 41-50")
    print("-" * 80)
    for net in networks[40:50]:
        print(f"#{net['id']:2d} | L={net['depth']:2d} | {format_structure(net['layer_structure'])}")
    
    print("\n" + "=" * 80)
