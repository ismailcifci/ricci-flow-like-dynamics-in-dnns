# Ricci Flow-Like Dynamics in DNNs

This repository contains the code, notebooks, outputs, and final report for a project studying whether deep neural networks transform data geometry in a way that resembles discrete Ricci flow. The experiments represent layer activations as k-nearest-neighbor graphs, compute global and local Ricci-style curvature quantities, and compare those geometric signals with classification accuracy, architecture depth, early stopping behavior, and long-term predictive performance.

The project is built mainly on two reference lines: Baptista et al.'s global Ricci-flow formulation for deep learning, and Weber-related local Ricci feature-geometry work for layer-wise/local curvature measurements.

The full report is in [`papers/FinallyFinal.pdf`](papers/FinallyFinal.pdf).

## Project Overview

The project is built around three related questions:

- Do DNN layer transformations show Ricci flow-like geometric behavior?
- Can global or local Ricci coefficients explain differences between architectures, datasets, or depths?
- Can early Ricci signals help with early stopping or architecture screening?

The experiments cover synthetic datasets, MNIST, Fashion-MNIST, and CIFAR-10 binary classification tasks. The main curvature tools are Baptista-style global Ricci coefficients, Weber-style local/layer-wise Ricci coefficients, Augmented Forman-Ricci curvature, and comparisons with layer-wise activation geometry.

## Repository Layout

```text
data/
  cifar10/                  Local data location for CIFAR-10 artifacts
  fmnist/                   Local data location for Fashion-MNIST artifacts
  synthetic/                Synthetic dataset assets

experiments/
  early_stopping/           Ricci-based early stopping experiments and summaries
  layer_ricci_weber/        Layer Ricci coefficient experiment based on Weber-style local curvature
  predictive/fmnist/        Fashion-MNIST predictive Ricci experiment
  predictive/cifar10/       CIFAR-10 local/global predictive Ricci experiments and results
  two_phase_screening/      Two-phase Ricci-based architecture screening notebooks

src/
  training_v2.py            Trains DNNs and saves layer activations
  knn_fixed_2_1.py          Earlier global Ricci coefficient implementation
  knn_fixed_3.py            Global Ricci coefficient and k-sweep implementation
  network_50_config.py      Architecture grid definitions
  network_50_grid_search.py Network-grid training and global Ricci analysis
  results/                  Main saved global Ricci and network-grid outputs

papers/
  FinallyFinal.pdf          Final project report
  DeepLearninAsRicci.pdf    Main reference paper
  LocalRicci.pdf            Local Ricci reference material

reference_code/
  anthbapt-global/          Reference global Ricci implementation
  weber-local/              Reference local Ricci implementation

visuals/                    Figures and dataset visualizations used for reporting
```

## Main Experiments

- **Global Ricci analysis:** follows the Baptista et al. global formulation. It trains networks, saves activations, builds k-NN graphs at each layer, computes graph geodesic mass and global Forman-Ricci curvature, then correlates their layer-to-layer changes.
- **Network-grid analysis:** compares many flat and bottleneck architectures over several k values and stores per-network Ricci/accuracy summaries under `src/results/network_grid/`.
- **Layer/local Ricci analysis:** follows the Weber-related local feature-geometry formulation. It computes local layer-wise curvature behavior to study how geometry changes across hidden layers.
- **Ricci early stopping:** compares curvature stabilization against standard validation-based early stopping.
- **Predictive Ricci experiments:** tests whether early Ricci behavior predicts long-term accuracy on Fashion-MNIST and CIFAR-10.
- **Local vs global CIFAR-10 comparison:** compares early local Ricci, raw global Ricci, and Fisher-normalized global Ricci against long-term CIFAR-10 accuracy.

## References

The main methodological references are:

- **Global Ricci reference:** Baptista, A., Barp, A., Chakraborti, T., Harbron, C., MacArthur, B. D., and Banerji, C. R. S. (2024). *Deep Learning as Ricci Flow*. This is the basis for the global Ricci coefficient, graph geodesic mass, Forman-Ricci curvature aggregation, and Fisher-normalized global coefficient used in `src/knn_fixed_3.py` and the global CIFAR experiments.
- **Local Ricci reference:** Weber-related local Ricci feature geometry work, represented here by `papers/LocalRicci.pdf` and the `reference_code/weber-local/` implementation. This is the basis for local/layer-wise Ricci coefficients, neighborhood-level curvature behavior, and the layer Ricci experiments.

Reference implementations are kept under `reference_code/`: `anthbapt-global/` for the Baptista-style global workflow and `weber-local/` for the Weber-style local workflow.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, use:

```bash
source .venv/bin/activate
```

PyTorch installation can depend on your CUDA/CPU setup. If the generic install from `requirements.txt` is not suitable, install the correct `torch` and `torchvision` wheels from the official PyTorch selector, then install the remaining requirements.

## Running Code

The notebooks in `experiments/` are the easiest entry points for reproducing specific experiments. Most of the current project outputs are already saved in the corresponding experiment folders or under `src/results/`.

Examples:

```bash
python src/training_v2.py
python src/knn_fixed_3.py --help
python src/network_50_grid_search.py
```

Some notebooks were originally run in Google Colab and may contain Colab-specific mount cells. When running locally, update paths so outputs stay inside this repository.

## Results

Important saved outputs include:

- `src/results/global_ricci_depth_analysis/`: global Ricci depth, k, and dataset-level analysis.
- `src/results/network_grid/`: 50-network grid outputs and k-sweep summaries.
- `experiments/predictive/fmnist/ricci_predictive_results.csv`: Fashion-MNIST predictive Ricci summary.
- `experiments/predictive/cifar10/results/`: CIFAR-10 local/global Ricci predictive results, correlations, and plots.
- `experiments/early_stopping/`: Ricci early stopping summaries.
- `experiments/two_phase_screening/two_phase_results.csv`: two-phase architecture screening results.

## Notes

- The project mixes TensorFlow/Keras scripts and PyTorch notebooks because different experiment families were developed separately.
- `reference_code/` is kept for comparison with external implementations, not as the primary project source.
- Large experiments can be slow because k-NN graph construction, shortest-path computation, and curvature calculation are repeated across layers, epochs, architectures, and k values.
