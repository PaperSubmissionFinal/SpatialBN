# Spatial-VCBN: Spatially-Heterogeneous Causal Bayesian Networks for Seismic Multi-Hazard Estimation

This repository contains the implementation of Spatial-VCBN, a novel approach for seismic multi-hazard and impact estimation using spatially-varying causal parameters modeled through a combination of Gaussian Processes and normalizing flows.

## Overview

Spatial-VCBN addresses a significant limitation in existing disaster impact assessment methods by explicitly modeling how causal relationships between hazards (e.g., landslides, liquefaction) and impacts (e.g., building damage) vary geographically. This framework:

- Captures spatial heterogeneity in causal relationships through Gaussian Processes
- Models complex, non-Gaussian distributions of causal effects using normalizing flows
- Decouples multiple hazard signals from satellite-derived Damage Proxy Maps (DPMs)
- Provides robust performance even in signal-constrained environments

## Contents

This repository provides a complete, self-contained Jupyter notebook that implements the entire Spatial-VCBN pipeline. The notebook is structured to be run sequentially from top to bottom, with clear section headers guiding you through each step.

## Pipeline Workflow

1. **Data Loading**
   - Loading and preprocessing of satellite-derived Damage Proxy Maps (DPMs)
   - Loading geospatial features (DEM, slope, Vs30, CTI, land cover, coast distance)
   - Loading USGS ground failure models as priors for landslides and liquefaction

2. **Data Preprocessing**
   - Normalization and cleaning of input data
   - Conversion of areal percentages to probabilities for hazard priors
   - Pruning and classification of local models

3. **Model Configuration**
   - Setting up the Config class with hyperparameters
   - Defining the normalizing flow architecture
   - Setting up Gaussian Process priors with Matern kernels

4. **Spatial-VCBN Model Definition**
   - Implementation of the SpatialModel class that uses GP with normalizing flows
   - Initialization of inducing points for sparse GP approximation
   - Implementation of forward pass with spatial pruning

5. **Training and Inference**
   - Stochastic variational inference with expectation-maximization
   - Mini-batch processing for efficient computation
   - GPU-accelerated implementation for handling large geographical regions

6. **Evaluation and Visualization**
   - Calculation of evaluation metrics (AUC, F1 score)
   - Generation of spatial distributions of causal parameters
   - Visualization of posterior hazard probabilities

## Case Example: Puerto Rico Earthquake

The notebook contains a complete pipeline for processing the 2020 Puerto Rico earthquake data:

```python
# Example of running the pipeline for Puerto Rico data
MyDataLoader, test_PLS, test_PLF, test_PBD, IND = prepare_data(
    config, PLS, PLF, PBD, GeoFeatures, DPM, BF, LOCAL
)

# Initialize model
model = SpatialModel(config, total_num, sigma, GeoFeatures, PLS=PLS, PLF=PLF, PBD=PBD)

# Move model to appropriate device
if config.cuda:
    print("Model on GPU")
    model = model.to(config.device)

# Set optimizer
opt = optim.Adam(model.parameters(), lr=config.learning_rate)

# Run training and evaluation
test_results, global_params = run(MyDataLoader, model, opt, test_PLS, test_PLF, test_PBD, IND, config)

# Visualization
# Code for generating figures showing spatial distributions of hazards and causal parameters
```

## Requirements

The code requires the following packages:
- PyTorch
- torchinfo
- rasterio
- scipy
- scikit-learn
- matplotlib
- normflows

Install dependencies with:
```bash
pip install torchinfo rasterio scipy scikit-learn matplotlib normflows
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/PaperSubmissionFinal/SpatialBN.git
cd SpatialBN
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook spatial_vcbn.ipynb
```

3. Follow the notebook sections sequentially to execute the entire pipeline.

## Data Preparation

The notebook is set up to work with disaster data in the following format:
- Damage Proxy Maps (DPMs) from ARIA team (TIFF format)
- USGS ShakeMap and ground failure models as priors
- Geospatial features (DEM, Vs30, CTI, etc.) in TIFF format
- Building footprints (optional)

You can modify the data loading section to point to your specific data files.

## Model Configuration

Key hyperparameters can be adjusted in the Config class:
- `batch_size`: Mini-batch size for stochastic optimization
- `num_flows`: Number of normalizing flow layers (default is 6)
- `num_inducing`: Number of inducing points for sparse GP
- `learning_rate`: Learning rate for optimizer
- `epochs`: Number of training epochs
- `hidden_dim`: Dimension of hidden layers in neural networks
- `flow_type`: Type of normalizing flow (default is "planar")

## Computational Efficiency

The implementation includes several optimizations for handling large geographical regions:
- Sparse GP approximation with inducing points
- Local pruning strategy for focusing computational resources
- GPU acceleration (when available)
- Low-rank plus diagonal structure for covariance matrices

On a system with NVIDIA Tesla T4 GPU.



## Contact

For questions or feedback, please contact xli359@jhu.edu
