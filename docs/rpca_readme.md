# Robust PCA (RPCA) README

## Overview

This file contains a pure Python implementation of Robust Principal Component Analysis (RPCA) based on the R package `rpca` and the work of Candès et al. (2011). The implementation decomposes a data matrix into a low-rank component and a sparse component:

M=L+S.

The `RobustPCA` class performs the mathematical RPCA decomposition, while the `rpca` class is a higher-level wrapper designed for energy-consumption data and visualization.

## Requirements

The implementation uses:

```bash
pip install numpy pandas matplotlib seaborn
```

The code imports:
- `numpy`
     `pandas`
     `matplotlib`
     `seaborn`
     `logging`

## Importing the Classes

If the Python file is named `rpca.py`, import the classes with:

```python
from rpca import RobustPCA, rpca
```

## The RobustPCA Class

The `RobustPCA` class performs Principal Component Pursuit using an Augmented Lagrange Multiplier (ALM) / Alternating Directions algorithm.

The optimization problem is:

minimize ||L||_* + lambda ||S||_1

subject to

L + S = M.

Here:
- $M$ is the original data matrix.
     $L$ is the low-rank component representing the underlying structure.
     $S$ is the sparse component representing unusual or anomalous deviations.
     $lambda$ controls the sparsity penalty.

## Constructor

Use:

```python
RobustPCA(M, lambda_param=None, mu=None, tol=1e-7, max_iter=5000)
```

| Parameter | Description |
|---|---|
| `M` | Input numerical matrix to decompose. |
| `lambda_param` | Regularization parameter. If omitted, the implementation uses 1/sqrt(max(n1, n2)). |
| `mu` | ALM penalty parameter. If omitted, it is calculated from the dimensions and matrix norm. |
| `tol` | Convergence tolerance. Default is `1e-7`. |
| `max_iter` | Maximum number of iterations. Default is 5000. |

## Basic Usage

Create a numerical matrix and fit the RPCA model:

```python
import numpy as np
from rpca import RobustPCA

M = np.array([
    [1, 2, 3],
    [1, 2, 30],
    [1, 2, 3]
], dtype=float)

model = RobustPCA(M)
L, S = model.fit()
```

After fitting:

```python
print(L)
print(S)
```

The decomposition satisfies approximately:

```python
M = L + S
```

The low-rank matrix `L` represents the regular structure, while the sparse matrix `S` identifies deviations that are concentrated in relatively few observations.

## Understanding `fit()`

The `fit()` method initializes the sparse and dual matrices and iteratively performs three main operations.

### 1. Singular Value Thresholding

The low-rank matrix is updated using singular value thresholding:

L(k+1) = D(1/mu)
(M - S(k) + (1/mu)Y(k)).

This operation shrinks singular values toward zero, encouraging a low-rank solution.

### 2. Element-wise Shrinkage

The sparse component is updated using soft-thresholding:

S(k+1) = S(lambda/mu)
(M - L(k+1) + (1/mu)Y(k)).

This encourages most entries of $S$ to be zero while retaining large deviations.

### 3. Dual Variable Update

The dual matrix is updated using:

Y(k+1) = Y(k) + mu(M - L(k+1) - S(k+1)).

The algorithm stops when the relative Frobenius-norm residual is below the requested tolerance.

## Checking Convergence

The model stores convergence information in:

```python
model.convergence_info
```

It contains:
- `converged`: whether the algorithm reached the tolerance.
     `iterations`: number of iterations performed.
     `final_delta`: final relative residual.
     `all_delta`: convergence values for every iteration.

For example:

```python
print(model.convergence_info["converged"])
print(model.convergence_info["iterations"])
print(model.convergence_info["final_delta"])
```

## Low-Rank and Sparse Components

The fitted matrices are also stored as attributes:

```python
model.L
model.S
model.L_svd
```

`L` is the low-rank representation of the data, and `S` contains the sparse deviations.

For energy data, the sparse matrix can be interpreted as a collection of unusual deviations from the regular energy-consumption structure.

## The `rpca Wrapper Class`

The second class, named `rpca`, is designed to apply RPCA directly to a pandas DataFrame and produce energy-related visualizations.

Its constructor is:

```python
rpca(dataframe, total, meter=None,
     overage=None, timeframe=None, value=None)
```

| Parameter | Description |
|---|---|
| `dataframe` | DataFrame containing the energy data. |
| `total` | DataFrame containing total/cumulative energy values used by the overage plot. |
| `meter` | Name of the building or meter column to display in the actual-vs-low-rank plot. |
| `overage` | Optional collection of dates to mark on the actual-vs-low-rank plot. |
| `timeframe` | Optional two-element range specifying the beginning and ending dates to display. |
| `value` | Threshold used by the campus overage plot. |

When the wrapper is created, it immediately performs RPCA on `dataframe.values` and converts the resulting $L$ and $S$ matrices back into pandas DataFrames.

## Wrapper Example

A basic example is:

```python
model = rpca(
    dataframe=energy_df,
    total=total_df,
    meter="Building_A",
    overage=overage_dates,
    timeframe=["2025-01-01", "2025-03-31"],
    value=22000
)
```

The resulting matrices are available through:

```python
model.lmatrix
model.smatrix
```

## `heatmap()`

The `heatmap()` method visualizes the sparse RPCA component.

```python
model.heatmap()
```

The sparse matrix is transposed so that:
- rows represent buildings/meters;
     columns represent timestamps;
     cell values represent deviations.

Buildings are ordered according to the total absolute magnitude of their sparse deviations.

Large positive or negative values indicate stronger deviations from the low-rank structure.

## `stackplot()`

Use:

```python
model.stackplot()
```

This compares the actual energy usage for the selected meter against its RPCA low-rank component.

If `timeframe` is provided, only that date range is displayed.

If `overage` is provided, the specified dates are marked on the plot.

## `overages()`

The `overages()` method displays the cumulative campus energy values against the supplied threshold.

```python
model.overages()
```

The method:
1. converts the index to datetime;
     optionally restricts the plot to `timeframe`;
     plots the `Cummulative` column;
     draws the supplied threshold;
     identifies observations above the threshold;
     marks those observations on the graph.

## `all1()`

The `all1()` method combines the main visualizations into one analysis:

```python
model.all1()
```

It produces:
1. an RPCA sparse-deviation heatmap;
     an actual-versus-low-rank plot for the selected meter;
     a campus cumulative overage plot.

A timeframe can also be supplied:

```python
model.all1(tf=["2025-01-01", "2025-03-31"])
```

## Recommended Workflow for Energy Data

For an energy-consumption dataset, a typical workflow is:

```python
from rpca import rpca

model = rpca(
    dataframe=energy_df,
    total=total_df,
    meter="Building_A",
    overage=overage_dates,
    timeframe=["2025-01-01", "2025-03-31"],
    value=22000
)

L = model.lmatrix
S = model.smatrix

model.heatmap()
model.stackplot()
model.overages()
model.all1()
```

## Interpreting the Results

The main interpretation is:

M = L + S

- **Original matrix ($M$):** observed energy measurements.
     **Low-rank matrix ($L$):** regular/common energy-consumption structure.
     **Sparse matrix ($S$):** unusual deviations or energy events.

For the energy dashboard, the sparse matrix is especially useful for identifying timestamps and buildings where consumption differs substantially from the expected low-rank pattern.

## Troubleshooting

### The algorithm does not converge

Check:

```python
model.convergence_info
```

If necessary, experiment with `tol`, `max_iter`, or the regularization parameters.

### The input contains non-numeric values

The `RobustPCA` class converts the input to a floating-point NumPy array. The input matrix should therefore contain numeric values.

### The heatmap is difficult to read

The heatmap may contain many timestamps and buildings. Consider restricting the analysis to a smaller timeframe before plotting.

## Summary

The implementation provides two levels of RPCA analysis:

1. **`RobustPCA`** performs the mathematical decomposition of a matrix into low-rank and sparse components.
     **`rpca`** applies that decomposition to pandas energy data and provides heatmap, comparison, and overage visualizations.

The central result is:

M=L+S,

where $L$ captures the underlying low-rank structure and $S$ captures sparse deviations that can be investigated as potential unusual energy-consumption events.
