# Data cleaning

`DataCleaning` inspects meter time series for **missing values** and **zeros**, optionally **imputes** gaps, and detects or replaces **outliers** using several statistical or model-based methods.

## Role in the pipeline

Run cleaning **after** `DataProcessor` (or any source that yields an hourly indexed `DataFrame`) and **before** EDA or modeling on the cleaned series.

```mermaid
flowchart LR
  raw[Raw meter DataFrame]
  clean[DataCleaning]
  filtered[filtered attribute]
  raw --> clean --> filtered
```

## Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataframe` | — | Input data (datetime index). |
| `meter` | `None` | Column to clean; if `None`, all numeric columns are used. |
| `method` | `None` | Outlier method; must be set before calling `detect_outliers()`. |
| `window` | `12` | Rolling window (hours) for Hampel / Savitzky–Golay. |
| `n_sigma` | `3` | Threshold multiplier (Hampel, polynomial MAD, fencing). |
| `order` | `2` | Polynomial order for Savitzky–Golay and interpolation. |
| `imputem` | `True` | If `True`, polynomial interpolation fills NaNs on init. |
| `iw` | `0.95` | Interval width for Prophet-based anomaly detection. |
| `daily`, `weekly` | `False` | Seasonality flags for NeuralProphet outlier method. |
| `min`, `max` | `5`, `5` | Count of smallest/largest points for `sorted()` / `visual()`. |

On construction, the class sets:

- `missingcounts` — summary of missing indices, or a message if none.
- `zerocounts` — summary of zero-value indices, or a message if none.
- `filtered` — copy of the data updated when a detection method runs.

## Outlier methods

Call **`detect_outliers()`** after setting `method` in the constructor (or re-instantiate with the desired `method`).

| `method` | Approach |
|----------|----------|
| **`Hampel`** | Rolling median-based filter; edge windows use IQR fencing; replaces spikes with local medians. |
| **`Polynomial`** | Savitzky–Golay smooth; flags points whose residual exceeds `n_sigma` × MAD; negative fits replaced with column median. |
| **`Prophet`** | Fits Prophet on `ds`/`y`; points outside prediction intervals are replaced with `yhat`. |
| **`Fencing`** | Classic IQR fences per column; outliers replaced with column median. |
| **`Neural`** | NeuralProphet fit; large absolute residuals flagged via IQR on residuals. |
| **`Visual`** | Manual: mark min/max extremes or explicit timestamps as outliers, then interpolate. |

After detection, inspect:

- `outliers` — detected points.
- `replacements` — before/after values.
- `count` — short text summary of how many outliers were found.
- `filtered` — cleaned series to pass to EDA or `ModelingClass`.

## Example

```python
from timeries import DataCleaning

cleaner = DataCleaning(
    dataframe=proc.dataframe,
    meter="Total",
    method="Hampel",
    window=12,
    n_sigma=3,
)

cleaned = cleaner.detect_outliers()
print(cleaner.count)
print(cleaner.outliers.head())
```

## Choosing a method

- **Hampel** — fast, good default for smooth hourly energy data with short spikes.
- **Polynomial** — smooth baseline + residual threshold; useful when seasonality is gradual.
- **Prophet / Neural** — when outliers are “off the seasonal curve”; heavier runtime.
- **Fencing** — simple, distribution-based; no seasonality model.
- **Visual** — domain knowledge (known bad intervals or extreme ranked points).

## Other helpers

| Method | Purpose |
|--------|---------|
| `sorted(ascending=True)` | Return indices of the `min` smallest and `max` largest values. |
| `visual(ts=None)` | Drop or interpolate outliers at chosen timestamps or auto min/max extremes. |

## API reference

Full listing: [`DataCleaning` in the API reference](api.md#timeries.data_cleaning.DataCleaning).
