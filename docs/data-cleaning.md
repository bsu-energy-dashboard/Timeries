# DataCleaning Class: README / User Guide

## Overview

The `DataCleaning` class is designed to assist with cleaning time-series data, particularly numeric meter or energy-consumption data. The class provides several approaches for identifying and replacing unusual observations, including:

* Hampel filtering
* Piecewise polynomial detection
* Prophet-based anomaly detection
* IQR/fencing detection
* NeuralProphet-based detection
* Visual/manual detection
* Robust Principal Component Analysis (RPCA)

The class also checks for missing and zero values, can interpolate missing observations, and provides information about detected outliers and their replacements.

---

## Requirements

The Python file imports the following major packages:

```python
numpy
pandas
hampel
matplotlib
scipy
scikit-learn
prophet
neuralprophet
seaborn
```

For example, the required packages can generally be installed with:

```bash
pip install numpy pandas hampel matplotlib scipy scikit-learn prophet neuralprophet seaborn
```

> **Note:** Prophet and NeuralProphet can require additional system or Python dependencies depending on the computing environment.

---

## Importing the Class

Place the Python file containing the class in the same directory as your notebook or Python script. If the file is named `DataCleaning.py`, import the class with:

```python
os.chdir('/Users/amira/Library/CloudStorage/Box-Box/DataCleaning/pyfiles')
%run datacleaning.py
from datacleaning import DataCleaning
```

The file also contains a `RobustPCA` class, which is used internally by `DataCleaning` and is run automatically.

---

## Preparing the Data

The class expects a pandas DataFrame. For time-series applications, the DataFrame should normally have a datetime index and one or more numeric columns.

Example:

```python
import pandas as pd

df = pd.read_excel(
    "meter_data.xlsx",
    sheet_name="Sheet1",
    index_col="Date / Time",
    parse_dates=True
)

print(df.head())
```

A single meter column can then be selected by name when creating the `DataCleaning` object.

---

## Creating a DataCleaning Object

The basic syntax is:

```python
cleaner = DataCleaning(
    dataframe=df,
    meter="Meter_Name",
    method="Hampel"
)
```

The constructor is:

```python
DataCleaning(
    dataframe,
    window=12,
    n_sigma=3,
    meter=None,
    max=5,
    min=5,
    order=2,
    method=None,
    daily=False,
    weekly=False,
    imputem=True,
    iw=0.95,
    value=None
)
```

### Constructor Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `dataframe` | Required | Input pandas DataFrame. The class also uses the DataFrame values when initializing RPCA. |
| `window` | `12` | Window size used by the Hampel and RPCA procedures and by the polynomial procedure. |
| `n_sigma` | `3` | Threshold multiplier used by the Hampel, polynomial, and fencing procedures. |
| `meter` | `None` | Name of the meter/column to analyze. When supplied, the class restricts the working DataFrame to that column. |
| `max` | `5` | Number of largest observations considered by the visual method. |
| `min` | `5` | Number of smallest observations considered by the visual method. |
| `order` | `2` | Polynomial order used by the Savitzky–Golay filter. |
| `method` | `None` | Outlier-detection method. Valid choices in `detect_outliers()` are `Hampel`, `Polynomial`, `Prophet`, `Fencing`, `Neural`, and `Visual`. |
| `daily` | `False` | Passed to the NeuralProphet configuration as the daily-seasonality setting. |
| `weekly` | `False` | Passed to the NeuralProphet configuration as the weekly-seasonality setting. |
| `imputem` | `True` | If enabled and missing values exist, the constructor interpolates missing values using polynomial interpolation of order 2. |
| `iw` | `0.95` | Prophet interval width. |
| `value` | `None` | Threshold used by the RPCA outlier-detection method. |

---

## Detecting Outliers

The recommended general workflow is:

```python
cleaner = DataCleaning(
    dataframe=df,
    meter="Meter_Name",
    method="Hampel"
)

cleaned_df = cleaner.detect_outliers()
```

The `detect_outliers()` method calls the method specified in the `method` argument.

---

## Available Methods

### 1. Hampel Method

Use the Hampel method with:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Hampel",
    window=12,
    n_sigma=3
)

cleaned_df = cleaner.detect_outliers()
```

The method applies the Hampel filter to the observations and additionally handles observations near the beginning and end of the series using an IQR-based fencing procedure. Detected observations are replaced with values from the filtering procedure.

The original outliers are stored in:

```python
cleaner.outliers
```

The before-and-after replacement information is stored in:

```python
cleaner.replacements
```

A text description of the number of detected observations is available through:

```python
print(cleaner.count)
```

### 2. Piecewise Polynomial Method

Use:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Polynomial",
    window=13,
    order=2,
    n_sigma=3
)

cleaned_df = cleaner.detect_outliers()
```

The method uses a Savitzky–Golay filter to produce a local polynomial fit. Residuals are calculated as:

$$\text{residual}_t = y_t - \widehat{y}_t$$

The median absolute deviation (MAD) of the residuals is then used to define an outlier threshold. Observations whose absolute residual exceeds the threshold are identified as outliers.

The method stores the detected outliers in:

```python
cleaner.outliers
```

and the replacement information in:

```python
cleaner.replacements
```

The fitted values and outlier indicators are also available through the internal report:

```python
cleaner.report
```

> **Important:** The polynomial method requires the polynomial order to be smaller than the window size. If an even window is supplied, the class increments it by one.

### 3. Prophet Method

Use:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Prophet",
    iw=0.95
)

cleaned_df = cleaner.detect_outliers()
```

The method fits a Prophet model to the selected time series and identifies observations outside the model's prediction interval.

The yearly and weekly seasonalities can be controlled when calling the method:

```python
cleaned_df = cleaner.prophet(
    yearly=True,
    weekly=True
)
```

Detected observations are stored in:

```python
cleaner.outliers
```

and the replacement values are stored in:

```python
cleaner.replacements
```

### 4. Fencing Method

The fencing method uses an IQR-based rule:

$$\text{Lower Bound} = Q_1 - n_\sigma \cdot \text{IQR}$$

$$\text{Upper Bound} = Q_3 + n_\sigma \cdot \text{IQR}$$

where:

$$\text{IQR} = Q_3 - Q_1$$

Use:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Fencing",
    n_sigma=3
)

cleaned_df = cleaner.detect_outliers()
```

Values outside the calculated bounds are replaced with the median of the corresponding column.

### 5. NeuralProphet Method

Use:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Neural",
    daily=False,
    weekly=True
)

cleaned_df = cleaner.detect_outliers()
```

The method fits a NeuralProphet model using the configured lag and seasonality settings. It calculates absolute residuals and uses an IQR-based threshold to identify observations classified as outliers.

The fitted forecast is available as:

```python
cleaner.forecast
```

The detected observations and replacement information are available through:

```python
cleaner.outliers
cleaner.replacements
```

### 6. Visual Method

The visual method can be used to identify the smallest and largest observations automatically or to specify particular timestamps.

For the smallest and largest observations:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Visual",
    min=5,
    max=5
)

cleaned_df = cleaner.detect_outliers()
```

The selected minimum and maximum observations are available through:

```python
cleaner.min_df
cleaner.max_df
```

Specific timestamps can instead be supplied:

```python
timestamps = [
    "2026-01-15 12:00:00",
    "2026-02-03 14:00:00"
]

cleaned_df = cleaner.visual(ts=timestamps)
```

The selected values are replaced with polynomially interpolated values.

### 7. RPCA Method

The class also includes an RPCA-based procedure. RPCA decomposes a matrix into a low-rank component and a sparse component:

$$M = L + S$$

The `RobustPCA` class performs this decomposition using an augmented Lagrange multiplier / alternating-directions approach.

To detect and clean outliers with RPCA:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    window=13,
    order=2,
    value=0.95
)

cleaned_df = cleaner.rpca_detect()
```

The `value` argument determines the threshold used for the RPCA outlier score. The procedure replaces detected outliers with missing values, polynomially interpolates the resulting series, and then applies a Savitzky–Golay smoothing filter.

The final cleaned series is returned as a DataFrame.

---

## Checking Missing and Zero Values

The constructor checks the working DataFrame for missing and zero values.

Missing-value information is available through:

```python
cleaner.missingcounts
```

Zero-value information is available through:

```python
cleaner.zerocounts
```

If no missing values are found, the missing-value attribute contains a message indicating that there are no missing values. The same behavior is used for zero values.

---

## Missing-Value Imputation

By default, `imputem=True`. If missing values are present, the constructor uses polynomial interpolation of order 2:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    imputem=True
)
```

To disable this behavior:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    imputem=False
)
```

---

## Sorting and Inspecting Extreme Values

The `sorted()` method sorts the selected meter and identifies the smallest and largest observations.

Example:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    min=5,
    max=5
)

smallest, largest = cleaner.sorted()

print(smallest)
print(largest)
```

The resulting DataFrames are also stored as:

```python
cleaner.minval
cleaner.maxval
```

Their corresponding indices are stored as:

```python
cleaner.indexmin
cleaner.indexmax
```

---

## Understanding the Main Outputs

Most outlier-detection methods return a cleaned DataFrame:

```python
cleaned_df = cleaner.detect_outliers()
```

The following object attributes are commonly useful:

| Attribute | Purpose |
| :--- | :--- |
| `filtered` | DataFrame after detected observations have been replaced. |
| `outliers` | Original observations identified as outliers. |
| `replacements` | Information about values before and after replacement. |
| `count` | Text description of the number/type of detected outliers. |
| `report` | Additional model or residual information for methods that create a report. |
| `forecast` | NeuralProphet forecast and residual information. |
| `missingcounts` | Missing-value information. |
| `zerocounts` | Zero-value information. |
| `lmatrix` | Low-rank RPCA component for the selected meter. |
| `smatrix` | Sparse RPCA component for the selected meter. |
| `cleaned_series` | Final RPCA-cleaned and smoothed series. |

---

## Complete Example

The following example demonstrates a typical workflow:

```python
import pandas as pd
from DataCleaning import DataCleaning

# Load time-series data
df = pd.read_excel(
    "meter_data.xlsx",
    sheet_name="Sheet1",
    index_col="Date / Time",
    parse_dates=True
)

# Create the data-cleaning object
cleaner = DataCleaning(
    dataframe=df,
    meter="Meter_Name",
    method="Hampel",
    window=12,
    n_sigma=3
)

# Detect and replace outliers
cleaned_df = cleaner.detect_outliers()

# Examine the results
print(cleaned_df.head())
print(cleaner.outliers)
print(cleaner.replacements)
print(cleaner.count)

# Save the cleaned data
cleaned_df.to_excel("cleaned_meter_data.xlsx")
```

---

## Choosing a Method

The choice of method depends on the structure of the time series.

| Method | Useful When |
| :--- | :--- |
| **Hampel** | A robust local filter is desired for time-series observations. |
| **Polynomial** | Local polynomial behavior is expected and a smooth fitted series can be used to identify unusual residuals. |
| **Prophet** | Seasonal time-series behavior can be modeled with Prophet. |
| **Fencing** | A simple distribution-based IQR rule is appropriate. |
| **Neural** | NeuralProphet-based forecasting and residual analysis are desired. |
| **Visual** | Extreme observations or specific timestamps need to be reviewed or manually selected. |
| **RPCA** | The data can be represented as a matrix with a low-rank structure plus sparse unusual observations. |

---

## Important Usage Notes

1. The `meter` argument should match an existing DataFrame column when a specific meter is being analyzed.
2. The `DataCleaning` constructor initializes the RPCA decomposition using the full DataFrame values before restricting the working DataFrame to the selected meter. Therefore, the input DataFrame must be compatible with the RPCA procedure.
3. The RPCA procedure requires a numeric matrix.
4. The time-series methods assume that the DataFrame index can be interpreted as dates/times where appropriate.
5. `window` affects several methods. For the polynomial method, an even window is automatically increased by one.
6. The class modifies the working data during cleaning, but the original DataFrame passed by the user remains separately available through the object's stored DataFrame attributes.
7. The exact output type of `outliers`, `replacements`, and `report` varies by detection method. Inspect the relevant attribute after running a method.

---

## Troubleshooting

### Invalid Method Error

If `detect_outliers()` reports that the method is invalid, check that the method name exactly matches one of the supported values:

```python
"Hampel"
"Polynomial"
"Prophet"
"Fencing"
"Neural"
"Visual"
```

### Invalid Meter Error

If a meter is not found, check the DataFrame column names:

```python
print(df.columns.tolist())
```

Then pass the exact column name to `meter`.

### Polynomial Window Error

The polynomial method requires:

$$\text{order} < \text{window}$$

For example, `window=13` and `order=2` is valid.

---

## Summary

A basic use of the class can be reduced to four steps:

1. Load the time-series data into a pandas DataFrame.
2. Create a `DataCleaning` object and specify the meter and detection method.
3. Run `detect_outliers()` or `rpca_detect()`.
4. Inspect `filtered`, `outliers`, and `replacements`.

For example:

```python
cleaner = DataCleaning(
    df,
    meter="Meter_Name",
    method="Fencing"
)

cleaned_df = cleaner.detect_outliers()

print(cleaner.outliers)
print(cleaner.replacements)
```
