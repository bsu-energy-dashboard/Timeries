"""Outlier detection and cleaning helpers for meter time series."""

import numpy as np
import pandas as pd
from hampel import hampel
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter
from prophet import Prophet
from neuralprophet import NeuralProphet

# withlocal
class DataCleaning:
    def __init__(self, dataframe, window=12, n_sigma=3, meter=None, max=5, min= 5,order=2, method=None,daily=False, weekly= False, imputem=True,iw=0.95):
        self.dataframe = dataframe
        self.window = window
        self.n_sigma = float(n_sigma)
        self.meter = meter
        self.method = method
        self.all_column_outliers = {}
        self.dataframe1 = self.dataframe
        self.iw=float(iw)
        self.max = int(max)
        self.min=int(min)
        self.filtered = self.dataframe.copy()
        self.order=order
        self.daily = daily
        self.weekly = weekly 

        # Handle column filtering by meter
        if meter is not None:
            if meter not in self.dataframe.columns:
                raise ValueError(
                    f"Invalid meter: '{meter}'. "
                    f"Meter must be one of: {self.dataframe.columns.tolist()}"
                )
            self.dataframe = self.dataframe[[meter]]
        else:
            self.dataframe = self.dataframe.select_dtypes(include='number')

        self.missingcounts = self.__missing_values()
        self.zerocounts = self.__zero_values()
    
        if imputem:
            if self.dataframe.isnull().any().any():
                self.dataframe = self.dataframe.interpolate(method="polynomial", order=2)

    # --- Internal missing/zero checks ---
    def __missing_values(self):
        
        if self.dataframe[[self.meter]].isnull().sum().sum() != 0:
            data = {'Meter Name': [], 'Number of Missing': [], 'Missing Indices': []}
            for i in self.dataframe[[self.meter]]:
                missing_indices = self.dataframe1[self.dataframe[i].isnull()].index.tolist()
                if missing_indices:
                    data['Meter Name'].append(self.meter)
                    data['Number of Missing'].append(len(missing_indices))
                    data['Missing Indices'].append(missing_indices)
            return pd.DataFrame(data)
        else:
            return "There are no missing values in the DataFrame"

    def __zero_values(self):
        if (self.dataframe == 0).sum().sum() != 0:
            data = {'Meter Name': [], 'Number of Zeros': [], 'Zero Indices': []}
            for col in self.dataframe.columns:
                zero_indices = self.dataframe[self.dataframe[col] == 0].index.tolist()
                if zero_indices:
                    data['Meter Name'].append(col)
                    data['Number of Zeros'].append(len(zero_indices))
                    data['Zero Indices'].append(zero_indices)
            return pd.DataFrame(data)
        else:
            return "There are no zero values in the DataFrame"
    def detect_outliers(self):
        if self.method == 'Hampel':
            return self.hampel()
        elif self.method == 'Polynomial':
            return self.polynomial()
        elif self.method == 'Prophet':
            return self.prophet()
        elif self.method == 'Fencing':
            return self.fencing()
        elif self.method == 'Neural':
            return self.neural()
        elif self.method == 'Visual':
            return self.visual()
        else: 
            raise ValueError(
                "Not a valid outlier detection method. "
                "The methods of outlier detection are 'Hampel','Fencing','Polynomial', 'Prophet','Neural','Fencing' and 'Visual'.")
        

    # --- Method to get Hampel outliers ---
    def hampel(self):
        if self.method != 'Hampel':
            raise ValueError(f'The method you have specified is {self.method}.')
        else:
            #creating a list called all_outlier_indices which will later be EXTENEDED with outlier indices
            all_outlier_indices = []
            #creating a list called all_outlier_values which will later be EXTENEDED with outlier values
            all_outlier_values = []
            #creating a list called all_replacements which will later be EXTENED with outlier replacements
            all_replacements = []
            #creating a copy of the original dataframe which will later be filtered with all outlier replacements
            filtered_df = self.dataframe.copy()
            #creating a for loop for all columns in the dataframe
            for col in self.dataframe.columns:
                #creating a variable called values which are the actual values from that column
                values = self.dataframe[col]
                #creating a hampel object called hampel_obj that takes the values, window, and n_sigma as arguments
                hampel_obj = hampel(values, self.window, self.n_sigma)
                #importing math for later use
                import math
                #creating a variable called first_w which are the values UP TO the floored values of window/2
            first_w = values[:math.floor(self.window/2)].copy()
            #first quartile
            Q1 = first_w.quantile(0.25)
            #third quartile
            Q3 = first_w.quantile(0.75)
            #interquartile range
            IQR = Q3 - Q1
            #calculating the median for the first values
            median_val = first_w.median()
            #lower bound, notice that it depends on the inputted value of n_sigma
            lower_bound = Q1 - self.n_sigma * IQR
            #upper bound, notice that it depends on the inputted value of n_sigma
            upper_bound = Q3 + self.n_sigma * IQR
            #boolean list of the conditionals
            outlier_mask = (first_w < lower_bound) | (first_w > upper_bound)
            #replacements the first_w where the outlier_mask was true with the median_val
            first_w[outlier_mask] = median_val
            #locating the values in the filtered_df[col] up to that and replacing it with the filtered values
            filtered_df.loc[filtered_df.index[:math.floor(self.window/2)], col] = first_w.values
            #creating a variable called last_w which are the values STARTING AT the NEGATIVE floored values of window/2
            last_w = values[-math.floor(self.window/2):].copy()
            #first quartile of last_w
            Q1l = last_w.quantile(0.25)
            #third quartile of last_w
            Q3l = last_w.quantile(0.75)
            #interquartile range
            IQRl = Q3l-Q1l
            #median of last_w
            median_vall = last_w.median()
            #lower bound, notice that it depends on the inputted n_sigma
            lower_boundl = Q1l - self.n_sigma * IQRl
            #upper bound, notice that it dependds on the inputted n_sigma
            upper_boundl = Q3l + self.n_sigma * IQRl
            #bolean list for the conditional
            outlier_maskl = (last_w<lower_boundl) | (last_w> upper_boundl)
            #replacing values in last_w where outlier_maskl was true with the median value
            last_w[outlier_maskl]=median_vall
            #locating values in filtered_df[col] starting at the negative of that and replacing it with the filtered last_w values
            filtered_df.loc[filtered_df.index[-math.floor(self.window/2):], col] = last_w.values

            #getting the index of the first_w observations where outlier_mask was true and converting it to a list 
            local_outlier_indices = first_w[outlier_mask].index.tolist()
            #locating the original values in dataframe which were considered outliers from the local_outlier_indices
            local_outlier_values = values.loc[local_outlier_indices].values
            #locating the outliers values in first_w whose values have been changed
            local_replacements = first_w.loc[local_outlier_indices].values

            #getting the index of the last_w observations where outlier_maskl was true and converting it to a list 
            local_outlier_indicesl = last_w[outlier_maskl].index.tolist()
            #locating the original values in dataframe which were considered outliers from the local_outlier_indicesl
            local_outlier_valuesl = values.loc[local_outlier_indicesl].values
            #locating the outliers values in last_w whose values have been changed
            local_replacementsl = last_w.loc[local_outlier_indicesl].values


            #finding the indices in the original dataframe where hampel has considered them outliers
            hampel_outlier_indices = self.dataframe.index[hampel_obj.outlier_indices].tolist()
            #locating the original values in the values where hampel considered it an outlier
            hampel_outlier_values = values.iloc[hampel_obj.outlier_indices].values
            #finding the values in the filtered_data that were replaced
            #.filtered_data and .outlier_indices are attributes from the hampel package
            hampel_replacements = hampel_obj.filtered_data[hampel_obj.outlier_indices].values


            #copying the filtered_df after first and last outliers have been replaced
            filtered_values = filtered_df[col].copy()
            #locating the outlier indices  from hampel package in the filterd_values and setting them equal to the replaced values 
            filtered_values.iloc[hampel_obj.outlier_indices] = hampel_obj.filtered_data[hampel_obj.outlier_indices]
            #filtered_df is now filted_values
            filtered_df[col] = filtered_values

                #combining all indices
            combined_indices = local_outlier_indices + hampel_outlier_indices + local_outlier_indicesl 
            #combining all outlier values
            combined_values = list(local_outlier_values) + list(hampel_outlier_values) + list(local_outlier_valuesl)
            #combing all repalcements 
            combined_replacements = list(local_replacements) + list(hampel_replacements) + list(local_replacementsl)

            #filling the all_outlier_indices with the values from the combined_indices
            all_outlier_indices.extend(combined_indices)
            #filling the all_outlier_values with the orignal values
            all_outlier_values.extend(combined_values)
            #filling the all_replacements with the replaced values
            all_replacements.extend(combined_replacements)

            # creating attribute of outliers2 as dataframe (i.e, metername, number of outliers, and outlier indices)
            #creating attribute of filtered dataframe which has the cleaned dataframe rid of outliers
            self.filtered = filtered_df
            #dataframe with datatime and ORIGINAL values  of the outliers 
            self.count = f"There were {len(hampel_outlier_indices)} outlier(s) detected from the Hampel Filter and {len(local_outlier_indices)+len(local_outlier_indicesl)} outlier(s) detected from fencing."
            self.outliers = pd.DataFrame({
                "DateTime": pd.to_datetime(all_outlier_indices),
                "Value": all_outlier_values
            }).set_index("DateTime")
    #dataframe with datetime, value before replacement, and value after replacement 
            self.replacements = pd.DataFrame({
                "DateTime": pd.to_datetime(all_outlier_indices),
                "Before Replacement": all_outlier_values,
                "After Replacement": all_replacements
            }).set_index("DateTime")
            
            #returns clean dataframe

            return self.filtered
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter
        # The standard median_absolute_deviation function is removed for wider compatibility.
        # We will calculate MAD directly using numpy.

    def polynomial(
    self,
    value_col: str = None
):
        if self.method != 'Polynomial':
            raise ValueError(f'The method specified is {self.method}')
        else:
            if value_col is None:
                if self.meter is None:
                    raise ValueError("No column specified: provide value_col or set self.meter")
                value_col = self.meter
            if self.window % 2 == 0:
                self.window += 1

            if self.order >= self.window:
                raise ValueError("Polynomial order must be less than the window size.")
            #report tells residuals and stuff
            report = self.dataframe.copy()
            #filtered is updated dataframe with outliers replaced
            filtered = self.dataframe.copy()
            #

            # 1. Apply the local polynomial fit (Savitzky-Golay Filter)
            # This efficiently performs the piecewise polynomial regression.
            y_values = report[value_col].values

            # The Savitzky-Golay filter requires the input to be numeric.
            try:
                y_fit = savgol_filter(
                    x=y_values,
                    window_length=self.window,
                    polyorder=self.order,
                    mode='nearest' # Use nearest valid data points at the edges
                )
            except Exception as e:
                print(f"Error applying Savitzky-Golay filter: {e}")
                return report.assign(y_fit=np.nan, is_outlier=False)

            report['y_fit'] = y_fit
            report['residual'] = report[value_col] - report['y_fit']
            
            # Drop NaNs just in case, though savgol_filter handles edges
            residuals = report['residual'].dropna()

            # 2. Calculate the robust threshold using MAD (Median Absolute Deviation)
            # MAD is calculated manually for better compatibility: MAD = median(|residual - median(residual)|)
            residual_median = np.median(residuals)
            mad = np.median(np.abs(residuals - residual_median))
            
            # The scale factor (1.4826) makes the MAD consistent with the standard deviation
            # for normally distributed data, which is useful for setting thresholds.
            scale_factor = 1.4826
            sigma_estimate = mad * scale_factor

            # 3. Identify Outliers
            threshold = self.n_sigma * sigma_estimate
            report['is_outlier'] = np.abs(report['residual']) > threshold
        
            
            self.report = report[report['is_outlier']]

               


            # negative-fitted-values among outliers
           

            # Use masks aligned to the full dataframe (report)
            mask_outlier = report['is_outlier']                     # boolean series indexed as dataframe
            mask_negative_fit = report['y_fit'] < 0

            # Replace outliers:
            # - if fitted value is negative -> use median of the original column
            # - else -> use the fitted value
            median_value = self.dataframe.median()
            filtered.loc[mask_outlier & mask_negative_fit, value_col] = self.dataframe.median()
            filtered.loc[mask_outlier & ~mask_negative_fit, value_col] = report.loc[mask_outlier & ~mask_negative_fit, 'y_fit']

            self.filtered = filtered
        
            outlieridx = list(report.index[report['is_outlier']])


            polyreplacements = {'DateTime':[], 'Before Replacement': [], 'After Replacement': []}

            polyreplacements['DateTime'].extend(self.dataframe.loc[outlieridx, value_col].index)

            # Original values for the outliers
            polyreplacements['Before Replacement'].extend(self.dataframe.loc[outlieridx, value_col].values)

            # Replacement values from y_fit
    
    
            polyreplacements['After Replacement'].extend(report.loc[outlieridx, 'y_fit'].values)

            self.replacements = pd.DataFrame(polyreplacements).set_index('DateTime')
            
            
            
            outliers = pd.DataFrame({
                "Value": self.dataframe.loc[outlieridx, value_col].values
            }, index=pd.to_datetime(outlieridx))
                    
            count = f"There were {len(outlieridx)} outlier(s) detected from Piecewise Polynomial Detection. "
            self.count = count

            self.outliers = pd.DataFrame(outliers)
            
        

            return filtered
    import matplotlib.pyplot as plt
    def prophet(self,yearly=True, weekly=True):
        import logging
        logging.getLogger("cmdstanpy").disabled = True
        if self.method != 'Prophet':
            raise ValueError(f'The method you have specified is {self.method}.')
        else:
            run = self.dataframe.copy()
            model = Prophet(interval_width=self.iw, yearly_seasonality=yearly, weekly_seasonality=weekly)
            run_df = run.reset_index()
            run_df.columns = ['ds', 'y']
            model.fit(run_df)
            forecast = model.predict(run_df)
            performance = pd.merge(run_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
            performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
            anomalies = performance[performance['anomaly']==1].sort_values(by='ds')
            p1 = performance.set_index('ds')
            p1[p1['anomaly']==1]
            run.loc[p1['anomaly'] == 1, self.meter] = p1.loc[p1['anomaly'] == 1, 'yhat']
            #.filtered
            self.filtered = run 
            report = anomalies[anomalies['anomaly'] == True]
            report['Residual'] = abs(report['y']-report['yhat'])
            report1 = report[['ds','y','yhat','yhat_lower','yhat_upper','Residual']].rename(columns={
                'ds': 'Date / Time',
                'y': 'Y Actual',
                'yhat': 'Y Fit',
                'yhat_lower' : 'Y Fit Lower Bound',
                'yhat_upper': 'Y Fit Upper Bound'

            }).style.hide(axis="index")
            self.report = report1
            #. count 
            self.count = f"There were {len(anomalies)} outlier(s) detected from the Prophet Model. "
            #. outliers
            self.outliers = self.dataframe[p1['anomaly']==1]
            #.replacements
            prophetreplacements = {'DateTime':[], 'Before Replacement': [], 'After Replacement': []}

            prophetreplacements['DateTime'].extend(self.dataframe.loc[p1['anomaly'] == 1, self.meter].index)

            # Original values for the outliers
            prophetreplacements['Before Replacement'].extend(self.dataframe.loc[p1['anomaly'] == 1, self.meter].values)

            # Replacement values from y_fit
            prophetreplacements['After Replacement'].extend(p1.loc[p1['anomaly'] == 1, 'yhat'].values)

            self.replacements = pd.DataFrame(prophetreplacements).set_index('DateTime')


        return self.filtered 
    import pandas as pd

    import pandas as pd

    def fencing(self):
        if self.method != 'Fencing':
            raise ValueError(f'The method you have specified is {self.method}.')

        filtered_df = self.dataframe.copy()
        all_outlier_indices = []
        all_outlier_values = []
        all_replacements = []

        for col in self.dataframe.columns:
            values = filtered_df[col].copy()

            # Calculate quartiles and IQR
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            median_val = values.median()

            # Fencing bounds
            lower_bound = Q1 - self.n_sigma * IQR
            upper_bound = Q3 + self.n_sigma * IQR

            # Identify outliers
            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_indices = values[outlier_mask].index.tolist()
            outlier_values = values.loc[outlier_indices].values
            replacements = [median_val] * len(outlier_indices)

            # Replace outliers with median
            values.loc[outlier_mask] = median_val
            filtered_df[col] = values

            # Collect all outlier info
            all_outlier_indices.extend(outlier_indices)
            all_outlier_values.extend(outlier_values)
            all_replacements.extend(replacements)

        # Save results in the object
        self.filtered = filtered_df
        self.count = f"There were {len(all_outlier_indices)} outlier(s) detected from Fencing."
        
        self.outliers = pd.DataFrame({
            "DateTime": pd.to_datetime(all_outlier_indices),
            "Value": all_outlier_values
        }).set_index("DateTime")
        
        self.replacements = pd.DataFrame({
            "DateTime": pd.to_datetime(all_outlier_indices),
            "Before Replacement": all_outlier_values,
            "After Replacement": all_replacements
        }).set_index("DateTime")

        return self.filtered
    def neural(self):
        if self.method != 'Neural':
            raise ValueError(f'The method you have specified is {self.method}.')
        df = self.dataframe.copy()
        df = df.reset_index()   # index → column
        df.columns = ['ds', 'y']
        m = NeuralProphet(n_changepoints = 0,n_lags=10,
        weekly_seasonality=self.weekly,
        daily_seasonality=self.daily)
        
        metrics = m.fit(df)
        forecast = m.predict(df)
        forecast['residual'] = abs(forecast['y'] - forecast['yhat1'])

        # Assuming 'residuals' is a NumPy array of model residuals
        abs_residuals = np.abs(forecast['residual'])
        Q1 = abs_residuals.quantile(0.25)
        Q3 = abs_residuals.quantile(0.75)
        IQR = Q3 - Q1 
        lower = Q1 - self.n_sigma*IQR
        upper = Q3 + self.n_sigma *IQR
        forecast['outlier'] = ((abs_residuals < lower) | (abs_residuals > upper)).astype(int)
        self.forecast = forecast
        outliers = self.forecast[self.forecast['outlier'] == 1]
        outliers1 = outliers[['ds', 'y']]
        outliers1 = outliers1.rename(columns={
            'ds': 'Date / Time',
            'y': f'{self.meter}'
        })
        self.outliers = outliers1
        replacements = outliers[['ds', 'y', 'yhat1']]
        replacements = replacements.rename(columns={
            'ds': 'Date / Time',
            'y' : 'Before Replacement',
            'yhat1': 'After Replacement'
        })

        self.outliers = outliers1.style.hide(axis="index")
        self.replacements = replacements.style.hide(axis="index")
        self.report = (
        outliers[['ds', 'y', 'yhat1', 'residual']]
        .rename(columns={
            'ds': 'Date / Time',
            'y': 'Y Actual',
            'yhat1': 'Y Fit',
            'residual': 'Residual'
        })
        .style.hide(axis="index")
)

        filtered = self.dataframe.copy()
        forecast_subset = forecast[forecast['outlier'] == 1]
        filtered.loc[filtered.index.isin(outliers['ds']), self.meter] = outliers['yhat1'].values


        # If filtered index is datetime matching forecast['ds'], align
        filtered.loc[filtered.index.isin(forecast_subset['ds']), self.meter] = forecast_subset['yhat1'].values
        self.filtered = filtered


        

                


        return self.filtered
    def sorted(self,ascending=True):
        sort = self.dataframe.sort_values(by=self.meter, ascending=ascending)
        self.sort = sort
        sort_min = sort.iloc[:self.min]
        sort_max = sort.iloc[-self.max:] if max != 0 else pd.DataFrame(columns=sort.columns)
        string_min = {f'The {self.min} smallest values are at the indices {sort_min.index}'}
        string_max ={f'The {self.max} largest values are at the indices {sort_max.index} '}
        self.minval = sort_min
        self.maxval = sort_max
        self.indexmin = sort_min.index
        self.indexmax = sort_max.index

        return string_min,string_max

    def visual(self, ts=None, all=False):
        if self.method != 'Visual':
            raise ValueError(f'The method you have specified is {self.method}.')

        filtered = self.dataframe.copy()

        if ts is None:
            sort = filtered.sort_values(by=self.meter, ascending=True)
            self.sort = sort

            sort_min = sort.iloc[:self.min]
            sort_max = sort.iloc[-self.max:] if self.max > 0 else sort.iloc[0:0]

            self.min_df = sort_min
            self.max_df = sort_max
            self.indexmin = sort_min.index
            self.indexmax = sort_max.index

            mask = filtered.index.isin(self.indexmin.union(self.indexmax))

        else:
            if isinstance(ts, (list, tuple, pd.Index)):
                select_datetimes = pd.to_datetime(ts)
            else:
                select_datetimes = pd.to_datetime([ts])

            mask = filtered.index.isin(select_datetimes)

        filtered.loc[mask, self.meter] = np.nan
        self.filtered = filtered

        if filtered.loc[mask, self.meter].isna().any():
            interp = self.filtered[self.meter].interpolate(
                method="polynomial",
                order=2
            )
            self.filtered.loc[mask, self.meter] = interp.loc[mask]


        # Before / After replacements
        replacements = self.dataframe.loc[mask, [self.meter]].copy()
        replacements['After Replacements'] = self.filtered.loc[mask, self.meter]
        self.replacements = replacements

        return self.filtered[[self.meter]]
