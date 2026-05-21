"""Exploratory analysis and visualization for energy-style time series."""

import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
import pandas as pd 
from pandas.api.types import CategoricalDtype
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pymannkendall as mk
from statsmodels.graphics.tsaplots import plot_acf

class EDA:
    def __init__(self, dataframe, meter, covariates=None, weather=None, timeframe=None): 
        start_date = None
        end_date = None
        if covariates is not None:
            dataframe = pd.concat([dataframe, covariates], axis=1)
        
        dataframe.index = pd.to_datetime(dataframe.index)
        if dataframe.index.tz is not None:
            dataframe.index = dataframe.index.tz_localize(None)

        if timeframe is not None:
            try:
                start_date = timeframe[0] 
                end_date = timeframe[1]
                dataframe = dataframe.loc[start_date:end_date]
            except Exception as e:
                print(f"Warning: Could not apply timeframe {timeframe}. Ensure it is a valid date slice (e.g., ('2023-01-01', '2023-12-31')). Error: {e}")
        
        if weather is not None:
            weather.index = pd.to_datetime(weather.index)
            if weather.index.tz is not None:
                weather.index = weather.index.tz_localize(None)
            if start_date is not None and end_date is not None: 
                 weather = weather.loc[start_date:end_date]
            
        self.weather = weather
        self.dataframe = dataframe
        self.meter= meter
        self.timeframe = timeframe
        self.cat_type = CategoricalDtype(
            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            ordered=True
        )
        self.season_order = ['EarlyWinter', 'LateWinter', 'EarlySpring', 'LateSpring', 'Summer', 'EarlyFall', 'LateFall']
        self.features_and_target = None
        
    def ts_plot(self, title='Smoothed Time Series Plot', smoothing=['hp', 200], ylabel='Energy Usage', annotated=False):
        ts = self.dataframe[self.meter].to_frame()
        
        if smoothing[0] == 'ma':
            smoothed_ts = ts.rolling(window=smoothing[1]).mean()
        elif smoothing[0] == 'hp':
            cycle, trend = hpfilter(ts, lamb=smoothing[1])
            smoothed_ts = trend
        elif smoothing == 'None':
            smoothed_ts = None
        else:
            raise ValueError("Invalid smoothing method. Choose from 'ma', 'hp', 'None'")
        
        df = self.dataframe.copy() 
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df[self.meter], label='Energy Usage', color='tab:blue', marker='.', linestyle='-', markersize=4)
        
        if smoothed_ts is not None:
            ax.plot(smoothed_ts, color='tab:orange', linewidth=2, label=f'Smoothed ({smoothing[0]})')
        
        if annotated:
            def shade_and_label_by_name(column, target_name, color='gray'):
                mask = df[column] == target_name
                groups = (mask != mask.shift()).cumsum()
                for _, group in mask.groupby(groups):
                    if group.iloc[0]:
                        start = group.index[0]
                        end = group.index[-1]
                        ax.axvspan(start, end, color=color, alpha=0.3)
                        y_min, y_max = ax.get_ylim()
                        y_pos = y_min + 0.75 * (y_max - y_min)
                        ax.text(start + (end - start) / 2,
                                y_pos,
                                target_name,
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                rotation=90,
                                color='black')
            def shade_and_label(condition_column, label_column=None, default_label=None, color='gray'):
                mask = df[condition_column] == 0
                groups = (mask != mask.shift()).cumsum()
                for _, group in mask.groupby(groups):
                    if group.iloc[0]:
                        start = group.index[0]
                        end = group.index[-1]
                        ax.axvspan(start, end, color=color, alpha=0.3)
                        label = default_label
                        if label_column:
                            label = df.loc[start:end, label_column].mode().iloc[0]
                        y_min, y_max = ax.get_ylim()
                        y_pos = y_min + 0.75 * (y_max - y_min)
                        ax.text(start + (end - start) / 2,
                                y_pos,
                                label,
                                ha='center',
                                va='bottom',
                                fontsize=9,
                                rotation=90,
                                color='black')
            if 'is_not_weekend' in df.columns:
                shade_and_label('is_not_weekend', default_label='Weekend', color='gray')
            if 'Holiday_name' in df.columns:
                shade_and_label_by_name('Holiday_name', 'Winter Break', color='orange')
                shade_and_label('IsNotHoliday', label_column='Holiday_name', color='red')
            if 'IsnotSummerBreak' in df.columns:
                shade_and_label('IsnotSummerBreak', default_label='Summer Break', color='green')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Date')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_seasonal_usage(self, bins=[0, 215, 415, 531, 831, 1015, 1130, 1231],labels=['LateWinter', 'EarlySpring', 'LateSpring', 'Summer', 'EarlyFall', 'LateFall', 'EarlyWinter']):
        df = self.dataframe.copy()
        
        df['date'] = pd.Series(df.index, index=df.index)
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekday'] = df['date'].dt.day_name().astype(self.cat_type)
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['date_offset'] = df['month'] * 100 + df['dayofmonth']
        df['season'] = pd.cut(df['date_offset'], bins=bins, labels=labels, right=True, include_lowest=True)
        df['season'] = pd.Categorical(df['season'], categories=self.season_order, ordered=True)
        
        if 'InSession' in df.columns:
            df['session_status'] = df['InSession'].map({1: 'In Session', 0: 'Not in Session'})
        else:
            raise KeyError("'InSession' column not found in the dataframe. Make sure covariates were included.")

        df = df.dropna(subset=[self.meter, 'season', 'session_status'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df,
            x='season',
            y=self.meter,
            hue='session_status',
            palette='Set2',
            ax=ax,
            linewidth=1,
            dodge=True
        )
        ax.set_title('Energy Usage by Season and Session Status')
        ax.set_xlabel('Season')
        ax.set_ylabel('Energy Usage')
        ax.legend(title='Session Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_usage_by_temperature_InSession(self, intervals=[-10, 45, 60, 90, 120]):
        
        if self.weather is None:
            raise AttributeError("Weather data is required for 'plot_usage_by_temperature_InSession', but 'weather' was not passed to the EDA constructor.")
        
        weather_df = self.weather.copy()
        full_df = self.dataframe.copy()
        
        merged_df = pd.merge(
            full_df,
            weather_df,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        if 'Temperature (°F)' not in merged_df.columns:
             raise KeyError("'Temperature (°F)' column not found in the merged dataframe. Check your weather dataframe columns.")

        merged_df['temp_bin_interval'] = pd.cut(
            merged_df['Temperature (°F)'],
            bins=intervals,
            include_lowest=True,
            right=True
        )

        bin_categories = merged_df['temp_bin_interval'].cat.categories
        
        category_map = {
            c: f"({c.left:.2f}, {c.right:.2f}]"
            for c in bin_categories
        }

        merged_df['temp_bin'] = merged_df['temp_bin_interval'].map(category_map)

        merged_df['temp_bin'] = pd.Categorical(
            merged_df['temp_bin'],
            categories=list(category_map.values()),
            ordered=True
        )

        merged_df = merged_df.drop(columns=['temp_bin_interval'])

        merged_df['weekday'] = merged_df.index.day_name()
        merged_df['day_type'] = merged_df['weekday'].isin(['Saturday', 'Sunday']).map({True: 'Weekend', False: 'Weekday'})
        
        if 'InSession' in merged_df.columns:
            merged_df['session_status'] = merged_df['InSession'].map({1: 'In Session', 0: 'Not in Session'})
        else:
            raise KeyError("'InSession' column not found in the merged dataframe. Make sure covariates are included.")
            
        merged_df = merged_df.dropna(subset=[self.meter, 'temp_bin'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(data=merged_df,
                    x='temp_bin',
                    y=self.meter,
                    hue='session_status',
                    palette='coolwarm',
                    ax=ax,
                    linewidth=1)
                    
        ax.set_title('Energy Usage by Temperature Range and Session Status')
        ax.set_xlabel('Temperature Range (°F)')
        ax.set_ylabel('Energy Usage')
        ax.legend(title='Session Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def seasonal_decompose(self, model='additive', period=None):
        series = self.dataframe[self.meter].dropna()
        result = seasonal_decompose(series, model=model, period=period)
        result.plot()

    def run_trend_tests(self, seasonal_periods=24):
        ts = self.dataframe[self.meter].dropna()
        print("🔹 Mann-Kendall Test:")
        mk_result = mk.original_test(ts)
        print(f"Trend: {mk_result.trend}, p-value: {mk_result.p}, Tau: {mk_result.Tau}")
        print("\n🔹 Seasonal Kendall Test:")
        seasonal_result = mk.seasonal_test(ts, period=seasonal_periods)
        print(f"Trend: {seasonal_result.trend}, p-value: {seasonal_result.p}, Tau: {seasonal_result.Tau}")
    
    def autocorr_plot(self, lags=50, title='Autocorrelation Function (ACF) Plot'):
        """
        Generates an Autocorrelation Function (ACF) plot for the time series.
        ACF measures the correlation between a time series and a lagged version of itself.
        Parameters:
        - lags (int): The number of lags (time steps) to display on the x-axis.
        - title (str): The title of the plot.
        """
        try:
            # 1. Select the target series and drop NaNs for accurate calculation
            series = self.dataframe[self.meter].dropna()
            
            # 2. Check for minimum required data points (lags + 1)
            if len(series) < lags + 1:
                print(f"Warning: Not enough data points ({len(series)}) to calculate {lags} lags. Setting lags to max possible ({len(series) - 1}).")
                lags = len(series) - 1

            if lags < 1:
                print("Error: Time series has too few data points to calculate autocorrelation.")
                return

            # 3. Create the plot using statsmodels plot_acf
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_acf(series, lags=lags, ax=ax, title=title)
            
            ax.set_xlabel('Lags')
            ax.set_ylabel('Autocorrelation')
            
            # The blue shaded area represents the 95% confidence interval.
            # Any bar extending beyond this indicates statistically significant correlation.
            
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"An error occurred during ACF plotting: {e}")