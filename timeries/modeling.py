"""Time series forecasting with Prophet, NeuralProphet, or SARIMAX."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
# specific import for NeuralProphet
from neuralprophet import NeuralProphet
# specific imports for SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ModelingClass:
    def __init__(self, dataframe, meter, weather=None, covariates=None, features=None, 
                 model='Prophet', test_hours=168, interval_width=0.95, 
                 interval_type='two-sided',  # 'two-sided', 'upper-bounded', or 'lower-bounded'
                 n_lags=24, n_forecasts=1,
                 sarimax_order=(2, 1, 1), sarimax_seasonal_order=(1, 1, 1, 24),
                 forecast_only=False):  # NEW: if True, train on all data and forecast forward
        """
        Initialize the modeling class.
        
        Args:
            dataframe (pd.DataFrame): Source data.
            meter (str): Target column.
            weather (pd.DataFrame): Weather data.
            covariates (pd.DataFrame): External features.
            features (str/list): Specific feature columns to use.
            model (str): 'Prophet', 'NeuralProphet', or 'SARIMAX'.
            test_hours (int): If forecast_only=False: hours to hold out for testing.
                             If forecast_only=True: hours to forecast into the future.
                             When forecast_only=True, covariates and weather must extend
                             for len(dataframe) + test_hours rows.
            interval_width (float): Confidence level (0.95 = 95%).
            interval_type (str): 'two-sided', 'upper-bounded', or 'lower-bounded'.
            n_lags (int): (NP only) Number of past hours to look back.
            n_forecasts (int): (NP only) Number of steps to predict at once.
            sarimax_order (tuple): (p,d,q) order for SARIMAX.
            sarimax_seasonal_order (tuple): (P,D,Q,s) seasonal order for SARIMAX.
            forecast_only (bool): If True, train on ALL data and forecast forward. 
                                  If False, use train/test split for assessment.
        """
        self.dataframe = dataframe[[meter]]
        self.weather = weather 
        self.model = model
        self.test_hours = test_hours
        self.interval_width = interval_width
        self.interval_type = interval_type
        self.n_lags = n_lags
        self.n_forecasts = n_forecasts
        self.sarimax_order = sarimax_order
        self.sarimax_seasonal_order = sarimax_seasonal_order
        self.forecast_only = forecast_only  # NEW

        # Validate interval_type
        if self.interval_type not in ['two-sided', 'upper-bounded', 'lower-bounded']:
            raise ValueError("interval_type must be 'two-sided', 'upper-bounded', or 'lower-bounded'")

        # --- Covariates & Features Logic ---
        if isinstance(covariates, pd.DataFrame):
            if features is None:
                raise ValueError("If 'covariates' is provided, you must specify 'features'.")
            check_features = [features] if isinstance(features, str) else features
            missing_cols = [f for f in check_features if f not in covariates.columns]
            if missing_cols:
                raise ValueError(f"Features not found: {missing_cols}")
            self.covariates = covariates[check_features]
            self.features = check_features
        else:
            self.covariates = None
            self.features = []

        # --- Data Prep ---
        # All models benefit from the standardized 'ds', 'y' format
        self.prophet_data = self.data_transform()

        # --- Validate data lengths for forecast_only mode ---
        if self.forecast_only:
            required_length = len(self.dataframe) + self.test_hours
            
            if isinstance(self.covariates, pd.DataFrame):
                if len(self.covariates) < required_length:
                    raise ValueError(
                        f"forecast_only=True requires covariates to extend into the forecast period.\n"
                        f"Current covariates length: {len(self.covariates)}\n"
                        f"Required length: {required_length} (dataframe: {len(self.dataframe)} + forecast: {self.test_hours})\n"
                        f"Please provide covariates with {required_length - len(self.covariates)} additional rows."
                    )
            
            if isinstance(self.weather, pd.DataFrame):
                if len(self.weather) < required_length:
                    raise ValueError(
                        f"forecast_only=True requires weather data to extend into the forecast period.\n"
                        f"Current weather length: {len(self.weather)}\n"
                        f"Required length: {required_length} (dataframe: {len(self.dataframe)} + forecast: {self.test_hours})\n"
                        f"Please provide weather data with {required_length - len(self.weather)} additional rows."
                    )

        # --- Routing ---
        if self.model == 'Prophet': 
            self.forecast_df, self.fitted_model = self.fit_prophet()
        elif self.model == 'NeuralProphet':
            self.forecast_df, self.fitted_model = self.fit_neuralprophet()
        elif self.model == 'SARIMAX':
            self.forecast_df, self.fitted_model = self.fit_sarimax()

    def data_transform(self):
        """Prepares data: joins covariates/weather and renames to 'ds', 'y'."""
        df_p = self.dataframe.copy()
        
        # Merge Covariates
        if isinstance(self.covariates, pd.DataFrame): 
            df_p = df_p.join(self.covariates, how='left')
            df_p = df_p.reset_index()
            df_p.columns = ['ds', 'y'] + self.features
        else: 
            df_p = df_p.reset_index()
            df_p.columns = ['ds', 'y']
            
        df_p['ds'] = pd.to_datetime(df_p['ds'])
        
        # Merge Weather
        if isinstance(self.weather, pd.DataFrame): 
            w_copy = self.weather.copy().reset_index()
            w_copy.columns = ['ds', 'weather']
            df_p = pd.merge(df_p, w_copy, on='ds', how='left')
            
        return df_p 

    def fit_prophet(self):
        """Fit standard Prophet model."""
        m = Prophet(weekly_seasonality=True, daily_seasonality=True, interval_width=self.interval_width)
        
        if self.forecast_only:
            # Train on ALL data
            train = self.prophet_data.copy()
        else:
            # Train/test split
            train = self.prophet_data.iloc[:(len(self.dataframe) - self.test_hours)]
        
        if self.features:
            for feature in self.features: m.add_regressor(feature)
        if isinstance(self.weather, pd.DataFrame): 
            m.add_regressor('weather')
            
        m.fit(train)
        
        if self.forecast_only:
            # Forecast into the future
            future = m.make_future_dataframe(periods=self.test_hours, freq='h')
            
            # Fill future regressors (we've validated these extend far enough)
            if self.features:
                future[self.features] = self.covariates[self.features].values[:len(future)]
                        
            if isinstance(self.weather, pd.DataFrame):
                future['weather'] = self.weather['Temperature (°F)'].values[:len(future)]
        else:
            # Train/test split mode - original behavior
            future = m.make_future_dataframe(periods=self.test_hours, freq='h')
            
            if self.features:
                future[self.features] = self.covariates[self.features].values
            if isinstance(self.weather, pd.DataFrame):
                future['weather'] = self.weather['Temperature (°F)'].values
        
        forecast = m.predict(future)

        # Calculate all bound types
        std_dev = (forecast['yhat_upper'] - forecast['yhat_lower']) / (2 * 1.96)
        
        forecast['yhat_lower'] = forecast['yhat_lower']
        forecast['yhat_upper'] = forecast['yhat_upper']
        
        z_upper = self._get_one_sided_z_score(self.interval_width)
        forecast['yhat_upper_ub'] = forecast['yhat'] + z_upper * std_dev
        
        z_lower = self._get_one_sided_z_score(self.interval_width)
        forecast['yhat_lower_lb'] = forecast['yhat'] - z_lower * std_dev

        # Merge actual 'y' values back
        forecast = pd.merge(forecast, self.prophet_data[['ds', 'y']], on='ds', how='left')

        return (forecast, m)

    def fit_neuralprophet(self):
        """Fit NeuralProphet with Iterative Loop."""
        total_len = len(self.prophet_data)
        
        if self.forecast_only:
            # Train on ALL data
            train_df = self.prophet_data.copy()
            # Create empty future dataframe for forecasting
            last_date = train_df['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                         periods=self.test_hours, freq='h')
            test_df = pd.DataFrame({'ds': future_dates})
            
            # Add features and weather to test_df (we've validated these extend far enough)
            start_idx = len(train_df)
            end_idx = start_idx + self.test_hours
            
            if self.features:
                for feature in self.features:
                    test_df[feature] = self.covariates[feature].values[start_idx:end_idx]
            
            if isinstance(self.weather, pd.DataFrame):
                test_df['weather'] = self.weather['Temperature (°F)'].values[start_idx:end_idx]
        else:
            # Train/test split
            train_df = self.prophet_data.iloc[:(total_len - self.test_hours)].copy()
            test_df = self.prophet_data.iloc[-self.test_hours:].copy()

        # Filter Constant Features
        valid_features = []
        if self.features:
            for feature in self.features:
                if train_df[feature].nunique() <= 1:
                    print(f"Warning: Feature '{feature}' is constant in training data. Dropping from NeuralProphet.")
                    train_df = train_df.drop(columns=[feature])
                    if feature in test_df.columns:
                        test_df = test_df.drop(columns=[feature])
                else:
                    valid_features.append(feature)

        # Calculate Quantiles
        alpha = 1 - self.interval_width
        quantiles_two_sided = [round(alpha / 2, 2), round(1 - alpha / 2, 2)]
        quantiles_upper = [0.5, round(self.interval_width, 2)]
        quantiles_lower = [round(1 - self.interval_width, 2), 0.5]
        all_quantiles = sorted(list(set(quantiles_two_sided + quantiles_upper + quantiles_lower)))
        
        # Initialize Model
        m = NeuralProphet(
            n_lags=self.n_lags,
            n_forecasts=self.n_forecasts,
            quantiles=all_quantiles,
            weekly_seasonality=True,
            daily_seasonality=True,
            epochs=30
        )
        
        # Add Regressors
        for feature in valid_features:
            m.add_future_regressor(feature)
            
        if isinstance(self.weather, pd.DataFrame):
            m.add_future_regressor("weather")

        # Fit
        metrics = m.fit(train_df, freq="h")

        # Iterative Forecasting Loop
        results = self._iterative_np_loop(m, train_df, test_df, quantiles_two_sided, quantiles_upper, quantiles_lower)
        predictions = results['predictions']
        lower_two_sided = results['lower_two_sided']
        upper_two_sided = results['upper_two_sided']
        upper_ub = results['upper_ub']
        lower_lb = results['lower_lb']

        # Package results
        if self.forecast_only:
            # Create new dataframe with historical + forecast
            full_df = self.prophet_data.copy()
            
            # Append forecast rows
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'y': np.nan,
                'yhat': predictions,
                'yhat_lower': lower_two_sided,
                'yhat_upper': upper_two_sided,
                'yhat_upper_ub': upper_ub,
                'yhat_lower_lb': lower_lb
            })
            
            # Add feature columns if they exist
            if self.features:
                for feature in self.features:
                    if feature in full_df.columns:
                        forecast_df[feature] = np.nan
            if 'weather' in full_df.columns:
                forecast_df['weather'] = np.nan
                
            full_df = pd.concat([full_df, forecast_df], ignore_index=True)
        else:
            # Original behavior
            full_df = self.prophet_data.copy()
            full_df['yhat'] = np.nan
            full_df['yhat_lower'] = np.nan
            full_df['yhat_upper'] = np.nan
            full_df['yhat_upper_ub'] = np.nan
            full_df['yhat_lower_lb'] = np.nan
            
            start_idx = total_len - self.test_hours
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat')] = predictions
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_lower')] = lower_two_sided
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_upper')] = upper_two_sided
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_upper_ub')] = upper_ub
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_lower_lb')] = lower_lb

        return (full_df, m)

    def fit_sarimax(self):
        """Fit SARIMAX model."""
        df = self.prophet_data.copy()
        df = df.set_index('ds')
        
        # Handle Exogenous Variables
        exog_cols = self.features.copy()
        
        if 'weather' in df.columns:
            scaler = StandardScaler()
            df['weather'] = scaler.fit_transform(df[['weather']])
            exog_cols.append('weather')

        if self.forecast_only:
            # Train on ALL data, forecast forward
            y_train = df['y']
            exog_train = df[exog_cols] if exog_cols else None
            
            # For forecasting, get future exogenous variables (we've validated they exist)
            start_idx = len(df)
            end_idx = start_idx + self.test_hours
            
            if exog_cols:
                # Create future exog dataframe
                exog_test = pd.DataFrame(index=range(self.test_hours))
                for col in exog_cols:
                    if col == 'weather':
                        # Weather is already scaled in df, need to scale the future values
                        future_weather = self.weather['Temperature (°F)'].values[start_idx:end_idx]
                        # Use the same scaler that was fit on training data
                        scaler = StandardScaler()
                        scaler.fit(df[['weather']])
                        exog_test['weather'] = scaler.transform(future_weather.reshape(-1, 1)).flatten()
                    else:
                        exog_test[col] = self.covariates[col].values[start_idx:end_idx]
            else:
                exog_test = None
        else:
            # Train/test split
            train_end = len(df) - self.test_hours
            y_train = df['y'].iloc[:train_end]
            exog_train = df[exog_cols].iloc[:train_end] if exog_cols else None
            exog_test = df[exog_cols].iloc[train_end:] if exog_cols else None

        # Fit Model
        model = SARIMAX(
            y_train, 
            exog=exog_train, 
            order=self.sarimax_order, 
            seasonal_order=self.sarimax_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        
        # Get forecasts
        alpha_two_sided = 1 - self.interval_width
        forecast_two_sided = results.get_forecast(steps=self.test_hours, exog=exog_test)
        summary_two_sided = forecast_two_sided.summary_frame(alpha=alpha_two_sided)
        
        alpha_one_sided = 2 * (1 - self.interval_width)
        forecast_one_sided = results.get_forecast(steps=self.test_hours, exog=exog_test)
        summary_one_sided = forecast_one_sided.summary_frame(alpha=alpha_one_sided)
        
        # Package Results
        if self.forecast_only:
            # Create dataframe with historical + forecast
            full_df = self.prophet_data.copy()
            
            # Create forecast dates
            last_date = full_df['ds'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                          periods=self.test_hours, freq='h')
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'y': np.nan,
                'yhat': summary_two_sided['mean'].values,
                'yhat_lower': summary_two_sided['mean_ci_lower'].values,
                'yhat_upper': summary_two_sided['mean_ci_upper'].values,
                'yhat_upper_ub': summary_one_sided['mean_ci_upper'].values,
                'yhat_lower_lb': summary_one_sided['mean_ci_lower'].values
            })
            
            full_df = pd.concat([full_df, forecast_df], ignore_index=True)
        else:
            # Original behavior
            full_df = self.prophet_data.copy()
            full_df['yhat'] = np.nan
            full_df['yhat_lower'] = np.nan
            full_df['yhat_upper'] = np.nan
            full_df['yhat_upper_ub'] = np.nan
            full_df['yhat_lower_lb'] = np.nan

            start_idx = len(y_train)
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat')] = summary_two_sided['mean'].values
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_lower')] = summary_two_sided['mean_ci_lower'].values
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_upper')] = summary_two_sided['mean_ci_upper'].values
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_upper_ub')] = summary_one_sided['mean_ci_upper'].values
            full_df.iloc[start_idx:, full_df.columns.get_loc('yhat_lower_lb')] = summary_one_sided['mean_ci_lower'].values
        
        return (full_df, results)

    def _get_one_sided_z_score(self, confidence_level):
        """Calculate z-score for one-sided confidence interval."""
        from scipy import stats
        return stats.norm.ppf(confidence_level)

    def _iterative_np_loop(self, model, train_data, test_data, quantiles_two_sided, quantiles_upper, quantiles_lower):
        """Helper method to run the blind feedback loop and extract all bound types."""
        lower_two_pct = f"{int(quantiles_two_sided[0] * 100)}%"
        upper_two_pct = f"{int(quantiles_two_sided[1] * 100)}%"
        upper_ub_pct = f"{int(quantiles_upper[1] * 100)}%"
        lower_lb_pct = f"{int(quantiles_lower[0] * 100)}%"
        
        predictions = []
        lower_two_sided = []
        upper_two_sided = []
        upper_ub = []
        lower_lb = []
        
        history = train_data.copy()
        
        for i in range(0, len(test_data), self.n_forecasts):
            end_idx = min(i + self.n_forecasts, len(test_data))
            chunk_size = end_idx - i
            
            future_chunk = test_data.iloc[i:end_idx].copy()
            if 'y' in future_chunk.columns:
                future_chunk['y'] = None 
            
            future_df = pd.concat([history.tail(model.n_lags), future_chunk]).reset_index(drop=True)
            
            forecast = model.predict(future_df)
            res = forecast.iloc[model.n_lags:].reset_index(drop=True)
            
            for j in range(chunk_size):
                step = j + 1
                col = f'yhat{step}'
                pred = res[col].iloc[j] if col in res.columns else res['yhat1'].iloc[j]
                predictions.append(pred)
                
                # Two-sided lower bound
                l_col_two = f'{col} {lower_two_pct}'
                if l_col_two in res.columns: 
                    lower_two = res[l_col_two].iloc[j]
                elif f'yhat1 {lower_two_pct}' in res.columns: 
                    lower_two = res[f'yhat1 {lower_two_pct}'].iloc[j]
                else: 
                    lower_two = pred * 0.9 
                lower_two_sided.append(lower_two)

                # Two-sided upper bound
                u_col_two = f'{col} {upper_two_pct}'
                if u_col_two in res.columns: 
                    upper_two = res[u_col_two].iloc[j]
                elif f'yhat1 {upper_two_pct}' in res.columns: 
                    upper_two = res[f'yhat1 {upper_two_pct}'].iloc[j]
                else: 
                    upper_two = pred * 1.1 
                upper_two_sided.append(upper_two)
                
                # Upper-bounded (upper only)
                u_col_ub = f'{col} {upper_ub_pct}'
                if u_col_ub in res.columns: 
                    upper_bound = res[u_col_ub].iloc[j]
                elif f'yhat1 {upper_ub_pct}' in res.columns: 
                    upper_bound = res[f'yhat1 {upper_ub_pct}'].iloc[j]
                else: 
                    upper_bound = pred * 1.1 
                upper_ub.append(upper_bound)
                
                # Lower-bounded (lower only)
                l_col_lb = f'{col} {lower_lb_pct}'
                if l_col_lb in res.columns: 
                    lower_bound = res[l_col_lb].iloc[j]
                elif f'yhat1 {lower_lb_pct}' in res.columns: 
                    lower_bound = res[f'yhat1 {lower_lb_pct}'].iloc[j]
                else: 
                    lower_bound = pred * 0.9 
                lower_lb.append(lower_bound)

            next_hist = future_chunk.copy()
            next_hist['y'] = predictions[-chunk_size:]
            history = pd.concat([history, next_hist], ignore_index=True).tail(1000)

        return {
            'predictions': np.array(predictions),
            'lower_two_sided': np.array(lower_two_sided),
            'upper_two_sided': np.array(upper_two_sided),
            'upper_ub': np.array(upper_ub),
            'lower_lb': np.array(lower_lb)
        }

    def assessment(self):
        """Calculates metrics on the test set."""
        results = self.forecast_df.iloc[-self.test_hours:]
        results = results.dropna(subset=['yhat', 'y'])
        
        y = results['y']
        yhat = results['yhat']
        
        rmse = np.sqrt(mean_squared_error(y, yhat))
        mae = mean_absolute_error(y, yhat)
        mape = mean_absolute_percentage_error(y, yhat)
        
        return (rmse, mae, mape)

    def plot_full(self):
        """Plot historical data + future forecast + confidence intervals (no forecast over historical)."""
        df = self.forecast_df.copy()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot ALL historical data (including training period)
        historical = df[df['y'].notna()]
        ax.plot(historical['ds'], historical['y'], 
               label='Historical Data', color='black', alpha=0.7, linewidth=1.5)
        
        # Plot ONLY future forecast (where y is NaN)
        forecast = df[df['yhat'].notna() & df['y'].isna()]
        
        if len(forecast) == 0:
            print("No future forecast available. This plot is designed for forecast_only=True mode.")
            print("For train/test mode, use plot_test() instead.")
            return
        
        ax.plot(forecast['ds'], forecast['yhat'], 
               label=f'{self.model} Forecast', color='#0072B2', linewidth=2)
        
        # Determine bounds based on interval_type
        if self.interval_type == 'two-sided':
            lower_bound = forecast['yhat_lower']
            upper_bound = forecast['yhat_upper']
            label = f'{int(self.interval_width*100)}% Confidence Interval'
        elif self.interval_type == 'upper-bounded':
            lower_bound = forecast['yhat_lower'].min() * 0.8
            upper_bound = forecast['yhat_upper_ub']
            label = f'{int(self.interval_width*100)}% Upper Confidence Bound'
        else:  # lower-bounded
            lower_bound = forecast['yhat_lower_lb']
            upper_bound = forecast['yhat_upper'].max() * 1.2
            label = f'{int(self.interval_width*100)}% Lower Confidence Bound'
        
        # Fill confidence region
        ax.fill_between(
            forecast['ds'],
            lower_bound,
            upper_bound,
            color='#0072B2',
            alpha=0.2,
            label=label
        )
        
        # Plot bound lines
        if self.interval_type == 'upper-bounded':
            ax.plot(forecast['ds'], forecast['yhat_upper_ub'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Upper Bound')
        elif self.interval_type == 'lower-bounded':
            ax.plot(forecast['ds'], forecast['yhat_lower_lb'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Lower Bound')
        
        ax.legend(loc='best')
        plt.ylabel(self.dataframe.columns[0])
        plt.xlabel('Date')
        
        interval_type_str = {
            'two-sided': 'Two-Sided',
            'upper-bounded': 'Upper-Bounded',
            'lower-bounded': 'Lower-Bounded'
        }[self.interval_type]
        
        plt.title(f'Historical Data + Future Forecast ({self.model} - {interval_type_str} CI)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def plot_forecast_only(self):
        """Plot ONLY the forecast + confidence intervals (no historical data)."""
        df = self.forecast_df.copy()
        
        # Get only forecasted rows
        forecast = df[df['yhat'].notna() & df['y'].isna()]
        
        if len(forecast) == 0:
            print("No forecast-only data available. Use forecast_only=True when initializing the model.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
               label=f'{self.model} Forecast', color='#0072B2', linewidth=2)
        
        # Determine bounds based on interval_type
        if self.interval_type == 'two-sided':
            lower_bound = forecast['yhat_lower']
            upper_bound = forecast['yhat_upper']
            label = f'{int(self.interval_width*100)}% Confidence Interval'
        elif self.interval_type == 'upper-bounded':
            lower_bound = forecast['yhat_lower'].min() * 0.8
            upper_bound = forecast['yhat_upper_ub']
            label = f'{int(self.interval_width*100)}% Upper Confidence Bound'
        else:  # lower-bounded
            lower_bound = forecast['yhat_lower_lb']
            upper_bound = forecast['yhat_upper'].max() * 1.2
            label = f'{int(self.interval_width*100)}% Lower Confidence Bound'
        
        # Fill confidence region
        ax.fill_between(
            forecast['ds'],
            lower_bound,
            upper_bound,
            color='#0072B2',
            alpha=0.2,
            label=label
        )
        
        # Plot bound lines
        if self.interval_type == 'upper-bounded':
            ax.plot(forecast['ds'], forecast['yhat_upper_ub'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Upper Bound')
        elif self.interval_type == 'lower-bounded':
            ax.plot(forecast['ds'], forecast['yhat_lower_lb'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Lower Bound')
        elif self.interval_type == 'two-sided':
            # For two-sided, plot both bounds
            ax.plot(forecast['ds'], forecast['yhat_upper'], 
                   color='#0072B2', linestyle='--', alpha=0.5, linewidth=1, label='Upper Bound')
            ax.plot(forecast['ds'], forecast['yhat_lower'], 
                   color='#0072B2', linestyle='--', alpha=0.5, linewidth=1, label='Lower Bound')
        
        ax.legend(loc='best')
        plt.ylabel(self.dataframe.columns[0])
        plt.xlabel('Date')
        
        interval_type_str = {
            'two-sided': 'Two-Sided',
            'upper-bounded': 'Upper-Bounded',
            'lower-bounded': 'Lower-Bounded'
        }[self.interval_type]
        
        plt.title(f'Forecast Only ({self.model} - {interval_type_str} CI)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def plot_test(self):
        """Plots forecast vs actuals on TEST SET (original method for backwards compatibility)."""
        if self.forecast_only:
            print("plot_test() is for train/test split mode. Use plot_full() or plot_forecast_only() for forecast_only mode.")
            return
            
        results = self.forecast_df.iloc[-self.test_hours:]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(results['ds'], results['yhat'], label=f'{self.model} Predictions', color='#0072B2', linewidth=2)
        
        # Determine which bounds to use based on interval_type
        if self.interval_type == 'two-sided':
            lower_bound = results['yhat_lower']
            upper_bound = results['yhat_upper']
            label = f'{int(self.interval_width*100)}% Confidence Interval'
        elif self.interval_type == 'upper-bounded':
            lower_bound = results['yhat_lower'].min() * 0.8
            upper_bound = results['yhat_upper_ub']
            label = f'{int(self.interval_width*100)}% Upper Confidence Bound'
        else:  # lower-bounded
            lower_bound = results['yhat_lower_lb']
            upper_bound = results['yhat_upper'].max() * 1.2
            label = f'{int(self.interval_width*100)}% Lower Confidence Bound'
        
        # Fill the confidence region
        ax.fill_between(
            results['ds'],
            lower_bound,
            upper_bound,
            color='#0072B2',
            alpha=0.2,
            label=label
        )
        
        # Plot the actual relevant bound line
        if self.interval_type == 'upper-bounded':
            ax.plot(results['ds'], results['yhat_upper_ub'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Upper Bound')
        elif self.interval_type == 'lower-bounded':
            ax.plot(results['ds'], results['yhat_lower_lb'], 
                   color='#0072B2', linestyle='--', alpha=0.7, linewidth=1.5, label='Lower Bound')
        
        # Plot actual values
        ax.plot(results['ds'], results['y'], label='True Energy Usage', color='red', alpha=0.8, linewidth=2)
        
        ax.legend(loc='best')
        plt.ylabel(self.dataframe.columns[0])
        plt.xlabel('Date')
        
        interval_type_str = {
            'two-sided': 'Two-Sided',
            'upper-bounded': 'Upper-Bounded',
            'lower-bounded': 'Lower-Bounded'
        }[self.interval_type]
        
        plt.title(f'Forecast vs Actuals ({self.model} - {interval_type_str} CI)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()