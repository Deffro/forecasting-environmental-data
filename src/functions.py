import time
import numpy as np
import pandas as pd

import statsmodels.api as sm
import pymannkendall as mk
from scipy.stats import kruskal
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sklearn.preprocessing import MinMaxScaler

### Read Functions ###

def convert_to_datetime_and_set_index(data, dataset_name):
    # convert to datetime and remove timezone
    if dataset_name == 'Solcast':
        data['datetime'] = pd.to_datetime(data['PeriodEnd']).dt.tz_localize(None)
        data = data.set_index('datetime').asfreq('1H')
    elif dataset_name == 'ERA5':
        data['date'] = data['date'].apply(lambda x: x.split('-')[0]+x.split('-')[1]+'19'+x.split('-')[2])
        data['datetime']= pd.to_datetime(data['date'], format='%d%m%Y')
        data = data.set_index('datetime').asfreq('1D')
    elif dataset_name == 'ORAS5':
        data['datetime']= pd.to_datetime(data['date'], format='%Y%m')
        data = data.set_index('datetime').asfreq('1MS')    
    elif dataset_name == 'AQPiraeus':
        data['datetime'] = pd.to_datetime(data['datetime']).dt.tz_localize(None)
        data = data.set_index('datetime').asfreq('1H')
    elif dataset_name == 'Jena':
        data['datetime'] = pd.to_datetime(data['Date Time'])
        data = data.set_index('datetime')
        print('Resampling...')
        data = data.resample('1H').last()
        data = data.reset_index()
        data = data.set_index('datetime').asfreq('1H')
    elif dataset_name == 'Satellite' or dataset_name == 'NOAA':
        data['datetime'] = pd.to_datetime(data['date'], format='%Y%m%d')
        data = data.set_index('datetime').asfreq('1D')
    elif dataset_name == 'CO2':
        data['datetime'] = pd.to_datetime(data['Year'], format='%Y')
        data = data.set_index('datetime').asfreq('YS')
    
    # print('Conversion to datetime was successfull. Frequency is', data.index.freq)
    return data

def handle_missing_values(data, dataset_name):
    if dataset_name == 'AQPiraeus' or dataset_name == 'Jena':
        data = data.fillna(-999)
        for variable in data.columns:
            new_col = []
            for i, row in enumerate(data[variable]):
                if i > 24*365:
                    if row == -999:
                        if data[variable].iloc[i-24*365] != -999:
                            new_col.append(data[variable].iloc[i-24*365]+999)
                        elif data[variable].iloc[i-2*24*365] != -999:
                            new_col.append(data[variable].iloc[i-2*24*365]+999)
                        else:
                            new_col.append(np.nan)
                    else:
                        new_col.append(0)
                else:
                    new_col.append(0)

            data[variable] = data[variable] + new_col
            data[variable] = data[variable].replace(-999, np.nan)
            data[variable] = data[variable].interpolate(method ='linear', limit_direction ='forward')
        data = data.dropna()        
    return data

def remove_columns(data, dataset_name):
    if dataset_name == 'Solcast':
        # SnowWater was removed because it is almost always 0
        data = data.drop(columns=['PeriodEnd', 'PeriodStart', 'Period', 'SnowWater'])
        print(f"Columns 'PeriodEnd', 'PeriodStart', 'Period', 'SnowWater' were removed from {dataset_name}.")
    elif dataset_name == 'ERA5':
        data = data.drop(columns=['date'])
        print(f"Columns 'date' were removed from {dataset_name}.")
    elif dataset_name == 'ORAS5':
        # so26chgt, so28chgt were removed because tey have almost always the same values
        # so14chgt had errors in the data
        data = data.drop(columns=['date', 'so14chgt', 'so26chgt', 'so28chgt'])
        print(f"Columns 'date', 'so14chgt', 'so26chgt', 'so28chgt' were removed from {dataset_name}.")  
    elif dataset_name == 'Jena':
        data = data.drop(columns=['Date Time'])
        print(f"Column 'Date Time' was removed from {dataset_name}.") 
    elif dataset_name == 'Satellite':
        data = data.drop(columns=['date'])
        print(f"Column 'date' was removed from {dataset_name}.")            
    elif dataset_name == 'NOAA':
        data = data.drop(columns=['Date'])
        print(f"Column 'Date' was removed from {dataset_name}.")      
    elif dataset_name == 'CO2':
        data = data.drop(columns=['Year'])
        data = data.rename({'Total': 'CO2'}, axis=1)
        print(f"Column 'Year' was removed from {dataset_name}.")    
    return data
        
def get_stats(df, sort_by='Different Values', sort_how=False, exclude=[]):
        columns = [c for c in df.columns.values if c not in exclude]
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Data types
        data_types = df.dtypes
        
        # Different Values
        other_values = pd.DataFrame(columns=['Different Values','Most Common','% of Most Common','Skewness',
                                            'Kurtosis','Mean','Min','25% quantile','Median','75% quantile','Max'])
        for c in columns:
            if (df[c].dtype != 'object'):
                other_values = other_values.append({
                    'Name' : c,
                    'Different Values' : df[c].value_counts().count(),
                    'Most Common' : df[c].value_counts().idxmax(),
                    '% of Most Common' : 100*df[c].value_counts().max() / df[c].value_counts().sum(),
                    'Skewness' : df[c].skew(),
                    'Kurtosis' : df[c].kurt(),
                    'Mean' : df[c].mean(),
                    'Min' : df[c].min(),
                    '25% quantile' : df[c].quantile(0.25),
                    'Median' : df[c].median(),
                    '75% quantile' : df[c].quantile(0.75),
                    'Max' : df[c].max(),
                                                    }, ignore_index=True)
            else:
                other_values = other_values.append({
                    'Name' : c,
                    'Different Values' : df[c].value_counts().count(),
                    'Most Common' : df[c].value_counts().idxmax(),
                    '% of Most Common' : 100*df[c].value_counts().max() / df[c].value_counts().sum()
                                                    }, ignore_index=True)                
        other_values = other_values.set_index('Name')
        
        
        # Make a table with the results
        mis_val_table = pd.concat([data_types, other_values, mis_val, mis_val_percent], axis=1, sort=False)       
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0:'Type',
                   'Different Values':'Different Values',
                   'Most Common':'Most Common',
                   '% of Most Common':'% of Most Common',
                   'Skewness':'Skewness', 
                   'Kurtosis':'Kurtosis', 
                   1:'Missing Values', 
                   2:'% of Missing Values',
                   'Pearson Corr' : 'Pearson Corr',
                   'Mean' : 'Mean',
                   'Min' : 'Min',
                   '25% quantile' : '25% quantile',
                   'Median' : 'Median',
                   '75% quantile' : '75% quantile',
                   'Max' : 'Max',
        })
        
        # Sort the table 
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values(sort_by, ascending=sort_how)
        df = mis_val_table_ren_columns
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[0]) + " columns. (" +
        str(df[df['Type']=='object'].shape[0])+" categorical and "+str(df[df['Type']!='object'].shape[0])+" numerical).\n"
        "There are " + str(df[df['Missing Values']>0].shape[0]) +" columns that have missing values " +
        str(df[(df['Type']=='object')&(df['Different Values']<5)].shape[0]) + 
        " object type columns have less than 5 different values and you can consider one-hot encoding, while " + 
        str(df[(df['Type']=='object')&(df['Different Values']>=5)].shape[0]) +
        " have more than 5 colums and you can consider label encoding.\n" +
        str(df[df['Skewness']>1].shape[0]) + " columns are highly positively skewed (skewness>1), while " +
        str(df[df['Skewness']<-1].shape[0]) + " columns are highly negatively skewed (skewness<-1).\n" +
        str(df[(df['Skewness']>-0.5)&(df['Kurtosis']<0.5)].shape[0]) + " columns are symmetrical (-0.5<skewness<0.5).\n" +
        str(df[df['Kurtosis']>3].shape[0]) + " columns have high kurtosis (kurtosis>3) and should be check for outliers, while " + 
        str(df[df['Kurtosis']<3].shape[0]) + " columns have low kurtosis (kurtosis<3). "
        )        
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
def read_file(dataset_name, data_path='../../data/'):
    accepted_dataset_names = ['Solcast', 'ERA5', 'ORAS5', 'AQPiraeus', 'Jena', 'Satellite', 'NOAA', 'CO2']
    if dataset_name not in accepted_dataset_names:
        raise ValueError(f'dataset_name accepted values are {accepted_dataset_names}')
        
    dataset_names = {'Solcast': '40.554572_25.084034_Solcast_PT60M.csv',
                     'ERA5': 'ERA5 daily data on pressure levels from 1950 to 1978.csv',
                     'ORAS5': 'ORAS5 global ocean reanalysis monthly data from 1958 to present.csv',
                     'AQPiraeus': 'Air Quality in Piraeus GR0030A Station.csv',
                     'Jena': 'jena_climate_2009_2016.csv',
                     'Satellite': 'Sea level daily gridded data from satellite observations for the global ocean from 1993 to present.csv',
                     'NOAA': 'FCE_FlamingoRS_ClimDB_data.csv',
                     'CO2': 'Global CO2 emissions dataset.csv'}
    data = pd.read_csv(f'{data_path}{dataset_names[dataset_name]}')
    
    data = convert_to_datetime_and_set_index(data, dataset_name)
    data = remove_columns(data, dataset_name)
    data = handle_missing_values(data, dataset_name)
    
    if str(data.index.freq) == '<Day>':
        frequency_yearly_period = 365
    elif str(data.index.freq) == '<MonthBegin>':
        frequency_yearly_period = 12
    elif str(data.index.freq) == '<Hour>':
        frequency_yearly_period = 24*365
    elif str(data.index.freq) == '<YearBegin: month=1>':
        frequency_yearly_period = 10
    return data, frequency_yearly_period   


### Pre-processing Tests Functions ###

def check_stationarity(series, variable_name):
    is_stationary = False
    result = adfuller(series, autolag='AIC')
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    
    for key,val in result[4].items():
        out['critical value ({})'.format(key)]=val
        
    # print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        # print("Strong evidence against the null hypothesis")
        # print("Reject the null hypothesis")
        # print("Data has no unit root and is stationary")
        print(f"{variable_name} is stationary (p-value={result[1]}).")
        is_stationary = True
    else:
        # print("Weak evidence against the null hypothesis")
        # print("Fail to reject the null hypothesis")
        # print("Data has a unit root and is non-stationary")
        print(f"{variable_name} is NOT stationary (p-value={result[1]}).")
    return is_stationary

def seasonality_test(series, variable_name, seasonal_period):
    seasonal = False
    idx = np.arange(len(series.index)) % seasonal_period
    H_statistic, p_value = kruskal(series, idx)
    # print(H_statistic, p_value)
    if p_value <= 0.05:
        print(f'{variable_name} has seasonality.')
        seasonal = True
    else:
        print(f'{variable_name} has NO seasonality.')
    return seasonal

def trend_test(series, variable_name):
    has_trend = False
    trend_test_df = pd.DataFrame(columns=['method', 'has_trend', 'p_value'])

    # Original Mann-Kendall test is a nonparametric test, which does not consider serial correlation or seasonal effects.
    trend_test_df = trend_test_df.append({'method': 'original_test', 'has_trend': mk.original_test(series).h, 'p_value': mk.original_test(series).p}, ignore_index=True)

    # This modified MK test proposed by Hamed and Rao (1998) to address serial autocorrelation issues.
    # They suggested a variance correction approach to improve trend analysis.
    # User can consider first n significant lag by insert lag number in this function.
    # By default, it considered all significant lags.
    trend_test_df = trend_test_df.append({'method': 'hamed_rao_modification_test', 
                                          'has_trend': mk.hamed_rao_modification_test(series).h, 
                                          'p_value': mk.hamed_rao_modification_test(series).p}, ignore_index=True)

    # This is also a variance correction method for considered serial autocorrelation proposed by Yue, S., & Wang, C. Y. (2004).
    # User can also set their desired significant n lags for the calculation.
    trend_test_df = trend_test_df.append({'method': 'yue_wang_modification_test', 
                                          'has_trend': mk.yue_wang_modification_test(series).h, 
                                          'p_value': mk.yue_wang_modification_test(series).p}, ignore_index=True)

    # This test suggested by Yue and Wang (2002) to using Pre-Whitening the time series before the application of trend test.
    trend_test_df = trend_test_df.append({'method': 'pre_whitening_modification_test', 
                                          'has_trend': mk.pre_whitening_modification_test(series).h, 
                                          'p_value': mk.pre_whitening_modification_test(series).p}, ignore_index=True)

    # This test also proposed by Yue and Wang (2002) to remove trend component and then Pre-Whitening the time series before application of trend test.
    trend_test_df = trend_test_df.append({'method': 'trend_free_pre_whitening_modification_test', 
                                          'has_trend': mk.trend_free_pre_whitening_modification_test(series).h, 
                                          'p_value': mk.trend_free_pre_whitening_modification_test(series).p}, ignore_index=True)

    # For seasonal time series data, Hirsch, R.M., Slack, J.R. and Smith, R.A. (1982) proposed this test to calculate the seasonal trend.
    trend_test_df = trend_test_df.append({'method': 'seasonal_test', 
                                          'has_trend': mk.seasonal_test(series).h, 
                                          'p_value': mk.seasonal_test(series).p}, ignore_index=True)

    # This method proposed by Hipel (1994) used, when time series significantly correlated with the preceding one or more months/seasons.
    trend_test_df = trend_test_df.append({'method': 'correlated_seasonal_test', 
                                          'has_trend': mk.correlated_seasonal_test(series).h, 
                                          'p_value': mk.correlated_seasonal_test(series).p}, ignore_index=True)

    if trend_test_df['has_trend'].sum()/trend_test_df.shape[0] > 0.5:
        print(f"{variable_name} has trend. {trend_test_df['has_trend'].sum()}/{trend_test_df.shape[0]} metrics agree.")
        has_trend = True
    else:
        print(f"{variable_name} has NO trend. {trend_test_df['has_trend'].sum()}/{trend_test_df.shape[0]} metrics found a trend.")
    return has_trend


### Pre-processing Transformations ###

def remove_trend_by_decomposition(series, model='additive'):
    decomposition = sm.tsa.seasonal_decompose(series, model=model)
    detrended = series.values - decomposition.trend
    return detrended

def remove_trend_by_subtracting_the_line_of_best_fit(series):
    detrended = signal.detrend(series.values)
    detrended = pd.Series(data=detrended, index=series.index)
    return detrended

def apply_log_transformation(series):
    transformed = np.log(series)
    return transformed

def apply_sqrt_transformation(series):
    series = series
    transformed = np.sqrt(series)
    return transformed

def apply_box_cot_transformation(series):
    # series should be positive
    transformer = BoxCoxTransformer()
    transformed = transformer.fit_transform(series)
    # to produce the initial series == transformer.inverse_transform(transformed)
    return transformed, transformer

def apply_min_max_scaling(series):
    scaler = MinMaxScaler(feature_range=(1, 2))
    scaler.fit(series.values.reshape(-1, 1))
    scaled_data = scaler.transform(series.values.reshape(-1, 1))
    scaled_data = pd.DataFrame(data=scaled_data, index=series.index)[0]
    return scaled_data, scaler

class Differencing():
    def __init__(self, shift=1):
        self.shift = shift
    
    def fit_transform(self, series):
        self.series = series
        shifted_series = self.series.shift(periods=self.shift)
        deseasonalized = self.series - shifted_series
        return deseasonalized
    
    def inverse_transform(self, deseasonalized, test=pd.Series([], dtype=np.float64), y_pred=pd.Series([], dtype=np.float64), fh=1):
        '''
        deseasonalized: the deseasonalized data that were created with fit_transform()
        test:           the test dataset that we want to forecast
                        if empty, the trained data are inverse transformed only
        y_pred:         the prediction(s) for the test dataset. these will be in the same scale as the deseasonalized data that was used to train the model
                        if empty, the trained data are inverse transformed only
        fh:             the forecasting horizon
        '''
        series = pd.concat([self.series, test[:fh]])
        shifted_series = series.shift(periods=self.shift).values
        
        deseasonalized = pd.concat([deseasonalized, y_pred])
        inverted_series = deseasonalized + shifted_series
        return inverted_series
    

class MovingAverageSubtraction():
    def __init__(self, window=12):
        self.window = window    
        
    def fit_transform(self, series):
        self.series = series
        rolling_values = self.series.rolling(window = self.window).mean()
        detrended = self.series - rolling_values
        return detrended
    
    def inverse_transform(self, detrended, test=pd.Series([], dtype=np.float64), y_pred=pd.Series([], dtype=np.float64), fh=1):
        '''
        deseasonalized: the detrended data that were created with fit_transform()
        test:           the test dataset that we want to forecast
                        if empty, the trained data are inverse transformed only
        y_pred:         the prediction(s) for the test dataset. these will be in the same scale as the detrended data that was used to train the model
                        if empty, the trained data are inverse transformed only
        fh:             the forecasting horizon
        '''        
        series = pd.concat([self.series, test[:fh]])
        rolling_values = series.rolling(window = self.window).mean().values
        
        detrended = pd.concat([detrended, y_pred])
        inverted_series = detrended + rolling_values
        return inverted_series

def remove_seasonality_by_decomposition(series, model='additive'):
    if model == 'additive':
        decomposition = sm.tsa.seasonal_decompose(series, model=model)
        deseasonalized = series.values - decomposition.seasonal
    else:
        if series.min() > 0:
            decomposition = sm.tsa.seasonal_decompose(series, model=model)
            deseasonalized = series.values - decomposition.seasonal
        else:
            print('here')
            series = series + np.abs(series.min()) + 0.000000001
            decomposition = sm.tsa.seasonal_decompose(series, model=model)
            deseasonalized = series.values - decomposition.seasonal
    return deseasonalized

# do not use
def find_if_series_is_additive_or_multiplicative(series, variable_name, window=12):
    detrended = apply_sqrt_transformation(series)
    detrended = remove_trend_by_subtracting_moving_window(detrended, window=window)
    detrended = detrended/detrended.max()
    s = detrended.reset_index()
    s['timeframe'] = s['datetime'].dt.year

    std_of_seasonal = s.groupby('timeframe')[variable_name].std().std()
    print(s.groupby('timeframe')[variable_name].max() - s.groupby('timeframe')[variable_name].min())
    if std_of_seasonal >= 1:
        print(f'{variable_name}: std_of_seasonal is {std_of_seasonal}. Maybe consider multiplicative.')
    else:
        print(f'{variable_name}: std_of_seasonal is {std_of_seasonal}. Maybe consider additive.')