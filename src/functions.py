import numpy as np
import pandas as pd

import statsmodels.api as sm
import pymannkendall as mk
from scipy.stats import kruskal
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from sktime.transformations.series.boxcox import BoxCoxTransformer

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
    
    # print('Conversion to datetime was successfull. Frequency is', data.index.freq)
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
    accepted_dataset_names = ['Solcast', 'ERA5', 'ORAS5', 'AQThess']
    if dataset_name not in accepted_dataset_names:
        raise ValueError(f'dataset_name accepted values are {accepted_dataset_names}')
        
    dataset_names = {'Solcast': '40.554572_25.084034_Solcast_PT60M.csv',
                     'ERA5': 'ERA5 daily data on pressure levels from 1950 to 1978.csv',
                     'ORAS5': 'ORAS5 global ocean reanalysis monthly data from 1958 to present.csv',
                     'AQThess': 'Air Quality in Thessaloniki GR0018A Station.csv'}
    data = pd.read_csv(f'{data_path}{dataset_names[dataset_name]}')
    
    data = convert_to_datetime_and_set_index(data, dataset_name)
    data = remove_columns(data, dataset_name)
    
    if str(data.index.freq) == '<Day>':
        frequency_yearly_period = 365
    elif str(data.index.freq) == '<MonthBegin>':
        frequency_yearly_period = 12
    elif str(data.index.freq) == '<Hour>':
        frequency_yearly_period = 24*365
    
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
    transformed = np.log(series).replace([np.inf, -np.inf], 0)
    return transformed

def apply_sqrt_transformation(series):
    series = series + np.abs(series.min())
    transformed = np.sqrt(series)
    return transformed

def apply_box_cot_transformation(series):
    transformer = BoxCoxTransformer()
    transformed = transformer.fit_transform(series+np.abs(series)+0.00000001)
    # to produce the initial series == transformer.inverse_transform(transformed)-np.abs(series_init)-0.00000001
    return transformed, transformer

def remove_trend_by_subtracting_moving_window(series, window=12):
    rolling_values = series.rolling(window = window).mean()
    detrended = series - rolling_values
    # to produce the initial series == detrended+rolling_values
    return detrended, rolling_values

def remove_seasonality_by_differencing(series, shift=1):
    shifted_series = series.shift(periods=shift)
    deseasonalized = series - shifted_series
    return deseasonalized, shifted_series

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