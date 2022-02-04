# sktime
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.compose import MultiplexForecaster, AutoEnsembleForecaster, ColumnEnsembleForecaster, DirRecTabularRegressionForecaster, DirRecTimeSeriesRegressionForecaster, DirectTabularRegressionForecaster, DirectTimeSeriesRegressionForecaster, EnsembleForecaster, StackingForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.bats import BATS
from sktime.forecasting.croston import Croston
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.compose import ColumnwiseTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import StandardScaler

from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsoluteScaledError, mean_absolute_percentage_error, MeanAbsoluteError
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.difference import Differencer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
import lightgbm as lgbm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
from functions import *
import argparse

parser = argparse.ArgumentParser(description='Pipeline for sktime Classical and ML algorithms. No preprocessing, No tuning.')
parser.add_argument('dataset_name', help='Dataset Name')

args = parser.parse_args()
dataset_name = args.dataset_name

# define forecastin horizon
fh=1

# Read Data
#dataset_name = 'Solcast'
data, frequency_yearly_period, freq_sktime = read_file(dataset_name, data_path='../data/')
preprocess = False

# for multi step consider DirectTabularRegressionForecaster, RecursiveTabularRegressionForecaster, DirRecTabularRegressionForecaster
forecasters = [
    NaiveForecaster(sp=1, strategy='last', window_length=None),
    NaiveForecaster(sp=12, strategy='last', window_length=None),
    PolynomialTrendForecaster(degree=1, regressor=None, with_intercept=True),
    AutoARIMA(n_jobs=-1),
    ExponentialSmoothing(damped_trend=False, initial_level=None, initial_seasonal=None, initial_trend=None, initialization_method='estimated', seasonal=None, sp=frequency_yearly_period, trend='add', use_boxcox=None),
    AutoETS(n_jobs=-1),
    ThetaForecaster(deseasonalize=True, initial_level=None, sp=frequency_yearly_period),
    DirectTabularRegressionForecaster(estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False, positive=False)),
    DirectTabularRegressionForecaster(estimator=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=42, selection='cyclic', tol=0.0001, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=42, solver='auto', tol=0.001)),
    DirectTabularRegressionForecaster(estimator=ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=42, selection='cyclic', tol=0.0001, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=Lars(copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, n_nonzero_coefs=500, normalize=True, precompute='auto', random_state=42, verbose=False)),
    DirectTabularRegressionForecaster(estimator=LassoLars(alpha=1.0, copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, max_iter=500, normalize=True, positive=False, precompute='auto', random_state=42, verbose=False)),
    DirectTabularRegressionForecaster(estimator=BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300, normalize=False, tol=0.001, verbose=False)),
    DirectTabularRegressionForecaster(estimator=HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False, epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive', max_iter=1000, n_iter_no_change=5, random_state=42, shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=OrthogonalMatchingPursuit(fit_intercept=True, n_nonzero_coefs=None, normalize=True, precompute='auto', tol=None)),
    DirectTabularRegressionForecaster(estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5, p=2, weights='uniform')),
    DirectTabularRegressionForecaster(estimator=DecisionTreeRegressor(ccp_alpha=0.0,  criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=42, splitter='best')),
    DirectTabularRegressionForecaster(estimator=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_iter_no_change=None, random_state=42, subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=False)),
    DirectTabularRegressionForecaster(estimator=AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=42)),
    DirectTabularRegressionForecaster(estimator=lgbm.sklearn.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, importance_type='split', learning_rate=0.1, max_depth=-1, min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=31, objective=None, random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent='warn', subsample=1.0, subsample_for_bin=200000, subsample_freq=0))
]

# ONLY FOR SKTIME
# keep datetime as a column for plots
data['datetime'] = data.index
data.index = pd.PeriodIndex(data.index, freq=freq_sktime)

for target in data.columns:
    print('#'*70, target, '#'*70)
    
    # split data
    train, test, valid, train_without_valid, train_test_split_date, train_valid_split_date = train_valid_test_split(dataset_name, data)

    # save prediction in a df. a column per method
    predictions_valid = pd.DataFrame()
    predictions_valid['datetime'] = valid['datetime']
    predictions_valid['true_values'] = valid[target]

    # expanding window to fit test data
    cv = ExpandingWindowSplitter(step_length=1, fh=fh, initial_window=train_without_valid.shape[0])

    # define metrics
    rmse = MeanSquaredError(square_root=True)
    #mase = MeanAbsoluteScaledError(sp=frequency_yearly_period)
    #mase2 = MASE()
    smape = mean_absolute_percentage_error
    mae = MeanAbsoluteError()
    # keep track of scores, per method and fh
    scores_expanding = pd.DataFrame()

    for forecaster in forecasters:
        print('='*40, forecaster, '='*40)
        min_max_scaler = TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))
        pipe = TransformedTargetForecaster(steps=[
            # ("detrender", Detrender()),
            # ("deseasonalizer", Differencer(lags=1)),
            ("minmaxscaler", min_max_scaler),
            ("forecaster", forecaster),
        ])

        df = evaluate_sktime(forecaster=pipe, y=train[target], cv=cv, return_data=True, metrics=['MAE', 'RMSE', 'sMAPE', 'MASE'], 
                             preprocess=preprocess, frequency_yearly_period=frequency_yearly_period)

        # save predictions in a df
        forecasts = [i.values[0] for i in df['y_pred'].values]
        for i in range(fh-1):
            forecasts = np.insert(forecasts, 0, np.nan)
        predictions_valid[f'{forecaster}'] = forecasts

        total_runtime = np.sum(df['fit_time']) + np.sum(df['pred_time'])
        scores_expanding = scores_expanding.append({
            'Method': str(forecaster), 
            'Forecasting Horizon': fh, 
            'Preprocess': preprocess,
            'Runtime': total_runtime, 
            'MAE': df['MAE'].mean(),
            'RMSE': df['RMSE'].mean(),
            'sMAPE': df['sMAPE'].mean(),
            'MASE': df['MASE'].mean(),
            'MAE std': df['MAE'].std(),
            'RMSE std': df['RMSE'].std(),
            'sMAPE std': df['sMAPE'].std(),
            'MASE std': df['MASE'].std(),        

        }, ignore_index=True)

    predictions_valid.to_csv(f'../results/predictions/no_preprocess/{dataset_name}/fh.{fh}_{target}.csv', index=False)
    scores_expanding.to_csv(f'../results/scores/no_preprocess/{dataset_name}/fh.{fh}_{target}.csv', index=False)
    
    
    
    
