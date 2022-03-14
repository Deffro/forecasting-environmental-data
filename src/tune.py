# sktime
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.compose import MultiplexForecaster, AutoEnsembleForecaster, ColumnEnsembleForecaster, DirRecTabularRegressionForecaster, RecursiveTabularRegressionForecaster, DirRecTimeSeriesRegressionForecaster, DirectTabularRegressionForecaster, DirectTimeSeriesRegressionForecaster, EnsembleForecaster, StackingForecaster
from sktime.forecasting.ets import AutoETS
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

parser = argparse.ArgumentParser(description='Tuning for ML algorithms. Tune window_length and algorithm parameters.')
parser.add_argument('dataset_name', help='Dataset Name')
parser.add_argument('tune_only_window_length', help='If True, Just find the optimal window_length and not the algorithm parameters. Else tune both.')
parser.add_argument('fast', help='If True, use only one seasonal_period to tune for the expanding window')

args = parser.parse_args()
dataset_name = args.dataset_name
tune_only_window_length = args.tune_only_window_length
fast = args.fast

algorithms = {
    'decision_tree': {
        'estimator': 
            DecisionTreeRegressor(random_state=42)
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, None],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [2, 3]            
        },
	'n_jobs': -3     
    },
    'random_forest': {
        'estimator': 
            RandomForestRegressor(n_jobs=-10, random_state=42)
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, None],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [2, 3],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],        
        },
	'n_jobs': 1       
    },    
    'extra_trees': {
        'estimator': 
            ExtraTreesRegressor(n_jobs=-10, random_state=42)
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, None],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [2, 3],
            'forecaster__estimator__n_estimators': [10, 50, 100],
            'forecaster__estimator__warm_start': [True, False],     
        },
	'n_jobs': 1        
    },     
    'gradient_boosting': {
        'estimator': 
            GradientBoostingRegressor(random_state=42)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.5, 0.9],
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [2, 3, 5, 10, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2],
            'forecaster__estimator__min_samples_split': [2, 3],
            'forecaster__estimator__n_estimators': [10, 100, 200],
            'forecaster__estimator__learning_rate': [0.1, 0.01], 
        },
	'n_jobs': -3        
    },       
    'adaboost': {
        'estimator': 
            AdaBoostRegressor(random_state=42)
        ,
        'params': {
            'forecaster__estimator__loss': ['linear', 'square', 'exponential'],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],
            'forecaster__estimator__learning_rate': [0.1, 0.05, 0.01],
        },
	'n_jobs': -3        
    },      
    'lgb_regressor': {
        'estimator': 
            lgbm.sklearn.LGBMRegressor(random_state=42)
        ,
        'params': {
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, -1],
            'forecaster__estimator__num_leaves': [2, 3, 10, 20, 31, 100],
            'forecaster__estimator__min_child_samples': [5, 10, 20, 50],
            'forecaster__estimator__min_child_weight': [0.001, 0.005],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],
        },
	'n_jobs': -3        
    },   
    'knn': {
        'estimator': 
            KNeighborsRegressor(n_jobs=-1)
        ,
        'params': {
            'forecaster__estimator__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'forecaster__estimator__p': [1, 2, 3],
        },
	'n_jobs': 1        
    },    
    'passive_aggressive': {
        'estimator': 
            PassiveAggressiveRegressor(random_state=42)
        ,
        'params': {
            'forecaster__estimator__C': [0.1, 0.25, 0.5, 0.75, 1],
            'forecaster__estimator__early_stopping': [True, False],
            'forecaster__estimator__epsilon': [0.01, 0.05, 0.1, 0.2],
            'forecaster__estimator__max_iter': [500, 1000, 2000],
            'forecaster__estimator__n_iter_no_change': [1, 2, 3, 4, 5, 7],
            'forecaster__estimator__validation_fraction': [0.1, 0.2],
            'forecaster__estimator__tol': [None, 0.001, 0.002],
        },
	'n_jobs': -3        
    },       
    'huber': {
        'estimator': 
            HuberRegressor()
        ,
        'params': {
            'forecaster__estimator__alpha': [0.00005, 0.0001, 0.0005, 0.001],
            'forecaster__estimator__epsilon': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [50, 100, 200, 500],
            'forecaster__estimator__tol': [1e-05, 1e-06, 5e-05, 5e-04],
            'forecaster__estimator__warm_start': [True, False],
        },
	'n_jobs': -3        
    },     
    'bayesian_ridge': {
        'estimator': 
            BayesianRidge()
        ,
        'params': {
            'forecaster__estimator__alpha_1': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__alpha_2': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__lambda_1': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__lambda_2': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__compute_score': [True, False],
            'forecaster__estimator__n_iter': [100, 200, 300, 400],
            'forecaster__estimator__tol': [0.0005, 0.001, 0.005, 0.01, 0.05],
        },
	'n_jobs': -3        
    },        
    'lasso_lars': {
        'estimator': 
            LassoLars(random_state=42)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        },
	'n_jobs': -3        
    },        
    'lars': {
        'estimator': 
            Lars(random_state=42)
        ,
        'params': {
            'forecaster__estimator__n_nonzero_coefs': [1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        },
	'n_jobs': -3        
    },       
    'elastic_net': {
        'estimator': 
            ElasticNet(random_state=42)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
            'forecaster__estimator__warm_start': [True, False],
        },
	'n_jobs': -3        
    },        
    'ridge': {
        'estimator': 
            Ridge(random_state=42)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        },
	'n_jobs': -3        
    },     
    'lasso': {
        'estimator': 
            Lasso(random_state=42)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
            'forecaster__estimator__warm_start': [True, False],
        },
	'n_jobs': -3        
    },        
}


# define forecastin horizon
fh = 1

# Read Data
data, seasonal_period, freq_sktime = read_file(dataset_name, data_path='../data/')
preprocess = False

# metric
mase = MeanAbsoluteScaledError(sp=seasonal_period)

# ONLY FOR SKTIME
# keep datetime as a column for plots
data['datetime'] = data.index
data.index = pd.PeriodIndex(data.index, freq=freq_sktime)

for target in data.drop(columns=['datetime']):
    print('#'*70, target, '#'*70)

    # split data
    train, test, valid, train_without_valid, train_test_split_date, train_valid_split_date = train_valid_test_split(dataset_name, data)

    if fast == 'True':
        initial_window = train[:train.shape[0]-seasonal_period].shape[0]
    else:
        initial_window = train_without_valid.shape[0]

    # expanding window to fit test data
    cv = ExpandingWindowSplitter(step_length=1, fh=fh, initial_window=initial_window)
    min_max_scaler = TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))

    for algorithm_name, value in algorithms.items():
        print(algorithm_name)
        forecaster = value['estimator']

        estimator = DirectTabularRegressionForecaster(estimator=forecaster)
        
        if tune_only_window_length == 'True':
            window_tuned_model = pd.read_pickle(f'../results/tuned_models/just_window/{dataset_name}/{target}.{algorithm_name}.pkl')
            window_length = window_tuned_model.get_params()['forecaster'].get_params()['window_length']
            estimator = estimator.set_params(window_length=window_length)

        pipe = TransformedTargetForecaster(steps=[
            # ("detrender", Detrender()),
            # ("deseasonalizer", Differencer(lags=1)),
            ("minmaxscaler", min_max_scaler),
            ("forecaster", estimator),
        ])

        if seasonal_period == 1:
            window_size = 7
        elif seasonal_period == 12 or seasonal_period == 24:
            window_size = seasonal_period

        if tune_only_window_length == 'True':
            param_grid = {"forecaster__window_length": [int((i/2)*window_size) for i in range(2,9)]}
        else:
            param_grid = value['params']

        gscv = ForecastingGridSearchCV(
            forecaster = pipe, 
            strategy = "refit", 
            cv = cv, 
            param_grid = param_grid,
            scoring = mase,
            n_jobs = value['n_jobs'],
            verbose = 10
        )

        gscv.fit(train[target], fh=fh)

        if tune_only_window_length == 'True':
            pd.to_pickle(gscv.best_forecaster_, f'../results/tuned_models/just_window/{dataset_name}/{target}.{algorithm_name}.pkl')
        else:
            pd.to_pickle(gscv.best_forecaster_, f'../results/tuned_models/window_and_algorithm/{dataset_name}/{target}.{algorithm_name}.pkl')
