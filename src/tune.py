# sktime
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.compose import MultiplexForecaster, AutoEnsembleForecaster, ColumnEnsembleForecaster, DirRecTabularRegressionForecaster, RecursiveTabularRegressionForecaster, DirRecTimeSeriesRegressionForecaster, DirectTabularRegressionForecaster, DirectTimeSeriesRegressionForecaster, EnsembleForecaster, StackingForecaster
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

parser = argparse.ArgumentParser(description='Tuning for ML algorithms. Tune window_length and algorithm parameters.')
parser.add_argument('dataset_name', help='Dataset Name')
parser.add_argument('tune_only_window_length', help='If True, Just find the optimal window_length and not the algorithm parameters. Else tune both.')
parser.add_argument('fast', help='If True, use only one frequency_yearly_period to tune for the expanding window')

args = parser.parse_args()
dataset_name = args.dataset_name
tune_only_window_length = args.tune_only_window_length
fast = args.fast

algorithms = {
    'decision_tree': {
        'estimator': 
            DecisionTreeRegressor(ccp_alpha=0.0,  criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, 
                                  min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, 
                                  min_weight_fraction_leaf=0.0, random_state=42, splitter='best')
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, None],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [2, 3]            
        }     
    },
    'random_forest': {
        'estimator': 
            RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', 
                                  max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                  min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, 
                                  n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, None],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, None],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [1, 2, 3],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],        
        }     
    },    
    'extra_trees': {
        'estimator': 
            ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', 
                                max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, 
                                n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, -1],
            'forecaster__estimator__max_leaf_nodes': [3, 8, 16, 100, -1],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2, 3, 4],
            'forecaster__estimator__min_samples_split': [1, 2, 3],
            'forecaster__estimator__n_estimators': [10, 50, 100],
            'forecaster__estimator__warm_start': [True, False],     
        }     
    },     
    'gradient_boosting': {
        'estimator': 
            GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', 
                                      max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                      min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_iter_no_change=None, 
                                      random_state=42, subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.5, 0.9],
            'forecaster__estimator__ccp_alpha': [0, 0.01, 0.1],
            'forecaster__estimator__max_depth': [2, 3, 5, 10, -1],
            'forecaster__estimator__min_impurity_decrease': [0, 0.01, 0.1],
            'forecaster__estimator__min_samples_leaf': [1, 2],
            'forecaster__estimator__min_samples_split': [2, 3],
            'forecaster__estimator__n_estimators': [10, 100, 200],
            'forecaster__estimator__learning_rate': [0.1, 0.01], 
        }     
    },       
    'adaboost': {
        'estimator': 
            AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=42)
        ,
        'params': {
            'forecaster__estimator__loss': ['linear', 'square', 'exponential'],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],
            'forecaster__estimator__learning_rate': [0.1, 0.05, 0.01],
        }     
    },      
    'lgb_regressor': {
        'estimator': 
            lgbm.sklearn.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, importance_type='split', learning_rate=0.1, max_depth=-1, 
                                       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=31, objective=None, 
                                       random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent='warn', subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
        ,
        'params': {
            'forecaster__estimator__max_depth': [1, 2, 3, 4, 5, 10, -1],
            'forecaster__estimator__num_leaves': [2, 3, 10, 20, 31, 100],
            'forecaster__estimator__min_child_samples': [5, 10, 20, 50],
            'forecaster__estimator__min_child_weight': [0.001, 0.005],
            'forecaster__estimator__n_estimators': [10, 50, 100, 200],
        }     
    },   
    'knn': {
        'estimator': 
            KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5, p=2, weights='uniform')
        ,
        'params': {
            'forecaster__estimator__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'forecaster__estimator__p': [1, 2, 3],
        }     
    },    
    'passive_aggressive': {
        'estimator': 
            PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False, epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive', max_iter=1000, 
                                       n_iter_no_change=5, random_state=42, shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__C': [0.1, 0.25, 0.5, 0.75, 1],
            'forecaster__estimator__early_stopping': [True, False],
            'forecaster__estimator__epsilon': [0.01, 0.05, 0.1, 0.2],
            'forecaster__estimator__max_iter': [500, 1000, 2000],
            'forecaster__estimator__n_iter_no_change': [1, 2, 3, 4, 5, 7],
            'forecaster__estimator__validation_fraction': [0.1, 0.2],
            'forecaster__estimator__tol': [None, 0.001, 0.002],
        }     
    },       
    'huber': {
        'estimator': 
            HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.00005, 0.0001, 0.0005, 0.001],
            'forecaster__estimator__early_epsilon': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [50, 100, 200, 500],
            'forecaster__estimator__tol': [1e-05, 1e-06, 5e-05, 5e-04],
            'forecaster__estimator__warm_start': [True, False],
        }     
    },     
    'bayesian_ridge': {
        'estimator': 
            BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, 
                          lambda_2=1e-06, lambda_init=None, n_iter=300, normalize=False, tol=0.001, verbose=False)
        ,
        'params': {
            'forecaster__estimator__alpha_1': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__alpha_2': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__lambda_1': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__lambda_2': [1e-05, 5e-05, 1e-06, 5e-06],
            'forecaster__estimator__compute_score': [True, False],
            'forecaster__estimator__n_iter': [100, 200, 300, 400],
            'forecaster__estimator__tol': [0.0005, 0.001, 0.005, 0.01, 0.05],
        }     
    },        
    'lasso_lars': {
        'estimator': 
            LassoLars(alpha=1.0, copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, max_iter=500, 
                      normalize=True, positive=False, precompute='auto', random_state=42, verbose=False)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        }     
    },        
    'lars': {
        'estimator': 
            Lars(copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, n_nonzero_coefs=500, 
                 normalize=True, precompute='auto', random_state=42, verbose=False)
        ,
        'params': {
            'forecaster__estimator__n_nonzero_coefs': [1, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        }     
    },       
    'elastic_net': {
        'estimator': 
            ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, 
                       precompute=False, random_state=42, selection='cyclic', tol=0.0001, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
            'forecaster__estimator__warm_start': [True, False],
        }     
    },        
    'ridge': {
        'estimator': 
            Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=42, solver='auto', tol=0.001)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
        }     
    },     
    'lasso': {
        'estimator': 
            Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, 
                  random_state=42, selection='cyclic', tol=0.0001, warm_start=False)
        ,
        'params': {
            'forecaster__estimator__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
            'forecaster__estimator__max_iter': [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'forecaster__estimator__tol': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005],
            'forecaster__estimator__warm_start': [True, False],
        }     
    },        
}


# define forecastin horizon
fh = 1

# Read Data
data, frequency_yearly_period, freq_sktime = read_file(dataset_name, data_path='../data/')
preprocess = False

# metric
mase = MeanAbsoluteScaledError(sp=frequency_yearly_period)

# ONLY FOR SKTIME
# keep datetime as a column for plots
data['datetime'] = data.index
data.index = pd.PeriodIndex(data.index, freq=freq_sktime)

for target in data.drop(columns=['datetime']):
    print('#'*70, target, '#'*70)

    # split data
    train, test, valid, train_without_valid, train_test_split_date, train_valid_split_date = train_valid_test_split(dataset_name, data)

    if fast == 'True':
        initial_window = train[:train.shape[0]-frequency_yearly_period].shape[0]
    else:
        initial_window = train_without_valid.shape[0]

    # expanding window to fit test data
    cv = ExpandingWindowSplitter(step_length=1, fh=fh, initial_window=initial_window)
    min_max_scaler = TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))

    for algorithm_name, value in algorithms.items():
        print(algorithm_name)

        estimator = DirectTabularRegressionForecaster(estimator=value['estimator'])

        pipe = TransformedTargetForecaster(steps=[
            # ("detrender", Detrender()),
            # ("deseasonalizer", Differencer(lags=1)),
            ("minmaxscaler", min_max_scaler),
            ("forecaster", estimator),
        ])

        if tune_only_window_length == 'True':
            param_grid = {"forecaster__window_length": [int((i/2)*frequency_yearly_period) for i in range(2,9)]}
        else:
            param_grid = value['params']
            param_grid['forecaster__window_length'] = [int((i/2)*frequency_yearly_period) for i in range(2,9)]

        gscv = ForecastingGridSearchCV(
            forecaster = pipe, 
            strategy = "refit", 
            cv = cv, 
            param_grid = param_grid,
            scoring = mase,
            n_jobs = -1,
            verbose = 10
        )

        gscv.fit(train[target], fh=fh)

        if tune_only_window_length == 'True':
            pd.to_pickle(gscv.best_forecaster_, f'../results/tuned_models/just_window/{dataset_name}/{target}.{algorithm_name}.pkl')
        else:
            pd.to_pickle(gscv.best_forecaster_, f'../results/tuned_models/window_and_algorithm/{dataset_name}/{target}.{algorithm_name}.pkl')