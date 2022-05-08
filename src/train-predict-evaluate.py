# sktime
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA, ARIMA
from sktime.forecasting.compose import (
    MultiplexForecaster, 
    AutoEnsembleForecaster, 
    ColumnEnsembleForecaster, 
    TransformedTargetForecaster,
    DirRecTabularRegressionForecaster, 
    DirRecTimeSeriesRegressionForecaster, 
    DirectTabularRegressionForecaster, 
    DirectTimeSeriesRegressionForecaster, 
    EnsembleForecaster, 
    StackingForecaster
)
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.compose import ColumnwiseTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sklearn.preprocessing import StandardScaler

from sktime.performance_metrics.forecasting import (
    MeanSquaredError, 
    MeanAbsoluteScaledError, 
    mean_absolute_percentage_error, 
    MeanAbsoluteError
)
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter, 
    ForecastingGridSearchCV
)
from sktime.transformations.series.detrend import (
    Deseasonalizer, 
    Detrender
)
from sktime.transformations.series.difference import Differencer

from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    Lasso, 
    ElasticNet, 
    Lars, 
    LassoLars, 
    BayesianRidge, 
    HuberRegressor, 
    PassiveAggressiveRegressor, 
    OrthogonalMatchingPursuit
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor, 
    AdaBoostRegressor
)
import lightgbm as lgbm

import warnings
warnings.filterwarnings("ignore")

from functions import *
import argparse


parser = argparse.ArgumentParser(description='Pipeline for sktime Classical and ML algorithms. No preprocessing, No tuning.')
parser.add_argument('dataset_name', help='Dataset Name')
parser.add_argument('sample', help='valid or test')
parser.add_argument('fh', help='forecasting horizon. int')

args = parser.parse_args()
dataset_name = args.dataset_name
sample = args.sample
fh = int(args.fh)
fh=[i+1 for i in range (fh)]

# Read Data
#dataset_name = 'Solcast'
data, seasonal_period, freq_sktime = read_file(dataset_name, data_path='../data/')
preprocess = False

forecasters = {
    'naive_forecaster': {
        'estimator': 
            NaiveForecaster(sp=1, strategy='last', window_length=None)
        ,  
    },    
    'naive_forecaster_seasonal': {
        'estimator': 
            NaiveForecaster(sp=12, strategy='last', window_length=None)
        ,  
    },
    'polynomial_trend_forecaster': {
        'estimator': 
            PolynomialTrendForecaster(degree=1, regressor=None, with_intercept=True)
        ,  
    },    
    'arima': {
        'estimator': 
            AutoARIMA(n_jobs=-1)
        ,  
    },    
    'exponential_smoothing': {
        'estimator': 
            ExponentialSmoothing(damped_trend=False, initial_level=None, initial_seasonal=None, initial_trend=None, initialization_method='estimated', seasonal=None, sp=seasonal_period, trend='add', use_boxcox=None)
        ,  
    },    
    'ets': {
        'estimator': 
            AutoETS(n_jobs=-1)
        ,  
    },      
    'theta_forecaster': {
        'estimator': 
            ThetaForecaster(deseasonalize=True, initial_level=None, sp=seasonal_period)
        ,  
    },     
    'decision_tree': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=DecisionTreeRegressor(ccp_alpha=0.0,  criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, 
                                  min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, 
                                  min_weight_fraction_leaf=0.0, random_state=42, splitter='best'))
        ,  
    },
    'random_forest': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', 
                                  max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                  min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, 
                                  n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False))
        ,   
    },    
    'extra_trees': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', 
                                max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, 
                                n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False))
        ,    
    },     
    'gradient_boosting': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', 
                                      max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                      min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_iter_no_change=None, 
                                      random_state=42, subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=False))
        ,   
    },       
    'adaboost': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=42))
        ,  
    },      
    'lgb_regressor': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=lgbm.sklearn.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, importance_type='split', learning_rate=0.1, max_depth=-1, 
                                       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=31, objective=None, 
                                       random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent='warn', subsample=1.0, subsample_for_bin=200000, subsample_freq=0))
        ,    
    },   
    'knn': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=-1, n_neighbors=5, p=2, weights='uniform'))
        ,    
    },    
    'passive_aggressive': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=PassiveAggressiveRegressor(C=1.0, average=False, early_stopping=False, epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive', max_iter=1000, 
                                       n_iter_no_change=5, random_state=42, shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0, warm_start=False))
        ,    
    },       
    'huber': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False))
        ,    
    },     
    'bayesian_ridge': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, 
                          lambda_2=1e-06, lambda_init=None, n_iter=300, normalize=False, tol=0.001, verbose=False))
        ,    
    },        
    'lasso_lars': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=LassoLars(alpha=1.0, copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, max_iter=500, 
                      normalize=True, positive=False, precompute='auto', random_state=42, verbose=False))
        ,   
    },        
    'lars': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=Lars(copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True, jitter=None, n_nonzero_coefs=500, 
                 normalize=True, precompute='auto', random_state=42, verbose=False))
        ,     
    },       
    'elastic_net': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, 
                       precompute=False, random_state=42, selection='cyclic', tol=0.0001, warm_start=False))
        ,    
    },        
    'ridge': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=42, solver='auto', tol=0.001))
        ,    
    },     
    'lasso': {
        'estimator': 
            DirectTabularRegressionForecaster(estimator=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, 
                  random_state=42, selection='cyclic', tol=0.0001, warm_start=False))
        ,    
    },        
}
ML_forecasters = ['decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting', 'adaboost', 'lgb_regressor', 
                  'knn', 'passive_aggressive', 'huber', 'bayesian_ridge' , 'lasso_lars', 'lars', 'elastic_net', 
                  'ridge', 'lasso']

# ONLY FOR SKTIME
# keep datetime as a column for plots
data['datetime'] = data.index
data.index = pd.PeriodIndex(data.index, freq=freq_sktime)

if sample == 'valid':

    for target in data.drop(columns=['datetime']).columns:
    
        # check if file already exists
        target_ = target
        target = target.replace('/', '')
        if os.path.isfile(f'../results/predictions/valid/no_preprocess/{dataset_name}/{target_}.csv') is False:
    
            print('#'*70, target, '#'*70)

            # split data
            train, test, valid, train_without_valid, train_test_split_date, train_valid_split_date = train_valid_test_split(dataset_name, data)
            initial_window = train_without_valid.shape[0]

            # save prediction in a df. a column per method
            predictions_valid = pd.DataFrame()
            predictions_valid['datetime'] = valid['datetime']
            predictions_valid['true_values'] = valid[target_]

            # define metrics
            rmse = MeanSquaredError(square_root=True)
            mase = MeanAbsoluteScaledError(sp=seasonal_period)
            smape = mean_absolute_percentage_error
            mae = MeanAbsoluteError()
            # keep track of scores, per method and fh
            scores_expanding = pd.DataFrame()

            for forecaster_name, value in forecasters.items():
                forecaster = value['estimator']
                # if ML forecaster
                if forecaster_name in ML_forecasters:
                    tuned_pipe = pd.read_pickle(f'../results/tuned_models/window_and_algorithm/{dataset_name}/{target}.{forecaster_name}.pkl')
                    print("Window Length:", tuned_pipe.get_params()['forecaster'].get_params()['window_length'])
                else:
                    min_max_scaler = TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))
                    tuned_pipe = TransformedTargetForecaster(steps=[
                        # ("detrender", Detrender()),
                        # ("deseasonalizer", Differencer(lags=1)),
                        ("minmaxscaler", min_max_scaler),
                        ("forecaster", forecaster),
                    ])
                    
                print('='*40, tuned_pipe, '='*40)

                df = evaluate_sktime(tuned_pipe, train[target_], fh=fh, initial_window=initial_window, 
                                     metrics=['MAE', 'RMSE', 'sMAPE', 'MASE'], seasonal_period=seasonal_period)

                # save predictions in a df
                for ii in fh:
                    forecasts = []
                    for v in df['y_pred'].values:
                        try:
                            forecasts.append(v.values[ii-1])
                        except IndexError:
                            pass
                    if ii < np.max(fh):
                        for i in range(len(predictions_valid)-len(forecasts)-ii+1):
                            forecasts.append(np.nan)
                    for i in range(ii-1):
                        forecasts = np.insert(forecasts, 0, np.nan)
                    predictions_valid[f'fh={ii} {forecaster}'] = forecasts

                total_runtime = np.sum(df['fit_time']) + np.sum(df['pred_time'])

                # evaluate forecasting horizons on the same number of samples
                p = predictions_valid.dropna()

                scores_expanding = scores_expanding.append({
                    'Method': str(forecaster).replace('\n', '').replace(' ', ''), 
                    'Forecasting Horizon': fh, 
                    'Preprocess': preprocess,
                    'Runtime': total_runtime,      

                }, ignore_index=True)

                for i in fh:
                    maes, rmses, smapes, mases = [], [], [], []
                    for ii, (j, row) in enumerate(p.iterrows()):
                        maes.append(mae([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        rmses.append(rmse([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        smapes.append(smape([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        mases.append(mase([row['true_values']], [row[f'fh={i} {forecaster}']], y_train=df.iloc[max(fh)-1+ii]['y_train']))


                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MAE'] = np.mean(maes)   
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} RMSE'] = np.mean(rmses) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} sMAPE'] = np.mean(smapes) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MASE'] = np.mean(mases) 

                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MAE std'] = np.std(maes)   
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} RMSE std'] = np.std(rmses) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} sMAPE std'] = np.std(smapes) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MASE std'] = np.std(mases) 

            predictions_valid.to_csv(f'../results/predictions/valid/no_preprocess/{dataset_name}/{target_}.csv', index=False)
            scores_expanding.to_csv(f'../results/scores/valid/no_preprocess/{dataset_name}/{target_}.csv', index=False)

elif sample == 'test':
    for target in data.drop(columns=['datetime']).columns:
    
        # check if file already exists
        target_ = target
        target = target.replace('/', '_')
        if os.path.isfile(f'../results/predictions/test/no_preprocess/{dataset_name}/{target_}.csv') is False:    
    
            print('#'*70, target, '#'*70)

            # split data
            train, test, valid, train_without_valid, train_test_split_date, train_valid_split_date = train_valid_test_split(dataset_name, data)
            initial_window = train.shape[0]

            # save prediction in a df. a column per method
            predictions_test = pd.DataFrame()
            predictions_test['datetime'] = test['datetime']
            predictions_test['true_values'] = test[target_]

            # define metrics
            rmse = MeanSquaredError(square_root=True)
            mase = MeanAbsoluteScaledError(sp=seasonal_period)
            smape = mean_absolute_percentage_error
            mae = MeanAbsoluteError()
            # keep track of scores, per method and fh
            scores_expanding = pd.DataFrame()

            for forecaster_name, value in forecasters.items():
                forecaster = value['estimator']
                # if ML forecaster
                if forecaster_name in ML_forecasters:
                    tuned_pipe = pd.read_pickle(f'../results/tuned_models/window_and_algorithm/{dataset_name}/{target}.{forecaster_name}.pkl')
                    print("Window Length:", tuned_pipe.get_params()['forecaster'].get_params()['window_length'])
                else:
                    min_max_scaler = TabularToSeriesAdaptor(MinMaxScaler(feature_range=(1, 2)))
                    tuned_pipe = TransformedTargetForecaster(steps=[
                        # ("detrender", Detrender()),
                        # ("deseasonalizer", Differencer(lags=1)),
                        ("minmaxscaler", min_max_scaler),
                        ("forecaster", forecaster),
                    ])
                    
                print('='*40, tuned_pipe, '='*40)

                df = evaluate_sktime(tuned_pipe, data[target_], fh=fh, initial_window=initial_window, 
                                     metrics=['MAE', 'RMSE', 'sMAPE', 'MASE'], seasonal_period=seasonal_period)

                # save predictions in a df
                for ii in fh:
                    forecasts = []
                    for v in df['y_pred'].values:
                        try:
                            forecasts.append(v.values[ii-1])
                        except IndexError:
                            pass
                    if ii < np.max(fh):
                        for i in range(len(predictions_test)-len(forecasts)-ii+1):
                            forecasts.append(np.nan)
                    for i in range(ii-1):
                        forecasts = np.insert(forecasts, 0, np.nan)
                    predictions_test[f'fh={ii} {forecaster}'] = forecasts

                total_runtime = np.sum(df['fit_time']) + np.sum(df['pred_time'])

                # evaluate forecasting horizons on the same number of samples
                p = predictions_test.dropna()

                scores_expanding = scores_expanding.append({
                    'Method': str(forecaster).replace('\n', '').replace(' ', ''),  
                    'Forecasting Horizon': fh, 
                    'Preprocess': preprocess,
                    'Runtime': total_runtime,      

                }, ignore_index=True)

                for i in fh:
                    maes, rmses, smapes, mases = [], [], [], []
                    for ii, (j, row) in enumerate(p.iterrows()):
                        maes.append(mae([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        rmses.append(rmse([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        smapes.append(smape([row['true_values']], [row[f'fh={i} {forecaster}']]))
                        mases.append(mase([row['true_values']], [row[f'fh={i} {forecaster}']], y_train=df.iloc[max(fh)-1+ii]['y_train']))


                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MAE'] = np.mean(maes)   
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} RMSE'] = np.mean(rmses) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} sMAPE'] = np.mean(smapes) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MASE'] = np.mean(mases) 

                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MAE std'] = np.std(maes)   
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} RMSE std'] = np.std(rmses) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} sMAPE std'] = np.std(smapes) 
                    scores_expanding.at[scores_expanding.index.max(), f'fh={i} MASE std'] = np.std(mases) 

            predictions_test.to_csv(f'../results/predictions/test/no_preprocess/{dataset_name}/{target_}.csv', index=False)
            scores_expanding.to_csv(f'../results/scores/test/no_preprocess/{dataset_name}/{target_}.csv', index=False)
