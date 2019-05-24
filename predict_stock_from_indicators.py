import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

import os
from glob import glob
from collections import OrderedDict


# We exclude the latest snapshots and share (stock) indicators that would highly correlate with our target.
# Include only quarterly reports.
feature_filenames = [
    filename for filename in
    [os.path.basename(file) for file in glob(os.path.join('data', 'metrics', '*.csv'))]
    if ('latest' not in filename.lower() and 'share' not in filename.lower())
       and ('quarterly' in filename.lower())
]

# Quarters for each indicator are our features, current company stock value is what we want to predict
stock_price = pd.read_csv("data/metrics/EarningsPerShareDiluted-quarterly.csv", index_col="SEC ID")
# We want to group earnings by quarter or company. We're grouping them by company so quarters are our features.
median_company_stock_price = stock_price.median(axis=1)

y = median_company_stock_price.to_numpy()
y[np.isnan(y)] = np.nanmean(y)

indicators = pd.DataFrame()
for filename in feature_filenames:
    feature_filepath = os.path.join('data', 'metrics', filename)
    feature_df = pd.read_csv(feature_filepath, index_col="SEC ID")
    indicators = pd.concat((indicators, feature_df), axis=1)


processing_pipeline = [('imputer', SimpleImputer(strategy='mean')), ('scaler', MinMaxScaler())]
pipeline = Pipeline(processing_pipeline)
X = pipeline.fit_transform(indicators)

# remove the few extreme stock value outliers present in our data
z = abs(stats.zscore(y))
outlier_idx = np.where(z > np.percentile(z, 75))
y = np.delete(y, outlier_idx, axis=0)
X = np.delete(X, outlier_idx, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42)

print("\nGrid Search: XGBoost")
xgb = XGBRegressor(n_jobs=2, random_state=42)
param_grid = {'n_estimators': [500, 750], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'gamma': [0, 0.1]}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=2,
                           iid=False, verbose=0)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

scores_dict = OrderedDict({
    "Train R2": r2_score(y_train, y_train_pred), "Test R2": r2_score(y_test, y_test_pred),
    "Train MAE": median_absolute_error(y_train, y_train_pred), "Test MAE": median_absolute_error(y_test, y_test_pred),
    "Train MSE": mean_squared_error(y_train, y_train_pred), "Test MSE": mean_squared_error(y_test, y_test_pred)
})

for key, value in scores_dict.items():
        print(f"{key}: {value:.3f}")



