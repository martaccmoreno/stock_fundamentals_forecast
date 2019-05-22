import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

import csv
from collections import OrderedDict

features = ['Assets', 'AssetsCurrent', 'CashAndCashEquivalentsAtCarryingValue', 'ComprehensiveIncomeNetOfTax',
            'Goodwill', 'Liabilities', 'LiabilitiesCurrent', 'NetCashProvidedByUsedInFinancingActivities',
            'NetCashProvidedByUsedInInvestingActivities', 'NetCashProvidedByUsedInOperatingActivities',
            'OperatingIncomeLoss', 'PropertyPlantAndEquipmentNet', 'Revenues']

# split capitalized string into lowercased words seperated by underscores:
derp = [
    ''.join(['_'+x.lower() if x.isupper() and idx > 0 else x.lower() for idx, x in enumerate(feature)])
    for feature in features
]

# Quarters for each indicator are our features, company stock value is what we want to predict

stock_price = pd.read_csv("data/metrics/EarningsPerShareDiluted-quarterly.csv", index_col="SEC ID")
# We want to group earnings by quarter or company. We're grouping them by company so quarters are our features.
mean_company_stock_price = stock_price.mean(axis=1)

y = mean_company_stock_price.to_numpy()
y[np.isnan(y)] = np.nanmean(y)

indicators = pd.DataFrame()
for feature in features:
    feature_file_name = "data/metrics/" + feature + "-quarterly.csv"
    feature_df = pd.read_csv(feature_file_name, index_col="SEC ID")
    indicators = pd.concat((indicators, feature_df), axis=1)


processing_pipeline = [('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]
pipeline = Pipeline(processing_pipeline)
X = pipeline.fit_transform(indicators)

# feature_sel_pipeline = [('percentile', SelectPercentile(f_regression, percentile=10))]
# pipeline = Pipeline(feature_sel_pipeline)
# X = pipeline.fit_transform(X, y)

# remove the few extreme stock value outliers present in our data
z = abs(stats.zscore(y))
print(np.percentile(z, 99))  # 99% of the y samples have z score below 0.032
outlier_idx = np.where(z > 0.032)
y = np.delete(y, outlier_idx, axis=0)
X = np.delete(X, outlier_idx, axis=0)

print("Y shape", y.shape)
print("X shape", X.shape, "\n")

print("min y", np.min(y))
print("max y", np.max(y))
print("median y", np.median(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=42)

# print("\nRandom Forest")
# rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
# rf.fit(X_train, y_train)
# y_train_pred = rf.predict(X_train)
# y_test_pred = rf.predict(X_test)

print("\nGrid Search: XGBoost")
xgb = XGBRegressor(n_jobs=2, random_state=42)
param_grid = {'n_estimators': [250, 500, 750], 'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [3, 5, 6],
              'gamma': [0, 0.1]}
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
#
#     scores_list.append(scores_dict)
#
# csv_columns = scores_list[0].keys()
# with open('results_mean.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=csv_columns)
#     writer.writeheader()
#     for scores in scores_list:
#         writer.writerow(scores)


# results_zero always has train / test correlations being sorta the inverse and completely unlike;
# plus there's underfitting with test error being lower than training errror




