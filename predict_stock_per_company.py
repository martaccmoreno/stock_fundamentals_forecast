import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error

# Attempt to predict how valuable a company0s stock will be in a given quarter after learning
# from previous quarters
# Note: not all companies share the same indicators! And some have less quarters than others.
# Some companies don't have EarningsPerShareDiluted for the latest evaluated quarter
# The biggest problem with these models is the lack of data points (only 20-so quarters)


company_ids = ["1750", "1800", "2034", "2098", "2186", "2488", "2491", "2969", "3116", "3146"]

for company_id in company_ids:
    company = pd.read_csv("data/companies/{}-quarterly.csv".format(company_id), index_col=0)

    X = company.drop("EarningsPerShareDiluted").to_numpy()[:, :-1].transpose()
    y = company.loc["EarningsPerShareDiluted"].to_numpy()[:-1]

    latest_snapshot_X = company.drop("EarningsPerShareDiluted").to_numpy()[:, -1].transpose().reshape(1, -1)
    latest_snapshot_y = company.loc["EarningsPerShareDiluted"].to_numpy()[-1]

    print("min y", np.min(y))
    print("max y", np.max(y))
    print("median y", np.median(y))

    y[np.isnan(y)] = 0
    processing_pipeline = [('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', Normalizer())]
    pipeline = Pipeline(processing_pipeline)
    X = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\nRandom Forest")
    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=4)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    print("Train MSE:", mean_squared_error(y_train, y_train_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))

    print("\nGrid Search: XGBoost")
    xgb = XGBRegressor(n_jobs=4, random_state=42)
    param_grid = {'n_estimators': [100, 500, 1000], 'learning_rate': [0.001, 0.1, 1.0], 'max_depth': [3, 5],
                  'gamma': [0, 0.1, 1.0]}
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=4, iid=False)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    y_train_pred = grid_search.best_estimator_.predict(X_train)
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    print("Train MSE:", mean_squared_error(y_train, y_train_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))

    plt.scatter(y_test, y_test_pred)
    plt.show()
    plt.close()

    print("Results for latest snapshot:")
    print("Predicted price:", grid_search.best_estimator_.predict(latest_snapshot_X)[0])
    print("Real price:", latest_snapshot_y)


# failed to predict 1750 - underprediction (-0.5); 2969 (-8.5)
# failed to predict 2098 - overprediction (+0.7)
# Good prediction for 1800, 2034, 2186, 3116
# OK prediction for 2488, 2491, 3146
# In short, this model is a bit of a crapshoot but not completely useless as a starting point