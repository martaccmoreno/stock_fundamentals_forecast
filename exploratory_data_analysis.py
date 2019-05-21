import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost.sklearn import XGBRegressor

# To begin with, we will focus on a single smaller data set to get a sense of the data.
snapshot = pd.read_csv('data/latest-snapshot-quarterly.csv')
print(snapshot.info(), '\n')

snapshot.hist(bins=50, figsize=(20, 15))
plt.show()
# "Current assets can be converted into cash is less than one year."
# All companies have slightly more total assets (average ~2K) than current assets
# Values appear to be in $USD
# SEC ID seems to be the variable by which we can unite this info with the info in companies-names-industries.csv


# We would like to predict: dilluted earnings per share
print(snapshot["EarningsPerShareDiluted"].describe(), '\n')
# Worst case scenario: each share is a huge loss (-60$)!
# Mean is higher than the median, suggesting we have some high-performers to compensate for the "failures"
# We would like to find those high-performers.

# Let's see how our target variable correlates to the other economical values (exclude date and ID)
corr_matrix = snapshot.drop(["SEC ID", "Report date"], axis=1).corr()
print(corr_matrix["EarningsPerShareDiluted"].sort_values(ascending=False), '\n')
# Non-neglible but weak positive correlation with ComprehensiveIncomeNetOfTax
# "Includes deferred gains or losses on qualifying hedges, unrealized holding gains or losses on available-for-sale
# securities, minimum pension liability, and cumulative translation adjustment."
# So, possibly, the more "potential" value a company has, the higher its stock value.

y = snapshot["EarningsPerShareDiluted"]
X = snapshot.drop(["EarningsPerShareDiluted", "Report date", "SEC ID"], axis=1)

# Dealing with missing values
print("Are there missing values in X?", any(pd.isna(X).any()))
print("Are there missing values in y?", pd.isna(y).any())
features = X.columns[X.isna().any()].tolist()
print("List of features:\n", ','.join(features))
print("Do all feature columns have NANs?", len(features) == X.shape[1])
# All feature columns have NANs
# Also need to remove NANs from target -- or do we?

median_imputer = SimpleImputer(strategy="median")
X = median_imputer.fit_transform(X)
# Fill strategy for NANs appears to have little effect on ML
median_target_val = y.median()
y = y.fillna(median_target_val)
y = y.to_numpy()

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Feature Scaling also appears to not explain the weak correlation found in ML (little effect)

# Feature importance
forest = ExtraTreesRegressor(n_estimators=250, random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
print("Feature importances:\n", importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature rankings:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
# None of the features appears to be particularly informative with such wide ranges of standard deviation.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nLinear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Poor correlation, error overfits a lot

print("\nRandom Forest")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Overfits correlation a great deal, and still overfits error

print("\nLasso Regression")
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# No overfitting in correlation (none found in fact) but severely overfits error

print("\nRidge Regression")
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Similar to Lasso

print("\nElastic Net")
en = ElasticNet(alpha=0.1)
en.fit(X_train, y_train)
y_train_pred = en.predict(X_train)
y_test_pred = en.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Similar to Lasso and Ridge

print("\nAdaBoost")
ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Similar to Random Forests with slightly less overfitting

print("\nGradient Boosting")
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
y_train_pred = gb.predict(X_train)
y_test_pred = gb.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Similar to AdaBoost

print("\nXGBoost")
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_train_pred = xgb.predict(X_train)
y_test_pred = xgb.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R^2:", r2_score(y_train, y_train_pred))
print("Test R^2:", r2_score(y_test, y_test_pred))
# Similar to the other boosting methods

# We probably need more data to improve predictions.

# Inspect structure for quaterly data per indicator:
assets = pd.read_csv('data/metrics/Assets-quarterly.csv')
assets = assets.drop("SEC ID", axis=1)
print('\n', assets.head())
print(assets.to_numpy())
# Each row is a company; each column is a quarter, from older to newest
# Load all quaterly reports for features, and merge, and load target also
print('\nQuarters', list(assets.columns))