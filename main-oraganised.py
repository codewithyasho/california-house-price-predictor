# 📦 Imports
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

# 📥 Load dataset
df = pd.read_csv('housing.csv')

# 🔄 Stratified split based on income_cat
df["income_cat"] = pd.cut(df["median_income"],
                          bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                          labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = df.loc[test_index].drop("income_cat", axis=1)

# 📄 Work with training set only
housing = strat_train_set.copy()
housing_features = housing.drop("median_house_value", axis=1)
housing_labels = housing["median_house_value"].copy()

# 🔍 Separate numerical and categorical features
num_attr = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attr = ["ocean_proximity"]

# 🔧 Pipelines
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attr),
    ('cat', cat_pipeline, cat_attr)
])

# 🚀 Prepare the data
housing_prepared = full_pipeline.fit_transform(housing_features)

# ==========================
# 🎯 RANDOM FOREST REGRESSOR
# ==========================
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
rf_grid.fit(housing_prepared, housing_labels)

print("Best RF Params:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_

# Train & Evaluate
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prepared, housing_labels)
rf_preds = rf_reg.predict(housing_prepared)
rf_cv_rmse = -cross_val_score(
    rf_reg, housing_prepared, housing_labels,
    scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1
)
print("\nRandom Forest RMSE:", np.mean(rf_cv_rmse))
print("Random Forest R² Score:", r2_score(housing_labels, rf_preds)*100, "%")

# ==========================
# 🎯 XGBOOST REGRESSOR
# ==========================
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}

xgb_grid = GridSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
    param_grid=xgb_params,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
xgb_grid.fit(housing_prepared, housing_labels)

print("Best XGBoost Params:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_

# Train & Evaluate
xgb_reg = XGBRegressor()
xgb_reg.fit(housing_prepared, housing_labels)
xgb_preds = xgb_reg.predict(housing_prepared)
xgb_cv_rmse = -cross_val_score(
    xgb_reg, housing_prepared, housing_labels,
    scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1
)
print("\nXGBoost RMSE:", np.mean(xgb_cv_rmse))
print("XGBoost R² Score:", r2_score(housing_labels, xgb_preds)*100, "%")

# Final comparison
print("\nFinal Comparison:")
print("Random Forest - R²:", r2_score(housing_labels,
      best_rf.predict(housing_prepared)) * 100, "%")
print("XGBoost - R²:", r2_score(housing_labels,
      best_xgb.predict(housing_prepared)) * 100, "%")

# Save the best model and preprocessing pipeline
joblib.dump(best_xgb, "best_xgboost_model.pkl")
joblib.dump(full_pipeline, "preprocessing_pipeline.pkl")
