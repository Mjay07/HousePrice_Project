# /model/model_building.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load the dataset
# Assuming train.csv is in the same directory
df = pd.read_csv('train.csv')

# 2. Feature Selection
# Allowed features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, BedroomAbvGr, FullBath, YearBuilt, Neighborhood
# Selected 6 features + Target (SalePrice)
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt', 'SalePrice']
df = df[selected_features]

# 3. Data Preprocessing
# a. Handling missing values
# Fill numeric missing values with the median (robust to outliers)
df.fillna(df.median(), inplace=True)

# b. Feature Selection (Separating X and y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Implement Algorithm: Random Forest Regressor
# No feature scaling is strictly required for Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 6. Save the trained model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as house_price_model.pkl")

# 7. Verification: Reload model
loaded_model = joblib.load('house_price_model.pkl')
print("Model loaded successfully for verification.")