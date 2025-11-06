import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib



data = pd.read_csv(r"C:\Users\kajal\OneDrive\Desktop\Business\DA\DATASET\crop_yield.csv")

print("✅ Data Loaded Successfully!\n")
print(data.head())

# Basic info
print("\n--- Data Info ---")
print(data.info())

print("\n--- Missing Values ---")
print(data.isnull().sum())

# Convert categorical columns into numeric using dummy encoding
data_encoded = pd.get_dummies(data, columns=['Crop', 'Season', 'State'], drop_first=True)

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_encoded.corr(), cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Selection (input vs output)
X = data_encoded[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data_encoded['Yield']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model Training Complete!")

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\n R² Score: {r2:.3f}")
print(f" Mean Squared Error: {mse:.3f}")

# Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# Save Model
joblib.dump(model, 'crop_yield_model.pkl')
print("\n Model saved as crop_yield_model.pkl")
