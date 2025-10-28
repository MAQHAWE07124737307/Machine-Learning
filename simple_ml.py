# simple_ml.py
# A simple machine learning example using scikit-learn
# This script trains a linear regression model to predict house prices.

from sklearn.linear_model import LinearRegression
import numpy as np

# Input features (square footage)
X = np.array([[500], [750], [1000], [1250], [1500]])
# Output labels (price in thousands)
y = np.array([150, 200, 250, 300, 350])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new house
new_house = np.array([[1200]])
predicted_price = model.predict(new_house)

print(f"Predicted price for a 1200 sq. ft house: ${predicted_price[0]*1000:.2f}")
