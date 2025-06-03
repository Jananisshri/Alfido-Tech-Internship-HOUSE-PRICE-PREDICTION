import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
dataset = pd.read_csv('data_Task2.csv')

# Display dataset shape and first 5 rows
print("Dataset Shape:", dataset.shape)
print("First 5 rows:\n", dataset.head(5))

# Select input features and target output
# Columns: 3rd (bedrooms), 4th (bathrooms), 5th (sqft_living), 7th (floors), 13th (yr_built)
X = dataset.iloc[:, [2, 3, 4, 6, 12]]
Y = dataset.iloc[:, 1]  # 2nd column: price

print("\nSelected Features (X):\n", X.head())
print("\nTarget Variable (Y):\n", Y.head())

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Take user input for prediction
feature_names = ['Bedrooms', 'Bathrooms', 'Sqft Living', 'Floors', 'Year Built']
user_input = []

for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Predict the price
predicted_price = model.predict([user_input])
print("\nPredicted House Price: ", round(predicted_price[0], 2))

# Save the trained model to a file
pickle.dump(model, open('model.pkl', 'wb'))
