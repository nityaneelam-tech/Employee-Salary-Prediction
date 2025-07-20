import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# You can use your own dataset or download one from Kaggle
# Example dataset should have columns like 'Experience', 'Education', 'Salary'
data = pd.read_csv("employee_data.csv")

# Display first few rows
print("Sample Data:")
print(data.head())

# Convert categorical columns if any (e.g., Education) using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Split into input features (X) and target variable (y)
X = data.drop("Salary", axis=1)
y = data["Salary"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict salary for a new input
# Example: 5 years experience, Master’s degree (adjust depending on your dataset columns)
# new_data = pd.DataFrame([[5, 1]], columns=["Experience", "Education_Master"])
# salary_pred = model.predict(new_data)
# print(f"\nPredicted Salary for input: ₹{salary_pred[0]:,.2f}")
