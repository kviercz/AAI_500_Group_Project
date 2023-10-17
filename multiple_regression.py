import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset as Pandas DataFrame
df = pd.read_csv('winequality/winequality-red.csv', sep=';')

# Define independent vars
X = df[['volatile acidity', 'residual sugar', 'alcohol', 'pH']]
# Define dependent vars
y = df['quality']

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create linear Regression model to fit training data
model = LinearRegression()
model.fit(X_train, y_train)

# Use test data as prediction
y_pred = model.predict(X_test)

# Evaluate the performance using mean squared error and r-squared test
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)
