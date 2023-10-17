import numpy as np
import statsmodels.api as sm
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

# Add a constant term (intercept) to the predictor variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Get the summary of the OLS model
ols_summary = model.summary()
print(ols_summary)
