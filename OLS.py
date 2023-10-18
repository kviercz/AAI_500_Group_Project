import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import itertools

def OLS(df, predictor_vars):
    # Define independent vars
    X = df[predictor_vars]
    # Define dependent vars
    y = df['quality']

    # Add a constant term (intercept)
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    # Get the summary of the OLS model
    ols_summary = model.summary()
    return ols_summary
    # print(ols_summary)

def multiple_regression(df, predictor_vars):
    # Define independent vars
    X = df[predictor_vars]
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
    return mse, r2
    # print("Mean Squared Error:", mse)
    # print("R-squared (R2) Score:", r2)

def best_fit(df):
    # Define your dependent variable (target) and list of potential predictors
    dependent_variable = 'quality'
    all_predictors = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', \
                      'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # Initialize variables to store the best results
    best_r_squared = -1 
    best_aic = float('inf')
    best_subset = []
    best_adjusted_r_squared = -1


    # Loop through all possible predictor subsets
    for subset_size in range(1, len(all_predictors) + 1):
        # Generate all combinations of predictor variables
        predictor_combinations = itertools.combinations(all_predictors, subset_size)

        for predictors in predictor_combinations:
            # Fit a linear regression model
            X = sm.add_constant(df[list(predictors)])  # Assuming 'df' is your DataFrame
            model = sm.OLS(df[dependent_variable], X).fit()
        
            # Calculate R-squared and AIC
            r_squared = model.rsquared
            aic = model.aic
            n = len(df[dependent_variable])
            p = len(predictors)
            adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))

            # Update best results if a better subset is found
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_aic = aic
                best_subset = list(predictors)
                best_adjusted_r_squared = adj_r_squared



    print("Best R-squared:", best_r_squared)
    print("Best AIC:", best_aic)
    print("Best Predictor Subset:", best_subset)
    print("Best Adjusted R-squared:", best_adjusted_r_squared)



# Load Dataset as Pandas DataFrame
red_df = pd.read_csv('winequality/winequality-red.csv', sep=';')
white_df = pd.read_csv('winequality/winequality-white.csv', sep=';')
predictor_vars = ['volatile acidity', 'residual sugar', 'alcohol', 'pH']

best_fit(red_df)
print("\n------WHITE WINE-------")
best_fit(white_df)

# red_ols = OLS(red_df, predictor_vars)
# white_ols = OLS(white_df, predictor_vars)

# print("Red OLS with: ", predictor_vars)
# print(red_ols)

# print("\nWhite OLS with: ", predictor_vars)
# print(white_ols)

# red_mr = multiple_regression(red_df, predictor_vars)
# white_mr = multiple_regression(white_df, predictor_vars)

# print("Red MR with: ", predictor_vars)
# print(red_mr)

# print("\nWhite MR with: ", predictor_vars)
# print(white_mr)

# predictor_vars = ['volatile acidity']

# red_ols = OLS(red_df, predictor_vars)
# white_ols = OLS(white_df, predictor_vars)

# print("Red OLS with: ", predictor_vars)
# print(red_ols)

# print("\nWhite OLS with: ", predictor_vars)
# print(white_ols)

# red_mr = multiple_regression(red_df, predictor_vars)
# white_mr = multiple_regression(white_df, predictor_vars)

# print("Red MR with: ", predictor_vars)
# print(red_mr)

# print("\nWhite MR with: ", predictor_vars)
# print(white_mr)