import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm

# import itertools

# OLS Function runs OLS model on given dataframe
def OLS(X, y, n, p):
    # Using stats model OLS function to find fit
    model = sm.OLS(y, X).fit()
    
    # Calculate r-squared, aic, adjusted r-squared and p-values
    r_squared = model.rsquared
    aic = model.aic
    p_value = model.pvalues
    adj_r_squared = 1 - (1-r_squared) * ((n-1)/(n-p-1))
    return model, r_squared, aic, adj_r_squared, p_value

# Multiple Linear Regression
def mlr(X, y, n, p):
    # Split the dataset into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create linear Regression model to fit training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use test data as prediction
    y_pred = model.predict(X_test)

    # Evaluate the performance using r-squared test
    r_squared = r2_score(y_test, y_pred)

    # Number of iteration to bootstrap
    n_iterations = 1000
    # Initialize empty list to store coefficients in
    coefs = [] 

    # Calculate p-values using bootstrapping
    for _ in range(n_iterations):
        # Set indices used for sampling
        indices = np.random.choice(len(X_train), len(X_train), replace=True)

        # Create samples
        X_train_sampled = X_train.values[indices]
        y_train_sampled = y_train.values[indices]

        # Calculate linear regression based on samples
        model = LinearRegression()
        model.fit(X_train_sampled, y_train_sampled)

        # Update the coefficient list with values from the model
        coefs.append(model.coef_)

    # Calculate p-values
    coef_means = np.mean(coefs, axis=0)
    coef_stds = np.std(coefs, axis=0)
    z_scores = coef_means / coef_stds
    p_values = [2 * (1 - norm.cdf(abs(z))) for z in z_scores]

    # Calculate the adjusted R-squared value
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Due to different formatting of performance variables, output results in function
    print(f"\n Performance Variables for: Multiple Linear Regression")
    print(f"R_Squared: {r_squared:.4f}")
    print(f"Adjusted_R_squared: {adj_r_squared:.4f}")
    print("P-Values: ")
    for i, p_value in enumerate(p_values):
        print(f'Coefficient {i}: {p_value}')

# GLM function with Gaussian fit
def glm(X, y, n, p):
    # Use Gaussian family for our GLM model
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    # Fit the model
    results = model.fit()
    
    # Calculate pseudo r-squared using null deviance and deviance
    null_deviance = results.null_deviance
    deviance = results.deviance
    r_squared = 1 - (deviance/null_deviance)

    # Use r-squared value to calculate the adj_r_squared
    adj_r_squared = 1 - (1-r_squared) * ((n-1)/n-p-1)

    # Calculate AIC
    aic = results.aic

    # Not using this model so pass bad p-values as placeholder
    p_value = 100

    return r_squared, adj_r_squared, aic, p_value, results

# Function to output performance values
def output(r_squared, aic, adj_r_squared, p_value,model, model_name):
    print(f"\n Performance Variables for: {model_name}")
    print("R_Squared: ", r_squared)
    print(f"AIC: {aic:.4f}")
    print(f"Adjusted_R_squared: {adj_r_squared:.4f}")
    print("P-values: ", p_value)
    print("\nSummary:", model.summary())
    
# Visualize the coefficient differences between OLS and Linear Regression
def visualize(df, model):

    # OLS Visual
    coefficients = [4.2987, -1.2766, -0.0064, -0.4283, 0.3306]
    predictors = ['volatile acidity', 'residual sugar', 'pH', 'alcohol']

    # Scatterplot with OLS regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(sm.add_constant(df[list(predictors)]), df['quality'])
    plt.plot(sm.add_constant(df[list(predictors)]), model, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatterplot with OLS Regression Line')
    plt.show()

    coefficients = [0, 0, 0, 0.9771, 0, 0.00791]
    predictors = ['volatile acidity', 'residual sugar', 'pH', 'alcohol']

    # Scatterplot with OLS regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(sm.add_constant(df[list(predictors)]), df['quality'])
    plt.plot(sm.add_constant(df[list(predictors)]), model, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatterplot with Multiple Linear Regression Line')
    plt.show()

# Main function to run the models
def main():
    red_df = pd.read_csv('winequality/winequality-red.csv', sep=';')
    predictor_vars = ['volatile acidity', 'residual sugar', 'alcohol', 'pH']

    # Use predictor_vars as independent vars
    X = red_df[predictor_vars]
    # Using quality as our dependent variable
    y = red_df['quality']

    # Add a constant term (ie. the intercept)
    X = sm.add_constant(X)

    # Calculate n and p values for adj_r_squared
    n = len(y)
    p = len(X)

    # Run through each of our 3 models 
    ols_model, r_squared, aic, adj_r_squared, p_value = OLS(X, y, n, p)
    glm_model, r_squared, aic, adj_r_squared, p_value = glm(X, y, n, p)
    # Multiple Linear Regression function to calculate and output calculateions
    mlr(X, y, n, p)

    # Get the output (r-squared, aic, adj_r_squared, p-values)
    #  to be used for comparison
    output(ols_model, r_squared, aic, adj_r_squared, p_value, "OLS")
    output(glm_model, r_squared, aic, adj_r_squared, p_value, "GLM")

    # Visualization function, mostly to help with analysis
    visualize(red_df)

if __name__ == "__main__":
    main()   