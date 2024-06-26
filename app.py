import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import streamlit as st

# Generate some synthetic data with outliers
np.random.seed(0)
n = 100
X = np.random.rand(n, 1) * 10
epsilon = np.random.randn(n, 1) * 2
y = 3 * X.squeeze() + 2 + epsilon.squeeze()

# Create initial DataFrame without outliers
df = pd.DataFrame({'X': X.squeeze(), 'y': y})

# Define function to update plot based on outliers
def update_plot(X_outliers, y_outliers):
    X_outliers = np.array([float(x) for x in X_outliers.split(',') if x.strip()]) if X_outliers else np.array([])
    y_outliers = np.array([float(y) for y in y_outliers.split(',') if y.strip()]) if y_outliers else np.array([])

    # Introduce outliers
    outliers_df = pd.DataFrame({'X': X_outliers, 'y': y_outliers})
    df_with_outliers = pd.concat([df, outliers_df]).reset_index(drop=True)
    
    # Fit both models
    model_ols = smf.ols('y ~ X', data=df_with_outliers).fit()
    model_rlm = smf.rlm('y ~ X', data=df_with_outliers, M=sm.robust.norms.HuberT()).fit()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(df['X'], df['y'], label='Data')
    if not outliers_df.empty:
        plt.scatter(outliers_df['X'], outliers_df['y'], color='red', marker='o', label='Outliers')
    
    X_range = np.linspace(df['X'].min(), df['X'].max(), 100)
    plt.plot(X_range, model_ols.predict({'X': X_range}), label='Linear', color='orange')
    plt.plot(X_range, model_rlm.predict({'X': X_range}), label='Huber', color='green')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Comparison of Linear and Huber Regression with Outliers')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Display MSE for both models
    mse_rlm = mean_squared_error(df['y'], model_rlm.predict(df['X']))
    mse_ols = mean_squared_error(df['y'], model_ols.predict(df['X']))
    st.write('Huber Regression MSE:', mse_rlm)
    st.write('Linear Regression MSE:', mse_ols)

    st.write(
        """
        **Notice:** The MSE of the Huber regressor does not significantly change compared to the linear regressor, 
        which means the performance of the Huber regressor is not affected by outliers.
        """
    )

# Streamlit interface
st.markdown("<h1 style='text-align: center;'>Comparison of Linear and Huber Regression with Outliers</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Welcome to the regression comparison dashboard!</p>", unsafe_allow_html=True)

X_outliers = st.text_input("Enter X outliers (comma-separated):", "")
y_outliers = st.text_input("Enter y outliers (comma-separated):", "")

if st.button('Update Plot'):
    update_plot(X_outliers, y_outliers)
else:
    update_plot('', '')
