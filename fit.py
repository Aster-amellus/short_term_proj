import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from datetime import datetime, timedelta
import GPy

# Set random seed
np.random.seed(123)

# Download and read the data
cUrl = "http://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/flask_co2/daily/daily_flask_co2_mlo.csv"
cFile = cUrl.split("/")[-1]

# Download file if it doesn't exist
urllib.request.urlretrieve(cUrl, cFile)

# Read the data
co2s = pd.read_csv(cFile, skiprows=69, header=None, names=["day", "time", "junk1", "junk2", "Nflasks", "quality", "co2"])

# Parse dates and calculate time in years from a specific origin
co2s['date'] = pd.to_datetime(co2s['day'] + ' ' + co2s['time'], format='%Y-%m-%d %H:%M')
co2s['day'] = pd.to_datetime(co2s['day'])
time_origin = datetime(2000, 3, 30)
co2s['timeYears'] = (co2s['day'] - time_origin).dt.total_seconds() / (365.25 * 24 * 3600)
co2s['dayInt'] = (co2s['day'] - time_origin).dt.days

# Filter dataset
observed_dataset = co2s.dropna(subset=['co2'])
observed_dataset['quality'] = np.where(co2s['quality'] > 0, 1, 0)
observed_dataset = observed_dataset[observed_dataset['quality'] == 0]
train_dataset = observed_dataset[(observed_dataset['timeYears'] >= 0) & (observed_dataset['timeYears'] <= 20)]

# Define periods and corresponding frequencies
years_cyl = [1, 0.5, 44/12, 9.1, 10.4]
periods = [1/y for y in years_cyl]
frequencies = [2 * np.pi * p for p in periods]

# Create features based on sine and cosine transformations
def create_features(x, frequencies):
    features = [np.cos(f * x) for f in frequencies] + [np.sin(f * x) for f in frequencies]
    return np.column_stack(features)

X_train = create_features(train_dataset['timeYears'], frequencies)

# Gaussian Process Regression
y_train = train_dataset['co2'].values.reshape(-1, 1)
kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X_train, y_train, kernel)
model.optimize()

# Create future time points for prediction
future_years = np.arange(20, 30, 1/365.25)  # Predict for the next 10 years
X_future = create_features(future_years, frequencies)

# Make predictions
mu_future, sigma_future = model.predict(X_future)

# Create a DataFrame for predictions
pred_df = pd.DataFrame({
    'timeYears': np.concatenate([observed_dataset['timeYears'], future_years]),
    'date': np.concatenate([observed_dataset['day'], [time_origin + timedelta(days=int(y * 365.25)) for y in future_years]]),
    'mean': np.concatenate([observed_dataset['co2'], mu_future.flatten()]),
    'lower': np.concatenate([observed_dataset['co2'], (mu_future.flatten() - 1.96 * np.sqrt(sigma_future.flatten()))]),
    'upper': np.concatenate([observed_dataset['co2'], (mu_future.flatten() + 1.96 * np.sqrt(sigma_future.flatten()))])
})

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(pred_df['date'], pred_df['mean'], color='blue', label='Predicted Mean')
plt.fill_between(pred_df['date'], pred_df['lower'], pred_df['upper'], color='red', alpha=0.3, label='Confidence Interval')
plt.scatter(train_dataset['day'], train_dataset['co2'], color='black', s=10, label='Observed CO2')
plt.axvline(x=train_dataset['day'].max(), color='purple', linestyle='dashed', label='End of Training Data')
plt.xlabel('Date')
plt.ylabel('CO2 Concentration')
plt.title('CO2 Concentration Predictions')
plt.legend()
plt.grid(True)
plt.show()

# Calculate training and testing errors
train_pred = pred_df[pred_df['timeYears'] <= train_dataset['timeYears'].max()]
train_rmse = np.sqrt(np.mean((train_pred['mean'].values - train_dataset['co2'].values) ** 2))
train_coverage = np.mean((train_pred['lower'] <= train_dataset['co2']) & (train_dataset['co2'] <= train_pred['upper']))

print(f"Training RMSE: {train_rmse}")
print(f"Training Coverage: {train_coverage}")

test_pred = pred_df[pred_df['timeYears'] > train_dataset['timeYears'].max()]
test_rmse = np.sqrt(np.mean((test_pred['mean'].values - mu_future.flatten()) ** 2))
test_coverage = np.mean((test_pred['lower'] <= mu_future.flatten()) & (mu_future.flatten() <= test_pred['upper']))

print(f"Testing RMSE: {test_rmse}")
print(f"Testing Coverage: {test_coverage}")