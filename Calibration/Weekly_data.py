import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats


def simulate(x0, kappa, mu, sigma, T, dt=1):
    X = np.zeros(T)
    X[0] = x0
    for t in range(1, T):
        dX = kappa * (mu - X[t-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + dX
    return X

# Load data
data = pd.read_csv("Calibration/european_wholesale_electricity_price_data_daily.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')


start_date = "2023-07-01"
end_date = "2025-07-01"

filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()


filtered_data = filtered_data.set_index("Date")
filtered_data = filtered_data.drop('ISO3 Code', axis=1)
filtered_data = filtered_data[~filtered_data['Country'].isin(['Finland', 'Sweden', 'Norway', 'Iceland', 'Ireland', 'United Kingdom'])]

weekly_data = (
    filtered_data
    .groupby('Country')
    .resample('W')
    .mean()
    .reset_index()
    .rename(columns={'Price (EUR/MWhe)': 'Weekly_Price'})
)



print(weekly_data.head())



results = []

for country in weekly_data['Country'].unique():
    country_data = weekly_data[weekly_data['Country'] == country]
    price = country_data["Weekly_Price"].values
    dates = country_data["Date"].values
    dt = 1 / len(price)

    def negative_log_likelihood(params):
        kappa, mu, sigma = params
        P_t = price[:-1]
        P_next = price[1:]
        mean_P = P_t + kappa * (mu - P_t) * dt
        var_P = (sigma ** 2) * dt
        log_likelihood = -0.5 * np.sum(((P_next - mean_P) ** 2) / var_P + np.log(var_P))
        return -log_likelihood

    initial_guess = [0.1, np.mean(price), np.std(price)]
    result = minimize(negative_log_likelihood, initial_guess, method="L-BFGS-B",
                      bounds=[(0, None), (0, None), (0.001, None)])
    kappa, mu, sigma = result.x

    T = len(price)
    simulatedX = simulate(price[0], kappa, mu, sigma, T, dt)

    residuals = price[1:] - (price[:-1] + kappa * (mu - price[:-1]) * dt)
    ks_statistic, ks_pvalue = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

    results.append({
        "Country": country,
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "p_value": ks_pvalue,
        "x0": float(np.mean(price)),
    })

summary_df = pd.DataFrame(results)
print(summary_df)


summary_df.to_csv('Calibration/calibrated_parameters.csv', index=False)















country = "Estonia"  # Change this to the country you want to visualize
price = weekly_data[weekly_data['Country'] == country]["Weekly_Price"].values
dates = weekly_data[weekly_data['Country'] == country]["Date"].values
dt = 1 / len(price)
T = len(price)

kappa = summary_df[summary_df['Country'] == country]["kappa"].values
mu = summary_df[summary_df['Country'] == country]["mu"].values
sigma = summary_df[summary_df['Country'] == country]["sigma"].values

simulatedX = simulate(price[0], kappa, mu, sigma, T, dt)

plt.figure(figsize=(10, 5))
plt.plot(dates, price, label='Real Log-Prices', color='b')
plt.plot(dates, simulatedX, label='Estimated OU Process', color='r', linestyle='--')
plt.legend()
plt.savefig("prices.png")
plt.show()


plt.hist(price, bins=10, density=True, alpha=0.6, color='b', label='Real Log-Prices')
plt.hist(simulatedX, bins=10, density=True, alpha=0.6, color='r', label='Estimated OU Process')
plt.legend()
plt.savefig("hist.png")
plt.show()

#validation

#plot the reisultal RESIDUALS= P_next - (P_t + kappa * (mu - P_t) * dt) and check if they are normally distributed

residuals = price[1:] - (price[:-1] + kappa * (mu - price[:-1]) * dt)
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g', label="Residuals")
plt.legend()
plt.show()


import statsmodels.api as sm

sm.graphics.tsa.plot_acf(residuals, lags=30)
plt.savefig("acf.png")
plt.show()


from scipy.stats import ks_2samp

ks_stat, ks_p_value = ks_2samp(price, simulatedX)
print(f"KS test statistic: {ks_stat}, p-value: {ks_p_value}")




import scipy.stats as stats

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.savefig("qq.png")
plt.show()


import numpy as np
from scipy import stats

# Assuming you have residuals from your previous calculations
# residuals = log_prices[1:] - (log_prices[:-1] + kappa * (mu - log_prices[:-1]) * dt)

# Perform the Kolmogorov-Smirnov test against a normal distribution
ks_statistic, ks_pvalue = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

print(f"Kolmogorov-Smirnov Test Statistic: {ks_statistic}")
print(f"Kolmogorov-Smirnov p-value: {ks_pvalue}")

