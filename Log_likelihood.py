import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from statsmodels.tsa.stattools import acf

def simulate(x0, kappa, mu, sigma, T, dt):
    X = np.zeros(T)
    X[0] = x0
    for t in range(1,T):
        dX = kappa*(mu - X[t-1])*dt + sigma*np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + dX
    return X

data = pd.read_csv("ForwardBaseLoadContinuation.csv", sep=';')

data.head()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

start_date = "01/01/2024"
end_date = "01/01/2025"
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
#i don t know why but there are some negative prices --> I removed them

prices = filtered_data["Price"].values

i=0
for price in prices:
    a=float(price.replace(',', '.'))
    prices[i]=a
    i+=1

dates = filtered_data["Date"].values
log_prices = np.log(prices.astype(float))


plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(dates, prices, label="Electricity Prices (EUR/MWh)", color='b')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Electricity Prices")

plt.subplot(2, 1, 2)
plt.plot(dates, log_prices, label="Log-Prices", color='r')
plt.xlabel("Date")
plt.ylabel("Log Price")
plt.title("Log of Electricity Prices")

plt.tight_layout()
plt.show()



dt=1/len(log_prices)

def negative_log_likelihood(params):
    kappa, mu, sigma = params
    P_t = log_prices[:-1]
    P_next = log_prices[1:]

    mean_P = P_t + kappa * (mu - P_t) * dt
    var_P = (sigma ** 2) * dt

    log_likelihood = -0.5 * np.sum(((P_next - mean_P) ** 2) / var_P + np.log(var_P))

    return -log_likelihood


initial_guess = [0.1, np.mean(log_prices), np.std(log_prices)]

result = minimize(negative_log_likelihood, initial_guess, method="L-BFGS-B",
                  bounds=[(0, None), (0, None), (0.001, None)])

kappa, mu, sigma = result.x

print(f"kappa = {kappa}, mu = {mu}, sigma = {sigma}")


T = len(log_prices)
simulatedX = simulate(log_prices[0], kappa, mu, sigma, T, dt)

plt.figure(figsize=(10, 5))
plt.plot(dates, log_prices, label='Real Log-Prices', color='b')
plt.plot(dates, simulatedX, label='Estimated OU Process', color='r', linestyle='--')
plt.legend()
plt.savefig("prices.png")
plt.show()


plt.hist(log_prices, bins=10, density=True, alpha=0.6, color='b', label='Real Log-Prices')
plt.hist(simulatedX, bins=10, density=True, alpha=0.6, color='r', label='Estimated OU Process')
plt.legend()
plt.savefig("hist.png")
plt.show()

#validation

#plot the reisultal RESIDUALS= P_next - (P_t + kappa * (mu - P_t) * dt) and check if they are normally distributed

residuals = log_prices[1:] - (log_prices[:-1] + kappa * (mu - log_prices[:-1]) * dt)
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g', label="Residuals")
plt.legend()
plt.show()


import statsmodels.api as sm

sm.graphics.tsa.plot_acf(residuals, lags=30)
plt.savefig("acf.png")
plt.show()

from scipy.stats import ks_2samp

ks_stat, ks_p_value = ks_2samp(log_prices, simulatedX)
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
