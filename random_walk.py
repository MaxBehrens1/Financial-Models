import yfinance as yf
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

'''
    Code that uses a random walk function that sampled from a gaussian to match the Apple stock over 2 years,
with a 1 day trading frequency. I tried to optimise the volatility and growth by minimising the MSE.
This does not work well, as the lowest MSE is always the one with the lowest volatility possible (although
volatility is normally 10-20%). So set volatility constant at 15% and optimised the growth parameter
'''

# Initial parameters of data
plot = True
volatility = 0.15
growth_array = np.arange(0.1, 0.4, 0.01)
dt = 1/503

def generate_model(grow, vol):
    V0 = close[0]
    values =[V0]
    for _ in range(int(1/dt-1)):
        current_val = values[-1]
        norm_val = normal()
        new_val = (1 + grow * dt) * current_val + vol * current_val * norm_val * (dt ** 0.5)
        values.append(new_val)
    return values

def mse(data1, data2):
    sum_square = 0
    for i in range(len(data1)):
        sum_square += (data1[i] - data2[i]) ** 2
    actual_mse = sum_square / len(data1)
    return actual_mse

#Load in market data
days = np.linspace(0, 1/dt - 1, int(1/dt))
aapl = yf.Ticker(ticker='AAPL')
history = aapl.history(interval='1d', period='24mo')
close = history.Close

# To find optimal parameters
MSE=np.zeros(shape=(len(growth_array)))

for count, gro in enumerate(growth_array): # growth
    total_mse = 0
    for _ in range (50): # running 50 times and taking average
        fit_data = generate_model(gro, volatility)
        total_mse += mse(fit_data, close)
    MSE[count] = total_mse/50
        
min_index = MSE.argmin()
ideal_growth = growth_array[min_index]
print('Optimal growth @ 15% volatility:', ideal_growth)

# To plot
if plot:
    plt.plot(days, close, color='black', label='Data')
    plt.plot(days, generate_model(ideal_growth, volatility), color='red', label='Fit')
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.show()