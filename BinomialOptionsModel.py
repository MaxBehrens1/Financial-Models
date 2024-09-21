import numpy as np

'''
    Code that uses the binomial model to value vanilla european call options for a simple asset.
    I use the equations V = (p'.V+ + (1-p').V-)/(1 + r.dt). Where p prime is the risk-free probability
'''

# initialise parameters
S0 = 100 # value at t=0
strike = 100 # strike price
t_mat = 1 # time to maturity in years
r = 0.06 # anual interest
N = 3 # number of time steps
u = 1.1 # up multiplier
d = 1/u # down multiplier to ensure joining of tree
dt = t_mat / N
volatility = (u - 1) / np.sqrt(dt) # chosen simple solution to simultaneous eqns

def binomail_tree(S0, strike, t_mat, r, N, u, volatility, dt):
    # asset prices at maturity
    asset_price = S0 * u**(np.arange(0, N+1, 1)) * d**(np.arange(N, -1, -1))
    print('Asset prices at expiery:', asset_price)
    
    # option values
    option_value = np.maximum(asset_price-strike, 0)
    print('Option values at expiery:', option_value)
    
    # stepping backwards through tree
    pprime = 0.5 + (r * np.sqrt(dt)) / (2 * volatility)
    for i in np.arange(N, 0, -1):
        option_value = (pprime * option_value[1:i+1] + (1 - pprime) * option_value[0:i]) / (1 + dt * r)
    print('Initial option value:', option_value)

binomail_tree(S0, strike, t_mat, r, N, u, volatility, dt)
