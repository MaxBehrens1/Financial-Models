# CQF-models

## 1. LognormalRandomWalk.py

Code that uses a random walk function sampled from a gaussian to match the Apple stock over 2 years, with a 1 day trading frequency. I tried to optimise the volatility and growth by minimising the MSE. This does not work well, as the lowest MSE is always the one with the lowest volatility possible (although volatility is normally 10-20%). So set volatility constant at 15% and optimised the growth parameter

## 2. BinomialOptionsModel.py

Code that uses the binomial model to value vanilla european call options for a simple asset. I use the equations $V = \frac{p'V^{+} + (1-p')V^{-}}{1 + r.dt}$. Where p prime is the risk-free probability, r is the interest rate, dt is the time-step and $V^+$ and $V^-$ are the higher and lower value of the option. 
