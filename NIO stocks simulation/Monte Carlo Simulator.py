# NIO Stonks simulator step 1. Download closing prices for an Asset
# DONE 
import pandas as pd 
import math as math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Step 2 : find the periodic daily returns 
# Today's price = yesterday * e^r , r = daily return

NIO_df = pd.DataFrame(pd.read_csv("NIO_HD.csv"))
precios = NIO_df.Price
daily_return = []
yesterstonk = 1

for stonk in precios:
    daily_return.append(math.log(stonk/precios[yesterstonk]))
    yesterstonk += 1
    if yesterstonk == precios.size: 
        break 

# List to min max axis limit
def find_reference(valList):
    return  max([abs(min(valList)),abs( max(valList))])*1.1
    

# Step 3 : Find the mean, Variance and STD of PDR (Periodic Daily Returns)
PDR = daily_return
_MEAN = np.mean(PDR)
_STD = np.std(PDR)
_VARIANCE = np.var(PDR)
# MEAN, VARIANCE, STD

# Step 4 : We create the formula of a Drift plus a random stochastic offset.
Drift = _MEAN - _VARIANCE/2
print(Drift)

# Step 5 : Do magic
# Tomorrow Stock Price = Previous Day Stock Price * exp(Drift + STD * NormSINV(rand()))
# Random percentages for the area under the normal curve
#   exp(Drift + STD * NormSINV(rand()))
#   math.exp(Drift + _STD * norm.ppf(np.random.normal(loc = _MEAN,scale = _STD),loc = _MEAN ,scale = _STD)))
print(math.exp(Drift + _STD * norm.ppf(np.random.normal(loc = _MEAN,scale = _STD),loc = _MEAN ,scale = _STD)))

# Simulate the stonks
stonk_vector = []
stonky = []
yesterstonk = precios.size 
last_stonk =  precios[0]

iteraciones = 100 # Numero de simulaciones a realizar
for stonklist in range(0,iteraciones):
    for stonk in precios:
        driftFactor = math.exp(Drift + _STD * norm.ppf(np.random.normal(loc = 0,scale = 1),loc = 0 ,scale = 1))
        stonk_vector.append(last_stonk * driftFactor)
    stonky.append(stonk_vector)
    plt.ylabel('Daily Expected')
    plt.xlabel('Days Past on Each Iteration ')
    plt.plot(range(0,precios.size),stonky[stonklist])
    stonk_vector.clear()

plt.show()

