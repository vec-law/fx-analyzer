import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# przykładowe ceny
prices = pd.Series([100,102,101,103,104,102,101,105,106,107])
n = 3  # okres RSI

# obliczamy delta, gain i loss
delta = prices.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# średnie kroczące
avg_gain = gain.rolling(n).mean()
avg_loss = loss.rolling(n).mean()

# RS i RSI
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# wykres
plt.figure(figsize=(10,5))
plt.plot(prices, label='Close Price', marker='o')
plt.plot(rsi, label=f'RSI({n})', marker='x')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('Price vs RSI')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
