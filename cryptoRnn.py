


import pandas as pd


df = pd.read_csv('crypto_data/LTC-USD.csv',
		names = ['time', 'low', 'high', 'open', 'close', 'volume'])


print(df.head())