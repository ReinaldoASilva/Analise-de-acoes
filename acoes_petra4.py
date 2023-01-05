import yfinance as yf
from pandas import json_normalize
import pandas as pd
#criando ticker
ticker = yf.Ticker('PBR')
#cotação
ticker.history(period='6d', interval='15m')
# Balanço trimestral
ticker.quarterly_balance_sheet.T
#Dividendos
ticker.dividends
ticker.get_info()['preMarketPrice']
df = pd.dataframe[ticker.history(period='6d', interval='15m')]
