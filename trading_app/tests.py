import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yfinance as yf
import environ
env = environ.Env()
environ.Env.read_env(overwrite=True)  # reading .env file


from .models import Ticker, DailyPrice

ticker_symbol = 'AAPL'
ticker = yf.Ticker(ticker_symbol)
ticker_info = ticker.info
print(ticker_info['exchange'])