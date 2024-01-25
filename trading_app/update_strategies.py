import logging

logging.basicConfig(level=logging.INFO)
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yfinance as yf
from time import sleep
from .models import Ticker, DailyPrice, FifteenMinPrice, FiveMinPrice, TickerCategory, TradingOpp, TradingStrategy

import pandas as pd
import pytz
from datetime import datetime, timedelta, timezone, date, time
from candlestick import candlestick
from . import db_candlestick
from django.utils import timezone
import logging
from decimal import Decimal
import math

logger = logging.getLogger('django')

def display_local_time():
    # Get the current datetime in UTC
    utc_now = datetime.utcnow()

    # Convert UTC datetime to the local timezone
    local_timezone = pytz.timezone('Europe/Riga')  # Replace with your local timezone
    local_datetime = utc_now.replace(tzinfo=pytz.utc).astimezone(local_timezone)

    # Format and print the local datetime
    local_datetime_str = local_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f'Current datetime: {local_datetime_str}')

class BaseStrategy:
    def __init__(self, ticker):
        self.ticker = ticker

    def check_criteria(self):
        raise NotImplementedError("Each strategy must implement its own check_criteria method.")

class StrategyA(BaseStrategy):
    name="Rising trend"
    data = {}

    def check_criteria(self):
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()

        # Check if latest_price meets the criteria
        if latest_price and latest_price.trend == 1:
            data ={'trend' : latest_price.trend, 'bullish_detected' : latest_price.bullish_detected}
            return True, data
        return False, data

from django.utils import timezone

def process_trading_opportunities():
    logger.error(f'Starting process_trading_opportunities()...')
    tickers = Ticker.objects.all()
    strategies = [StrategyA]  # List of strategy classes

    for ticker in tickers:
        #print('Ticker:',ticker.symbol)
        #logger.error(f'Ticker: "{str(ticker.symbol)}"...')
        for StrategyClass in strategies:
            strategy = StrategyClass(ticker)
            #print('strategy:', strategy)
            logger.error(f'Checking strategy: "{str(strategy.name)}" for ticker "{str(ticker.symbol)}"')
            strategy_valid, data = strategy.check_criteria()
            if strategy_valid:
                #print('Strategy criteria met for', ticker.symbol)
                logger.error(f'Strategy criteria met for "{str(ticker.symbol)}"...')
                strategy_instance = TradingStrategy.objects.get(name=strategy.name)
                TradingOpp.objects.create(
                    ticker=ticker,
                    strategy=strategy_instance,
                    datetime_identified=timezone.now(),
                    metrics_snapshot=data  # Capture relevant metrics
                )
    logger.error(f'Finished process_trading_opportunities().')

# Call this function daily after metrics update
process_trading_opportunities()
