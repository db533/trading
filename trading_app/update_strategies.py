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

class TAEStrategy(BaseStrategy):
    name="Trend - Area of value - Event strategy"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        #latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        ma_200_trend_strength = self.ticker.ma_200_trend_strength
        tae_strategy_score = self.ticker.tae_strategy_score
        bullish_detected = self.ticker.bullish_detected
        bullish_reversal_detected = self.ticker.bullish_reversal_detected
        bearish_detected = self.ticker.bearish_detected

        if tae_strategy_score > 0 and ma_200_trend_strength > 0 and (bullish_detected > 0 or bullish_reversal_detected > 0):
                action_buy = True
        if tae_strategy_score > 0 and ma_200_trend_strength < 0 and bearish_detected > 0:
                action_buy = False
        if action_buy is not None:
            data = {'tae_strategy_score': str(tae_strategy_score), 'ma_200_trend_strength' : str(ma_200_trend_strength),
                    'bullish_detected': str(bullish_detected), 'bearish_detected': str(bearish_detected), 'bullish_reversal_detected' : str(bullish_reversal_detected)}
            return action_buy, data
        return action_buy, data

class TwoPeriodCumRSI(BaseStrategy):
    name="Two period cumulative RSI"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        #latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        cumulative_two_period_two_day_rsi = self.ticker.cumulative_two_period_two_day_rsi

        if cumulative_two_period_two_day_rsi < 10:
            action_buy = True
        if cumulative_two_period_two_day_rsi > 65:
            action_buy = False
        if action_buy is not None:
            data = {'cumulative_two_period_two_day_rsi': str(cumulative_two_period_two_day_rsi),}
            return action_buy, data
        return action_buy, data

class DoubleSevens(BaseStrategy):
    name="Double 7's strategy"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        if latest_price is not None:
            latest_close_price = latest_price.close_price
            seven_day_max = self.ticker.seven_day_max
            seven_day_min = self.ticker.seven_day_min

            if latest_close_price is not None:
                if latest_close_price <= seven_day_min:
                    action_buy = True
                    data = {'latest_close_price' : str(latest_close_price), 'seven_day_min': str(seven_day_min), }
                elif latest_close_price >= seven_day_max:
                    action_buy = False
                    data = {'latest_close_price': str(latest_close_price), 'seven_day_max': str(seven_day_max), }
        return action_buy, data

def instance_difference_count(earlier_candle):
    # Given datetime index (ensure it's timezone-aware if your model uses timezone-aware datetimes)
    earlier_dt = earlier_candle.datetime
    # Count instances of DailyPrice where the date is greater than the given datetime
    count = DailyPrice.objects.filter(date__gt=given_datetime).count()
    return count

class GannPointFour(BaseStrategy):
    name="Gann's Buying / Selling point #4"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__isnull=False).only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point').order_by('-datetime')
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_counter = 1
        strategy_valid = True
        existing_downtrend = None
        T_prev = []
        latest_T = 0
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'LL':
                    existing_downtrend = True
                    action_buy = True
                    logger.info(f'Detected first swingpoint. LL')
                elif swing_point.swing_point_label == 'HH':
                    logger.info(f'Detected first swingpoint. HH')
                    existing_downtrend = False
                    action_buy = False
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.info(f'First swingpoint not HH or LL. Stratey not valid.')
                    strategy_valid = False
                    break
                # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(swing_point)
                swing_point_counter += 1
            elif swing_point_counter >1:
                if swing_point.swing_point_label == 'LH':
                    # Swing point is a high on the down trend.
                    # Sve the number of days that that it took to reach this swing point.
                    logger.info(f'Found a prior LH.')
                    T_prev.append(swing_point.candle_count_since_last_swing_point)
                elif swing_point.swing_point_label == 'LL':
                    logger.info(f'Found a prior LL.')
                elif swing_point.swing_point_label == 'HH':
                    # This must be the start of the prior down trend.
                    # Stop checking further swing points.
                    logger.info(f'Found a prior HH. So downtrend started here.')
                    break
                swing_point_counter += 1
        #if strategy_valid == True and len(T_prev) > 0:
        if strategy_valid == True:
            if len(T_prev) > 0:
                max_T = max(T_prev)
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)), 'count_T_prev': str(len(T_prev)), }
            else:
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
            #if latest_T <= max_T:
            #    strategy_valid = False

        return action_buy, data

from django.utils import timezone

def process_trading_opportunities():
    logger.error(f'Starting process_trading_opportunities()...')
    tickers = Ticker.objects.all()
    #strategies = [TAEStrategy, TwoPeriodCumRSI, DoubleSevens, GannPointFour]  # List of strategy classes
    strategies = [GannPointFour]  # List of strategy classes
    #ticker_id_in_strategy = []

    for ticker in tickers:
        for StrategyClass in strategies:
            strategy = StrategyClass(ticker)
            #print('strategy:', strategy)
            logger.error(f'Checking strategy: "{str(strategy.name)}" for ticker "{str(ticker.symbol)}"')
            action_buy, data = strategy.check_criteria()
            strategy_instance = TradingStrategy.objects.get(name=strategy.name)
            existing_tradingopp = TradingOpp.objects.filter(ticker=ticker).filter(is_active=1).filter(strategy=strategy_instance)
            if len(existing_tradingopp) > 0:
                existing_tradingopp = existing_tradingopp[0]
            else:
                existing_tradingopp = None
            if action_buy is not None:
                #print('Strategy criteria met for', ticker.symbol)
                logger.error(f'Criteria met for "{str(ticker.symbol)}" for trading strategy"{str(strategy.name)}"...')
                #ticker_id_in_strategy.append(ticker.id)
                if existing_tradingopp is not None:
                    # This Ticker / strategy exists as an active record. Increment the count.
                    existing_tradingopp.count += 1
                    existing_tradingopp.save()
                else:
                    # This Ticker / strategy is new. Create a new TradingOpp instance.
                    TradingOpp.objects.create(
                        ticker=ticker,
                        strategy=strategy_instance,
                        datetime_identified=timezone.now(),
                        metrics_snapshot=data, # Capture relevant metrics
                        count = 1,
                        action_buy = action_buy
                    )
            else:
                # The strategy is not valid for the ticker.
                # Check if there was an active TradingOpp for this Ticker / strategy and set is_active=0
                if existing_tradingopp is not None:
                    existing_tradingopp.is_active = False
                    existing_tradingopp.save()
    logger.error(f'Finished process_trading_opportunities().')

# Call this function daily after metrics update
process_trading_opportunities()
