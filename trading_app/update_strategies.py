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

def instance_difference_count(ticker, earlier_candle, later_candle=None):
    # Given datetime index (ensure it's timezone-aware if your model uses timezone-aware datetimes)
    earlier_dt = earlier_candle.datetime
    if later_candle is not None:
        later_dt = later_candle.datetime
        count = DailyPrice.objects.filter(ticker=ticker).filter(datetime__gt=earlier_dt).filter(datetime__lte=later_dt).count()
    else:
        # Count instances of DailyPrice where the date is greater than the given datetime
        count = DailyPrice.objects.filter(ticker=ticker).filter(datetime__gt=earlier_dt).count()
    return count

class GannPointFourBuy(BaseStrategy):
    name="Gann's Buying point #4"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point').order_by('-datetime')
        #latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_counter = 1
        T_prev = []
        latest_T = 0
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'LL':
                    logger.error(f'Detected first swingpoint. LL')
                    last_candle = swing_point
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not LL. Strategy not valid.')
                    break
                # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                most_recent_label = 'LL'
                swing_point_counter += 1
            elif swing_point_counter >1:
                if swing_point.swing_point_label == 'LH' and most_recent_label == 'LL':
                    # Swing point is a high.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    T_prev.append(swing_point.candle_count_since_last_swing_point)
                    most_recent_label = 'LH'
                elif swing_point.swing_point_label == 'LL':
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    most_recent_label = 'LL'

                elif swing_point.swing_point_label == 'HH' or swing_point.swing_point_label == 'HL':
                    # This must be the start of the prior up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. So downtrend started here.')
                    first_candle = swing_point
                    break
                swing_point_counter += 1
        if len(T_prev) > 0:
            max_T = max(T_prev)
            if max_T < latest_T and len(T_prev) > 1:
                # The most recent upward rally is longer than the longest upward rally during the down trend.
                # And we have had at least 2 sections of upward movement during the down trend.
                logger.error(f'Latest upswing LONGER than longest up swing during down trend. Strategy valid.')
                action_buy = True
            else:
                # The most recent upward rally is shorter than the longest upward rally during the down trend.
                logger.error(f'Latest upswing shorter than longest up swing during down trend. Strategy not valid.')
                action_buy = None
            # Compute the days between the start and end of the down trend.
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)), 'count_T_prev': str(len(T_prev)), 'prior_trend_duration' : str(prior_trend_duration) }
            logger.error(f'Max T during prior series of swings: {max_T}.')
        else:
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data

class GannPointFourSell(BaseStrategy):
    name="Gann's Selling point #4"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point').order_by('-datetime')
        swing_point_counter = 1
        existing_downtrend = None
        T_prev = []
        latest_T = 0
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'HH':
                    logger.error(f'Detected first swingpoint. HH')
                    last_candle = swing_point
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                most_recent_label = 'HH'
                swing_point_counter += 1
            elif swing_point_counter >1:
                if swing_point.swing_point_label == 'HL' and most_recent_label == 'HH':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    T_prev.append(swing_point.candle_count_since_last_swing_point)
                    most_recent_label = 'HL'
                elif swing_point.swing_point_label == 'HH':
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    most_recent_label = 'HH'
                elif swing_point.swing_point_label == 'LL' or swing_point.swing_point_label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. So downtrend / uptrend started here.')
                    first_candle = swing_point
                    break
                swing_point_counter += 1
        if len(T_prev) > 0:
            max_T = max(T_prev)
            if max_T < latest_T and len(T_prev) > 1:
                # The most recent downward rally is longer than the longest downward rally during the down trend.
                # And we have at least 2 sections of downward movement during the most recent upward trend.
                logger.error(f'Latest downswing LONGER than longest down swing during up trend. Strategy valid.')
                action_buy = False
            else:
                # The most recent downward rally is shorter than the longest downward rally during the down trend.
                logger.error(f'Latest downswing shorter than longest down swing during up trend. Strategy not valid.')
                action_buy = None
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)), 'count_T_prev': str(len(T_prev)), 'prior_trend_duration' : str(prior_trend_duration)}
            logger.error(f'Max T during prior series of swings: {max_T}.')
        else:
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data

class GannPointFiveBuy(BaseStrategy):
    name="Gann's Buying point #5"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point').order_by('-datetime')
        swing_point_counter = 1
        T_most_recent = None
        latest_T = 0
        section_count = 0
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'LL':
                    logger.error(f'Detected first swingpoint. LL')
                    last_candle = swing_point
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not LL. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.swing_point_label == 'LH':
                    # Swing point is a high on the down trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(
                        f'Found a prior {swing_point.swing_point_label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    #most_recent_swing_label = swing_point.swing_point_label
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                elif swing_point.swing_point_label == 'LL':
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    if swing_point.swing_point_label == 'LL':
                        section_count += 1
                        #most_recent_swing_label = swing_point.swing_point_label
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.swing_point_label == 'HH' or swing_point.swing_point_label == 'HL':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. So downtrend / uptrend started here.')
                    first_candle = swing_point
                    break
                swing_point_counter += 1
        if T_most_recent is not None:
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),
                    'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration)}
            logger.error(f'T_most_recent during prior series of swings: {T_most_recent}.')
            if T_most_recent < latest_T and section_count > 1:
                action_buy = True
        else:
            data = {'latest_T': str(latest_T),'section_count': str(section_count),}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data

class GannPointFiveSell(BaseStrategy):
    name="Gann's Selling point #5"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point').order_by('-datetime')
        swing_point_counter = 1
        T_most_recent = None
        latest_T = 0
        section_count = 0
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'HH':
                    logger.error(f'Detected first swingpoint. HH')
                    last_candle = swing_point
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.swing_point_label == 'HL':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(
                        f'Found a prior {swing_point.swing_point_label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                elif swing_point.swing_point_label == 'HH':
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    if swing_point.swing_point_label == 'HH':
                        section_count += 1
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.swing_point_label == 'LL' or swing_point.swing_point_label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.swing_point_label}. So uptrend started here.')
                    first_candle = swing_point
                    break
                swing_point_counter += 1
        if T_most_recent is not None:
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),
                    'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration)}
            logger.error(f'T_most_recent during prior series of swings: {T_most_recent}.')
            if T_most_recent < latest_T and section_count > 1:
                action_buy = False
        else:
            data = {'latest_T': str(latest_T),'section_count': str(section_count),}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data


class GannPointEightBuy(BaseStrategy):
    name="Gann's Buying point #8"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point', 'low_price', 'high_price').order_by('-datetime')
        swing_point_counter = 1
        latest_T = 0
        max_top = None
        max_variance_percent = 0.5  # A bottom must be within this many % of the last low price to count as a bottom.
        peak_percent_above_bottom = 3   # The peak must be at least this many % above the bottom
        bottoms = 0

        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'LL':
                    logger.error(f'Detected first swingpoint. LL')
                    last_candle = swing_point
                    last_low = swing_point.low_price
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH or LL. Stratey not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                bottoms = 1
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.swing_point_label == 'LH'  or swing_point.swing_point_label == 'HH':
                    # This is an upper swing point.
                    # If this is a new high above the bottom, save the high value.
                    if max_top is None or swing_point.high_price > max_top:
                        max_top = swing_point.high_price
                    if swing_point_counter == 2:
                        # This is the final peak. Save the High value.
                        last_high = swing_point.high_price
                    logger.error(
                        f'Found a prior {swing_point.swing_point_label}.')
                elif swing_point.swing_point_label == 'LL' or swing_point.swing_point_label == 'HL':
                    # This is potentially another bottom.
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    # Test if the bottom is within the threshold to be considered at the same level as the last low.
                    low_price_percent_variance = abs(swing_point.low_price - last_low)*100/last_low
                    if low_price_percent_variance < max_variance_percent:
                        logger.error(f'Low is within threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                        bottoms += 1
                        first_candle = swing_point
                    else:
                        logger.error(
                            f'Low is outside threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                        break
                swing_point_counter += 1
        # Check we have at least a double bottom and the peaks are at least significantly above the low
        if bottoms > 1 and max_top/last_low > (1 + (peak_percent_above_bottom/100)):
            bottom_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'bottoms': str(bottoms), 'bottom_duration' : str(bottom_duration)}
            logger.error(f'Multiple bottoms detected: {bottoms}.')
            # Temporarily set any double bottoms to be defined as a buy.
            #action_buy = True
            if latest_price.close_price > max_top:
                logger.error(f'Latest close ({latest_price.close_price}) is above max_top ({max_top}).')
                action_buy = True
        else:
            data = {}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data

class GannPointEightSell(BaseStrategy):
    name="Gann's Selling point #8"

    def check_criteria(self):
        data = {}
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_query = DailyPrice.objects.filter(ticker=self.ticker, swing_point_label__gt="").only('datetime', 'swing_point_label',
                                                                                                'candle_count_since_last_swing_point', 'low_price', 'high_price').order_by('-datetime')
        swing_point_counter = 1
        latest_T = 0
        min_bottom = None
        max_variance_percent = 0.5  # A bottom must be within this many % of the last low price to count as a bottom.
        peak_percent_below_top = 3   # The peak must be at least this many % above the bottom
        tops = 0

        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.datetime)}". swing_point_label:"{str(swing_point.swing_point_label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.swing_point_label == 'HH':
                    logger.error(f'Detected first swingpoint. HH')
                    last_candle = swing_point
                    last_high = swing_point.high_price
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point)
                tops = 1
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.swing_point_label == 'LL'  or swing_point.swing_point_label == 'HL':
                    # This is an lower swing point.
                    # If this is a new low below the top, save the low value.
                    if min_bottom is None or swing_point.low_price < min_bottom:
                        min_bottom = swing_point.low_price
                    if swing_point_counter == 2:
                        # This is the final peak. Save the High value.
                        last_low = swing_point.low_price
                    logger.error(
                        f'Found a prior {swing_point.swing_point_label}.')
                elif swing_point.swing_point_label == 'LH' or swing_point.swing_point_label == 'HH':
                    # This is potentially another top.
                    logger.error(f'Found a prior {swing_point.swing_point_label}.')
                    # Test if the top is within the threshold to be considered at the same level as the last high.
                    high_price_percent_variance = abs(swing_point.high_price - last_high)*100/last_high
                    if high_price_percent_variance < max_variance_percent:
                        logger.error(f'Low is within threshold {high_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                        tops += 1
                        first_candle = swing_point
                    else:
                        logger.error(
                            f'Low is outside threshold {high_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                        break
                swing_point_counter += 1
        # Check we have at least a double top and the peaks are at least significantly above the low
        if tops > 1 and min_bottom/last_high < (1 - (peak_percent_below_top/100)):
            top_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'tops': str(tops), 'top_duration' : str(top_duration)}
            logger.error(f'Multiple tops detected: {tops}.')
            # Temporarily set any double bottoms to be defined as a buy.
            #action_buy = True
            if latest_price.close_price < min_bottom:
                logger.error(f'Latest close ({latest_price.close_price}) is above max_top ({min_bottom}).')
                action_buy = False
        else:
            data = {}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data


from django.utils import timezone

def process_trading_opportunities():
    logger.error(f'Starting process_trading_opportunities()...')
    tickers = Ticker.objects.all()
    #tickers = Ticker.objects.filter(symbol="LUV")
    #strategies = [TAEStrategy, TwoPeriodCumRSI, DoubleSevens]  # List of strategy classes
    #strategies = [GannPointFourBuy, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell]  # List of strategy classes
    strategies = [GannPointFourBuy, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy, GannPointEightSell]  # List of strategy classes
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
