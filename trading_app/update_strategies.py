import logging

logging.basicConfig(level=logging.INFO)
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yfinance as yf
from time import sleep
from .models import Ticker, DailyPrice, FifteenMinPrice, FiveMinPrice, TickerCategory, TradingOpp, TradingStrategy, SwingPoint

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

def instance_price_difference_upswing(ticker, earlier_candle, later_candle=None):
    earlier_price = earlier_candle.low_price
    if later_candle is not None:
        later_price = later_candle.high_price
        price_difference = later_price - earlier_price
        logger.error(
            f'instance_price_difference_upswing. earlier_price = {earlier_price}, later_price = {later_price}, price_difference = {price_difference}')
    else:
        most_recent_price = latest_price = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
        price_difference = most_recent_price.high_price - earlier_price
        logger.error(
            f'instance_price_difference_upswing. earlier_price = {earlier_price}, most_recent_price.high_price = {most_recent_price.high_price}, price_difference = {price_difference}')
    return price_difference

def instance_price_difference_downswing(ticker, earlier_candle, later_candle=None):
    earlier_price = earlier_candle.high_price
    if later_candle is not None:
        later_price = later_candle.low_price
        price_difference = later_price - earlier_price
        logger.error(
            f'instance_price_difference_downswing. earlier_price = {earlier_price}, later_price = {later_price}, price_difference = {price_difference}')
    else:
        most_recent_price = latest_price = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
        price_difference = most_recent_price.low_price - earlier_price
        logger.error(
            f'instance_price_difference_downswing. earlier_price = {earlier_price}, most_recent_price.low_price = {most_recent_price.low_price}, price_difference = {price_difference}')
    return price_difference


class GannPointFourBuy2(BaseStrategy):
    name="Gann's Buying point #4"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            T_prev = []
            latest_T = 0
            recent_swing_points = []
            for swing_point in swing_point_query:
                # Check first is a LL or HH
                logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'LL':
                        logger.error(f'Detected first swingpoint. LL')
                        last_candle = swing_point.price_object
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'First swingpoint not LL. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.
                    latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                    most_recent_label = 'LL'
                    swing_point_counter += 1
                elif swing_point_counter >1:
                    if swing_point.label == 'LH' and most_recent_label == 'LL':
                        # Swing point is a high.
                        # Save the number of days that that it took to reach this swing point.
                        logger.error(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                        T_prev.append(swing_point.candle_count_since_last_swing_point)
                        most_recent_label = 'LH'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL':
                        logger.error(f'Found a prior {swing_point.label}.')
                        most_recent_label = 'LL'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' or swing_point.label == 'HL':
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        logger.error(f'Found a prior {swing_point.label}. So downtrend started here.')
                        first_candle = swing_point.price_object
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(T_prev) > 0:
                max_T = max(T_prev)
                if max_T < latest_T and len(T_prev) > 1 and latest_price.close_price > swing_point.price:
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
                final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)), 'count_T_prev': str(len(T_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'final_upswing_size' : str(final_upswing_size)} # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.error(f'Max T during prior series of swings: {max_T}.')
            else:
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
                action_buy = None
            logger.error(f'Latest T: {latest_T}.')
            logger.error(f'........')
            return action_buy, data
        except:
            print(f"Error in Gann #4 Selling for {ticker.symbol}: {e}")

class GannPointFourSell(BaseStrategy):
    name="Gann's Selling point #4"

    def check_criteria(self):
        data = {}
        action_buy = None
        swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_counter = 1
        existing_downtrend = None
        T_prev = []
        latest_T = 0
        recent_swing_points = []
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH':
                    logger.error(f'Detected first swingpoint. HH')
                    last_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                most_recent_label = 'HH'
                swing_point_counter += 1
            elif swing_point_counter >1:
                if swing_point.label == 'HL' and most_recent_label == 'HH':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    T_prev.append(swing_point.candle_count_since_last_swing_point)
                    most_recent_label = 'HL'
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'HH':
                    logger.error(f'Found a prior {swing_point.label}.')
                    most_recent_label = 'HH'
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'LL' or swing_point.label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.label}. So downtrend / uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if len(T_prev) > 0:
            max_T = max(T_prev)
            if max_T < latest_T and len(T_prev) > 1 and latest_price.close_price < swing_point.price:
                # The most recent downward rally is longer than the longest downward rally during the down trend.
                # And we have at least 2 sections of downward movement during the most recent upward trend.
                logger.error(f'Latest downswing LONGER than longest down swing during up trend. Strategy valid.')
                action_buy = False
            else:
                # The most recent downward rally is shorter than the longest downward rally during the down trend.
                logger.error(f'Latest downswing shorter than longest down swing during up trend. Strategy not valid.')
                action_buy = None
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            final_downswing_size = round((swing_point.price - latest_price.close_price) / swing_point.price, 3)
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)),
                    'count_T_prev': str(len(T_prev)),
                    'prior_trend_duration': str(prior_trend_duration),
                    'recent_swing_points': recent_swing_points,  # recent_swing_points not as a string as it gets removed and accessed if present.
                    'final_downswing_size': str(final_downswing_size)
                    }
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
        swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_counter = 1
        T_most_recent = None
        latest_T = 0
        section_count = 0
        recent_swing_points = []
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'LL':
                    logger.error(f'Detected first swingpoint. LL')
                    last_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not LL. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.label == 'LH':
                    # Swing point is a high on the down trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    #most_recent_swing_label = swing_point.label
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'LL':
                    logger.error(f'Found a prior {swing_point.label}.')
                    if swing_point.label == 'LL':
                        section_count += 1
                        #most_recent_swing_label = swing_point.swing_point_label
                        recent_swing_points.append(swing_point)
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.label == 'HH' or swing_point.label == 'HL':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.label}. So downtrend / uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if T_most_recent is not None and latest_price.close_price > swing_point.price:
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),
                    'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                    'final_upswing_size' : str(final_upswing_size)}
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
        swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_counter = 1
        T_most_recent = None
        latest_T = 0
        section_count = 0
        recent_swing_points = []
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH':
                    logger.error(f'Detected first swingpoint. HH')
                    last_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                most_recent_swing_point_label = 'HH'
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.label == 'HL':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.error(
                        f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                    most_recent_swing_point_label = 'HL'
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'HH':
                    logger.error(f'Found a prior {swing_point.label}.')
                    if swing_point.label == 'HH' and most_recent_swing_point_label == 'HL':
                        section_count += 1
                        recent_swing_points.append(swing_point)
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.label == 'LL' or swing_point.label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.error(f'Found a prior {swing_point.label}. So uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if T_most_recent is not None and latest_price.close_price < swing_point.price:
            final_downswing_size = round((swing_point.price - latest_price.close_price) / swing_point.price,3)
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration),
                    'recent_swing_points' : recent_swing_points, 'final_downswing_size' : str(final_downswing_size)}
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
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')
        swing_point_counter = 1
        latest_T = 0
        max_top = None
        max_variance_percent = 0.8  # A bottom must be within this many % of the last low price to count as a bottom.
        peak_percent_above_bottom = 3   # The peak must be at least this many % above the bottom to be coutned as a peak.
        bottoms = 0
        recent_swing_points = []
        if latest_price is not None:
            latest_close_price = latest_price.close_price
        else:
            # We have no price data for this ticker, so strategy cannot be detected.
            data = {}
            action_buy = None
            logger.error(f'Latest T: {latest_T}.')
            logger.error(f'........')
            return action_buy, data
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'LL' and latest_close_price > swing_point.price:
                    logger.error(f'Detected first swingpoint is LL and latest close price is higher.')
                    last_candle = swing_point.price_object
                    last_low = swing_point.price_object.low_price
                    recent_swing_points.append(swing_point)
                    most_recent_low_swing_point_candle = last_candle
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not LL. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
                most_recent_label = 'LL'
            elif swing_point_counter > 1:
                if swing_point.label == 'LH'  or swing_point.label == 'HH':
                    if (most_recent_label == 'LL' or most_recent_label == 'HL'):
                        # This is an upper swing point.
                        # If this is a new high above the bottom, save the high value.
                        if max_top is None or swing_point.price_object.high_price > max_top:
                            max_top = swing_point.price_object.high_price
                        #if swing_point_counter == 2:
                        #    # This is the final peak. Save the High value.
                        #    last_high = swing_point.price_object.high_price
                        recent_swing_points.append(swing_point)
                        most_recent_label = swing_point.label
                        first_candle = most_recent_low_swing_point_candle
                        logger.error(
                            f'Found a prior {swing_point.label} before a {most_recent_label}. Found a top.')
                        bottoms += 1
                    else:
                        logger.error(
                            f'Successive highs. Not creating / continuing valid bottoms pattern.')
                        break
                elif swing_point.label == 'LL' or swing_point.label == 'HL':
                    if (most_recent_label == 'LH' or most_recent_label == 'HH'):
                        # This is potentially another bottom.
                        logger.error(f'Found a prior {swing_point.label} after a {most_recent_label}.')
                        # Test if the bottom is within the threshold to be considered at the same level as the last low.
                        low_price_percent_variance = abs(swing_point.price_object.low_price - last_low)*100/last_low
                        recent_swing_points.append(swing_point)
                        most_recent_label = swing_point.label
                        if low_price_percent_variance < max_variance_percent:
                            logger.error(f'Low is within threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            most_recent_low_swing_point_candle = swing_point.price_object
                        else:
                            logger.error(
                                f'Low is outside threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            break
                    else:
                        logger.error(
                            f'Successive lows. Not creating / continuing valid bottoms pattern.')
                        break
                swing_point_counter += 1
        # Check we have at least a double bottom and the peaks are at least significantly above the low
        if bottoms > 1 and max_top/last_low > (1 + (peak_percent_above_bottom/100)):
            bottom_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'bottoms': str(bottoms), 'bottom_duration' : str(bottom_duration), 'recent_swing_points' : recent_swing_points}
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
        action_buy = None
        # Access the latest DailyPrice (or other relevant price model) for the ticker
        latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
        swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')
        swing_point_counter = 1
        latest_T = 0
        min_bottom = None
        max_variance_percent = 0.8  # A bottom must be within this many % of the last low price to count as a bottom.
        peak_percent_below_top = 3  # The peak must be at least this many % above the bottom to be coutned as a peak.
        tops = 0
        recent_swing_points = []
        if latest_price is not None:
            latest_close_price = latest_price.close_price
        else:
            # We have no price data for this ticker, so strategy cannot be detected.
            data = {}
            action_buy = None
            logger.error(f'Latest T: {latest_T}.')
            logger.error(f'........')
            return action_buy, data
        for swing_point in swing_point_query:
            # Check first is a HH
            logger.error(
                f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH' and latest_close_price < swing_point.price:
                    logger.error(f'Detected first swingpoint is HH and latest close price is lower.')
                    last_candle = swing_point.price_object
                    last_high = swing_point.price_object.high_price
                    recent_swing_points.append(swing_point)
                    most_recent_low_swing_point_candle = last_candle
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
                most_recent_label = 'HH'
            elif swing_point_counter > 1:
                if swing_point.label == 'HL' or swing_point.label == 'LL':
                    if (most_recent_label == 'HH' or most_recent_label == 'LH'):
                        # This is an lower swing point.
                        # If this is a new low below the top, save the low value.
                        if min_bottom is None or swing_point.price_object.low_price < min_bottom:
                            min_bottom = swing_point.price_object.low_price
                        recent_swing_points.append(swing_point)
                        most_recent_label = swing_point.label
                        first_candle = most_recent_low_swing_point_candle
                        logger.error(
                            f'Found a prior {swing_point.label} before a {most_recent_label}. Found a bottom.')
                        tops += 1
                    else:
                        logger.error(
                            f'Successive lows. Not creating / continuing valid peaks pattern.')
                        break
                elif swing_point.label == 'HH' or swing_point.label == 'LH':
                    if (most_recent_label == 'HL' or most_recent_label == 'LL'):
                        # This is potentially another top.
                        logger.error(f'Found a prior {swing_point.label} after a {most_recent_label}.')
                        # Test if the top is within the threshold to be considered at the same level as the last low.
                        high_price_percent_variance = abs(swing_point.price_object.high_price - last_high) * 100 / last_high
                        recent_swing_points.append(swing_point)
                        most_recent_label = swing_point.label
                        if high_price_percent_variance < max_variance_percent:
                            logger.error(
                                f'Low is within threshold {high_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            most_recent_low_swing_point_candle = swing_point.price_object
                        else:
                            logger.error(
                                f'Low is outside threshold {high_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            break
                    else:
                        logger.error(
                            f'Successive highs. Not creating / continuing valid bottoms pattern.')
                        break
                swing_point_counter += 1
        # Check we have at least a double top and the bottoms are at least significantly below the tops
        if tops > 1 and min_bottom / last_high > (1 - (peak_percent_below_top / 100)):
            top_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            data = {'latest_T': str(latest_T), 'tops': str(tops), 'top_duration': str(top_duration),
                    'recent_swing_points': recent_swing_points}
            logger.error(f'Multiple tops detected: {tops}.')
            if latest_price.close_price < min_bottom:
                logger.error(f'Latest close ({latest_price.close_price}) is below min_bottom ({min_bottom}).')
                action_buy = False
        else:
            data = {}
            action_buy = None
        logger.error(f'Latest T: {latest_T}.')
        logger.error(f'........')
        return action_buy, data

class GannPointThreeBuy(BaseStrategy):
    name="Gann's Buying point #3"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            P_prev = []
            larger_P = 0
            recent_swing_points = []
            for swing_point in swing_point_query:
                # Check first is a HL
                logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'HL':
                        logger.error(f'Detected first swingpoint. HL')
                        pullback_candle = swing_point.price_object
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'First swingpoint not HL. Strategy not valid.')
                        break
                    most_recent_label = 'HL'
                    swing_point_counter += 1
                elif swing_point_counter == 2:
                    if swing_point.label == 'HH':
                        latest_hh_candle = swing_point.price_object
                        retracement_P = instance_price_difference_upswing(self.ticker, pullback_candle,
                                                                          latest_hh_candle)
                        recent_swing_points.append(swing_point)
                        logger.error(f'Detected second swingpoint. HH. retracement_P: {retracement_P}')
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'Second swingpoint not HH. Strategy not valid.')
                        break

                    most_recent_label = 'HH'
                    swing_point_counter += 1
                elif swing_point_counter == 3:
                    if swing_point.label == 'LL':
                        lowpoint_candle = swing_point.price_object
                        latest_P = instance_price_difference_upswing(self.ticker, lowpoint_candle, latest_hh_candle)
                        recent_swing_points.append(swing_point)
                        logger.error(f'Detected third swingpoint. LL. latest_P: {latest_P}')
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'Third swingpoint not LL. Strategy not valid.')
                        break
                    most_recent_label = 'LL'
                    swing_point_counter += 1

                elif swing_point_counter >3:
                    if swing_point.label == 'LH' and most_recent_label == 'LL':
                        # Swing point is a high.
                        # Save the number of days that that it took to reach this swing point.
                        logger.error(f'Found a prior {swing_point.label}. ')
                        latest_lh_candle = swing_point.price_object
                        most_recent_label = 'LH'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL' and most_recent_label == 'LH':
                        most_recent_label = 'LL'
                        price_rise = instance_price_difference_upswing(self.ticker,swing_point.price_object, latest_lh_candle)
                        P_prev.append(price_rise)
                        logger.error(f'Found a prior {swing_point.label}. price_rise: {price_rise}')
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' or swing_point.label == 'HL':
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        first_candle = swing_point.price_object
                        downtrend_price_movement = instance_price_difference_upswing(self.ticker,lowpoint_candle, first_candle)
                        logger.error(f'Found a prior {swing_point.label}. So downtrend started here. downtrend_price_movement: {downtrend_price_movement}')
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(P_prev) > 0:
                max_P = max(P_prev)
                if max_P < latest_P and len(P_prev) > 2 and latest_price.close_price > pullback_candle.low_price:
                    logger.error(f'Strategy valid.')
                    action_buy = True
                else:
                    logger.error(f'Upswing price movement insufficient or 3+ sections not present in downswing. Strategy not valid.')
                    action_buy = None
                # Compute the days between the start and end of the down trend.
                prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=lowpoint_candle)
                secondary_upswing_size = round((latest_price.close_price - pullback_candle.low_price) / pullback_candle.low_price, 3) - 1
                initial_upswing_size = round(
                    (latest_hh_candle.high_price - lowpoint_candle.low_price) / lowpoint_candle.low_price, 3) - 1
                data = {'latest_P': str(latest_P), 'P_prev': str(P_prev), 'max_P': str(max_P), 'sections': str(len(P_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'secondary_upswing_size' : str(secondary_upswing_size), 'initial_upswing_size' : str(initial_upswing_size),
                        'retracement_P' : str(retracement_P), 'percent_retracement' : str(round(retracement_P*100/latest_P,1)),
                        'downtrend_price_movement' : str(downtrend_price_movement), 'initial_upswing_percent_of_downtrend' : str(round(initial_upswing_size*100/downtrend_price_movement,1))} # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.error(f'Max T during prior series of swings: {max_P}.')
            else:
                data = {}
                action_buy = None
            logger.error(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #3 Buying for {self.ticker.symbol}: {e}")

class GannPointThreeSell(BaseStrategy):
    name="Gann's Selling point #3"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            P_prev = []
            larger_P = 0
            recent_swing_points = []
            for swing_point in swing_point_query:
                # Check first is a HL
                logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'LH':
                        logger.error(f'Detected first swingpoint. LH')
                        pullback_candle = swing_point.price_object
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'First swingpoint not LH. Strategy not valid.')
                        break
                    most_recent_label = 'LH'
                    swing_point_counter += 1
                elif swing_point_counter == 2:
                    if swing_point.label == 'LL':
                        latest_ll_candle = swing_point.price_object
                        retracement_P = instance_price_difference_downswing(self.ticker, pullback_candle,
                                                                            latest_ll_candle)
                        logger.error(f'Detected second swingpoint. LL. retracement_P: {retracement_P}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'Second swingpoint not LL. Strategy not valid.')
                        break

                    most_recent_label = 'LL'
                    swing_point_counter += 1
                elif swing_point_counter == 3:
                    if swing_point.label == 'HH':
                        highpoint_candle = swing_point.price_object
                        latest_P = instance_price_difference_downswing(self.ticker, latest_ll_candle, highpoint_candle)
                        logger.error(f'Detected third swingpoint. HH. Price fall before retracement: {latest_P}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'Third swingpoint not HH. Strategy not valid.')
                        break
                    most_recent_label = 'HH'
                    swing_point_counter += 1

                elif swing_point_counter >3:
                    if swing_point.label == 'HL' and most_recent_label == 'HH':
                        # Swing point is a low.
                        logger.error(f'Found a prior {swing_point.label}. ')
                        latest_hl_candle = swing_point.price_object
                        most_recent_label = 'HL'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' and most_recent_label == 'HL':
                        most_recent_label = 'HH'
                        price_fall = instance_price_difference_downswing(self.ticker,latest_hl_candle, swing_point.price_object)
                        logger.error(f'Found a prior {swing_point.label}. Price fall: {price_fall}')
                        P_prev.append(price_fall)
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL' or swing_point.label == 'LH':
                        # This must be the start of the prior down trend.
                        # Stop checking further swing points.
                        first_candle = swing_point.price_object
                        uptrend_price_movement = instance_price_difference_upswing(self.ticker,first_candle,highpoint_candle)
                        logger.error(f'Found a prior {swing_point.label}. So downtrend started here. uptrend_price_movement : {uptrend_price_movement}')
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(P_prev) > 0:
                max_P = max(P_prev)
                if max_P < latest_P and len(P_prev) > 2 and latest_price.close_price < pullback_candle.high_price:
                    logger.error(f'Strategy valid.')
                    action_buy = False
                else:
                    logger.error(f'Downswing price movement insufficient or 3+ sections not present in upswing. Strategy not valid.')
                    action_buy = None
                # Compute the days between the start and end of the down trend.
                prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=highpoint_candle)
                secondary_upswing_size = round(((latest_price.close_price - pullback_candle.high_price )*100 / pullback_candle.high_price), 1)
                initial_downswing_size = round(((highpoint_candle.high_price - latest_ll_candle.high_price)*100 / highpoint_candle.high_price), 1)
                data = {'latest_P': str(latest_P), 'P_prev': str(P_prev), 'max_P': str(max_P), 'sections': str(len(P_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'secondary_upswing_size' : str(secondary_upswing_size), 'initial_downswing_size' : str(initial_downswing_size),
                        'retracement_P' : str(retracement_P), 'percent_retracement' : str(round(retracement_P*100/latest_P,1)),
                        'uptrend_price_movement' : str(uptrend_price_movement), 'initial_downswing_percent_of_uptrend' : str(round(initial_downswing_size*100/uptrend_price_movement,1))} # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.error(f'Max P during prior series of swings: {max_P}.')
            else:
                data = {}
                action_buy = None
            logger.error(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #3 Selling for {self.ticker.symbol}: {e}")

from django.utils import timezone

def process_trading_opportunities():
    logger.error(f'Starting process_trading_opportunities()...')
    tickers = Ticker.objects.all()
    #tickers = Ticker.objects.filter(symbol="LUV")
    #strategies = [TAEStrategy, TwoPeriodCumRSI, DoubleSevens]  # List of strategy classes
    strategies = [GannPointFourBuy2, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy, GannPointEightSell, GannPointThreeBuy, GannPointThreeSell]  # List of strategy classes


    try:
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
                    if 'recent_swing_points' in data:
                        recent_swing_points = data['recent_swing_points']
                        recent_swing_points_exist = True
                        del data['recent_swing_points']
                    else:
                        recent_swing_points_exist = False
                    if existing_tradingopp is not None:
                        logger.error(f'Existing TradingOpp being updated...')
                        # This Ticker / strategy exists as an active record. Increment the count.
                        existing_tradingopp.count += 1
                        # Update the metrics with the latest data e.g. current latest_T.
                        existing_tradingopp.metrics_snapshot = data
                        if recent_swing_points_exist == True:
                            for swing_point in recent_swing_points:
                                existing_tradingopp.swing_points.add(swing_point)
                        existing_tradingopp.save()
                    else:
                        # This Ticker / strategy is new.
                        logger.error(f'Creating new TradingOpp...')
                        # Create a new TradingOpp instance.
                        trading_opp = TradingOpp.objects.create(
                            ticker=ticker,
                            strategy=strategy_instance,
                            datetime_identified=timezone.now(),
                            metrics_snapshot=data, # Capture relevant metrics
                            count = 1,
                            action_buy = action_buy,
                        )
                        if recent_swing_points_exist == True:
                            for swing_point in recent_swing_points:
                                trading_opp.swing_points.add(swing_point)
                        trading_opp.save()
                else:
                    # The strategy is not valid for the ticker.
                    # Check if there was an active TradingOpp for this Ticker / strategy and set is_active=0
                    if existing_tradingopp is not None:
                        existing_tradingopp.is_active = False
                        existing_tradingopp.save()
        logger.error(f'Finished process_trading_opportunities().')
    except Exception as e:
        print(f"Error in process_trading_opportunities. Current ticker: {ticker.symbol}: {e}")

