import logging

logging.basicConfig(level=logging.INFO)
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yfinance as yf
from time import sleep
from .models import Ticker, DailyPrice, FifteenMinPrice, FiveMinPrice, TickerCategory, TradingOpp, TradingStrategy, SwingPoint, Params

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

def swing_point_price_difference(ticker, earlier_sp, later_sp=None):
    earlier_sp_price = earlier_sp.price
    if later_sp is not None:
        later_sp_price = later_sp.price
        price_difference = later_sp_price - earlier_sp_price
        logger.info(
            f'swing_point_price_difference. earlier_sp_price = {earlier_sp_price}, later_sp_price = {later_sp_price}, price_difference = {price_difference}')
    else:
        most_recent_price = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
        price_difference = most_recent_price.close_price - earlier_sp_price
        logger.info(
            f'swing_point_price_difference. earlier_sp_price = {earlier_sp_price}, most_recent_price.close_price = {most_recent_price.close_price}, price_difference = {price_difference}')
    return price_difference

class GannPointOneBuy(BaseStrategy):
    name="Gann's Buying point #1"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            prior_hl_sp = None
            recent_swing_points = []
            for swing_point in swing_point_query:
                # Check first is a HL
                logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". swing_point_counter:"{swing_point_counter}"')
                if swing_point_counter == 1:
                    if swing_point.label == 'HL':
                        last_sp = swing_point
                        last_sp_price = last_sp.price
                        logger.info(f'Detected first HL swingpoint. Price: {last_sp_price}. Latest close price: {latest_price.close_price}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'First swingpoint not HL. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.

                    most_recent_label = 'HL'
                elif swing_point_counter == 2:
                    logger.info(f'swing_point_counter == 2. most_recent_label: {most_recent_label}.')
                    if swing_point.label == 'HH' and most_recent_label == 'HL':
                        # Swing point is a HH.
                        most_recent_label = 'HH'
                        peak_sp = swing_point
                        peak_sp_price = peak_sp.price
                        logger.info(f'Found a prior HH. Price: {peak_sp_price}.')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Second swingpoint not HH. Strategy not valid.')
                        break
                elif swing_point_counter == 3:
                    if swing_point.label == 'HL' and most_recent_label == 'HH':
                        # Swing point is a previous HL.
                        most_recent_label = 'HL'
                        prior_hl_sp = swing_point
                        recent_swing_points.append(swing_point)
                        prior_hl_price = prior_hl_sp.price
                        logger.info(f'Found a prior HL. prior_hl_price: {prior_hl_price}. So swing points match the strategy.')
                    else:
                        logger.info(f'Third swingpoint not HL. Strategy not valid.')
                    break
                swing_point_counter += 1
            if prior_hl_sp is not None and latest_price.close_price > last_sp_price:
                duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                     later_candle=latest_price)
                action_buy = True
                elapsed__duration = instance_difference_count(self.ticker, prior_hl_sp.price_object, later_candle=last_sp.price_object)
                sp_price_diff_vs_prior_high =  last_sp_price-prior_hl_price
                price_retracement = last_sp_price - peak_sp_price
                retracement_as_percent = price_retracement * 100 / (peak_sp_price - prior_hl_price)
                rise_after_retracement = latest_price.close_price - last_sp_price
                rise_after_retracement_percent_of_retracement = rise_after_retracement * 100 / (-price_retracement)
                if rise_after_retracement / -price_retracement < 0.5:
                    data = {'sp_price_diff_vs_prior_high': str(sp_price_diff_vs_prior_high), 'price_retracement': str(price_retracement),
                            'retracement_as_percent': str(round(retracement_as_percent,1)), 'elapsed__duration': str(elapsed__duration),
                            'rise_after_retracement' : str(rise_after_retracement), 'recent_swing_points' : recent_swing_points,
                            'rise_after_retracement_percent_of_retracement' : str(round(rise_after_retracement_percent_of_retracement,1)),
                            'duration_after_latest_sp' : str(duration_after_latest_sp), 'last_sp_price' : str(last_sp_price),
                            'latest_price.close_price' : str(latest_price.close_price)} # recent_swing_points not as a string as it gets removed and accessed if present.
                    action_buy = True
                else:
                    # Strategy is not valid
                    logger.info(f'Price already rose > 50% of retracement. Strategy not valid.')
                    action_buy = None
            else:
                # Strategy is not valid
                action_buy = None
            logger.info(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #1 Buying for {self.ticker.symbol}: {e}")

class GannPointOneSell(BaseStrategy):
    name="Gann's Selling point #1"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            prior_lh_sp = None
            recent_swing_points = []
            for swing_point in swing_point_query:
                # Check first is a LH
                logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". swing_point_counter:"{swing_point_counter}"')
                if swing_point_counter == 1:
                    if swing_point.label == 'LH':
                        last_sp = swing_point
                        last_sp_price = last_sp.price
                        logger.info(f'Detected first LH swingpoint. Price: {last_sp_price}. Latest close price: {latest_price.close_price}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'First swingpoint not LH. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.

                    most_recent_label = 'LH'
                elif swing_point_counter == 2:
                    logger.info(f'swing_point_counter == 2. most_recent_label: {most_recent_label}.')
                    if swing_point.label == 'LL' and most_recent_label == 'LH':
                        # Swing point is a LL.
                        most_recent_label = 'LL'
                        trough_sp = swing_point
                        trough_sp_price = trough_sp.price
                        logger.info(f'Found a prior LL. Price: {trough_sp_price}.')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Second swingpoint not LL. Strategy not valid.')
                        break
                elif swing_point_counter == 3:
                    if swing_point.label == 'LH' and most_recent_label == 'LL':
                        # Swing point is a previous HL.
                        most_recent_label = 'LH'
                        prior_lh_sp = swing_point
                        recent_swing_points.append(swing_point)
                        prior_lh_price = prior_lh_sp.price
                        logger.info(f'Found a prior LH. prior_lh_price: {prior_lh_price}. So swing points match the strategy.')
                    else:
                        logger.info(f'Third swingpoint not LH. Strategy not valid.')
                    break
                swing_point_counter += 1
            if prior_lh_sp is not None and latest_price.close_price < last_sp_price:
                duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                     later_candle=latest_price)

                elapsed__duration = instance_difference_count(self.ticker, prior_lh_sp.price_object, later_candle=last_sp.price_object)

                sp_price_diff_vs_prior_low =  last_sp_price-prior_lh_price
                price_retracement = last_sp_price - trough_sp_price
                retracement_as_percent = price_retracement * 100 / (prior_lh_price - trough_sp_price)
                fall_after_retracement = latest_price.close_price - last_sp_price
                fall_after_retracement_percent_of_retracement = fall_after_retracement * 100 / (-price_retracement)
                if fall_after_retracement / -price_retracement < 0.5:
                    data = {'sp_price_diff_vs_prior_low': str(sp_price_diff_vs_prior_low), 'price_retracement': str(price_retracement),
                            'retracement_as_percent': str(round(retracement_as_percent,1)), 'elapsed__duration': str(elapsed__duration),
                            'fall_after_retracement' : str(fall_after_retracement), 'recent_swing_points' : recent_swing_points,  # recent_swing_points not as a string as it gets removed and accessed if present.
                            'fall_after_retracement_percent_of_retracement' : str(round(fall_after_retracement_percent_of_retracement,1)),
                            'duration_after_latest_sp' : str(duration_after_latest_sp)}
                    action_buy = False
                else:
                    # Strategy is not valid
                    logger.info(f'PRice fell >50% of retracement. Strategy not valid.')
                    action_buy = None
            else:
                # Strategy is not valid
                action_buy = None
            logger.info(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #1 Selling for {self.ticker.symbol}: {e}")

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
                logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'HL':
                        logger.info(f'Detected first swingpoint. HL')
                        pullback_sp = swing_point
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'First swingpoint not HL. Strategy not valid.')
                        break
                    most_recent_label = 'HL'
                    swing_point_counter += 1
                elif swing_point_counter == 2:
                    if swing_point.label == 'HH':
                        latest_hh_sp = swing_point
                        retracement_P = swing_point_price_difference(self.ticker, pullback_sp,latest_hh_sp)
                        recent_swing_points.append(swing_point)
                        logger.info(f'Detected second swingpoint. HH. retracement_P: {retracement_P}')
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Second swingpoint not HH. Strategy not valid.')
                        break

                    most_recent_label = 'HH'
                    swing_point_counter += 1
                elif swing_point_counter == 3:
                    if swing_point.label == 'LL':
                        lowpoint_sp = swing_point
                        latest_P = swing_point_price_difference(self.ticker, lowpoint_sp, latest_hh_sp)
                        recent_swing_points.append(swing_point)
                        logger.info(f'Detected third swingpoint. LL. latest_P: {latest_P}')
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Third swingpoint not LL. Strategy not valid.')
                        break
                    most_recent_label = 'LL'
                    swing_point_counter += 1

                elif swing_point_counter >3:
                    if swing_point.label == 'LH' and most_recent_label == 'LL':
                        # Swing point is a high.
                        # Save the number of days that that it took to reach this swing point.
                        logger.info(f'Found a prior {swing_point.label}. ')
                        latest_lh_sp = swing_point
                        most_recent_label = 'LH'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL' and most_recent_label == 'LH':
                        most_recent_label = 'LL'
                        price_rise = swing_point_price_difference(self.ticker,swing_point, latest_lh_sp)
                        P_prev.append(price_rise)
                        logger.info(f'Found a prior {swing_point.label}. price_rise: {price_rise}')
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' or swing_point.label == 'HL':
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        first_sp = swing_point
                        downtrend_price_movement = swing_point_price_difference(self.ticker,first_sp, lowpoint_sp)
                        logger.info(f'Found a prior {swing_point.label}. So downtrend started here. downtrend_price_movement: {downtrend_price_movement}')
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(P_prev) > 0:
                max_P = max(P_prev)
                if max_P < abs(latest_P) and len(P_prev) > 2 and latest_price.close_price > pullback_sp.price:
                    logger.info(f'Strategy valid.')
                    action_buy = True
                else:
                    logger.info(f'Upswing price movement insufficient or 3+ sections not present in downswing. Strategy not valid.')
                    action_buy = None
                # Compute the days between the start and end of the down trend.
                prior_trend_duration = instance_difference_count(self.ticker, first_sp.price_object, later_candle=lowpoint_sp.price_object)
                duration_after_latest_sp = instance_difference_count(self.ticker, pullback_sp.price_object,
                                                                     later_candle=latest_price)
                data = {'latest_P': str(latest_P), 'P_prev': str(P_prev), 'max_P': str(max_P), 'sections': str(len(P_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'retracement_P' : str(retracement_P), 'downtrend_price_movement' : str(downtrend_price_movement),
                        'duration_after_latest_sp' : str(duration_after_latest_sp)} # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.info(f'Max T during prior series of swings: {max_P}.')
            else:
                data = {}
                action_buy = None
            logger.info(f'........')
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
                logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'LH':
                        logger.info(f'Detected first swingpoint. LH')
                        pullback_sp = swing_point
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'First swingpoint not LH. Strategy not valid.')
                        break
                    most_recent_label = 'LH'
                    swing_point_counter += 1
                elif swing_point_counter == 2:
                    if swing_point.label == 'LL':
                        latest_ll_sp = swing_point
                        retracement_P = swing_point_price_difference(self.ticker, latest_ll_sp, pullback_sp)
                        logger.info(f'Detected second swingpoint. LL. retracement_P: {retracement_P}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Second swingpoint not LL. Strategy not valid.')
                        break

                    most_recent_label = 'LL'
                    swing_point_counter += 1
                elif swing_point_counter == 3:
                    if swing_point.label == 'HH':
                        highpoint_sp = swing_point
                        latest_P = swing_point_price_difference(self.ticker, highpoint_sp, latest_ll_sp)
                        logger.info(f'Detected third swingpoint. HH. Price fall before retracement: {latest_P}')
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'Third swingpoint not HH. Strategy not valid.')
                        break
                    most_recent_label = 'HH'
                    swing_point_counter += 1

                elif swing_point_counter >3:
                    if swing_point.label == 'HL' and most_recent_label == 'HH':
                        # Swing point is a low.
                        logger.info(f'Found a prior {swing_point.label}. ')
                        latest_hl_sp = swing_point
                        most_recent_label = 'HL'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' and most_recent_label == 'HL':
                        most_recent_label = 'HH'
                        price_fall = swing_point_price_difference(self.ticker,swing_point,latest_hl_sp )
                        logger.info(f'Found a prior {swing_point.label}. Price fall: {price_fall}')
                        P_prev.append(float(price_fall))
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL' or swing_point.label == 'LH':
                        # This must be the start of the prior down trend.
                        # Stop checking further swing points.
                        first_sp = swing_point
                        uptrend_price_movement = swing_point_price_difference(self.ticker,first_sp,highpoint_sp)
                        logger.info(f'Found a prior {swing_point.label}. So downtrend started here. uptrend_price_movement : {uptrend_price_movement}')
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(P_prev) > 0:
                min_P = min(P_prev)
                if min_P > latest_P and len(P_prev) > 2 and latest_price.close_price < pullback_sp.price:
                    logger.info(f'Strategy valid.')
                    action_buy = False
                else:
                    logger.info(f'Downswing price movement insufficient or 3+ sections not present in upswing. Strategy not valid.')
                    action_buy = None
                # Compute the days between the start and end of the down trend.
                prior_trend_duration = instance_difference_count(self.ticker, first_sp.price_object, later_candle=highpoint_sp.price_object)
                duration_after_latest_sp = instance_difference_count(self.ticker, pullback_sp.price_object,
                                                                     later_candle=latest_price)
                data = {'latest_P': str(latest_P), 'P_prev': str(P_prev), 'min_P': str(min_P), 'sections': str(len(P_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'retracement_P' : str(retracement_P), 'uptrend_price_movement' : str(uptrend_price_movement),
                        'duration_after_latest_sp' : str(duration_after_latest_sp)} # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.info(f'Min P during prior series of swings: {min_P}.')
            else:
                data = {}
                action_buy = None
            logger.info(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #3 Selling for {self.ticker.symbol}: {e}")

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
                logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'LL':
                        logger.info(f'Detected first swingpoint. LL')
                        last_candle = swing_point.price_object
                        last_sp = swing_point
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.info(f'First swingpoint not LL. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.
                    latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                    most_recent_label = 'LL'
                    swing_point_counter += 1
                elif swing_point_counter >1:
                    if swing_point.label == 'LH' and most_recent_label == 'LL':
                        # Swing point is a high.
                        # Save the number of days that that it took to reach this swing point.
                        logger.info(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                        T_prev.append(swing_point.candle_count_since_last_swing_point)
                        most_recent_label = 'LH'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'LL':
                        logger.info(f'Found a prior {swing_point.label}.')
                        most_recent_label = 'LL'
                        recent_swing_points.append(swing_point)
                    elif swing_point.label == 'HH' or swing_point.label == 'HL':
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        logger.info(f'Found a prior {swing_point.label}. So downtrend started here.')
                        first_candle = swing_point.price_object
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if len(T_prev) > 0:
                max_T = max(T_prev)
                if max_T < latest_T and len(T_prev) > 1:
                    # The most recent upward rally is longer than the longest upward rally during the down trend.
                    # And we have had at least 2 sections of upward movement during the down trend.
                    logger.info(f'Latest upswing LONGER than longest up swing during down trend. Strategy valid.')
                    action_buy = True
                else:
                    # The most recent upward rally is shorter than the longest upward rally during the down trend.
                    logger.info(f'Latest upswing shorter than longest up swing during down trend. Strategy not valid.')
                    action_buy = None
                # Compute the days between the start and end of the down trend.
                prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
                final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
                duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                     later_candle=latest_price)
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)), 'count_T_prev': str(len(T_prev)),
                        'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                        'final_upswing_size' : str(final_upswing_size), 'duration_after_latest_sp' : str(duration_after_latest_sp),
                        } # recent_swing_points not as a string as it gets removed and accessed if present.
                logger.info(f'Max T during prior series of swings: {max_T}.')
            else:
                data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
                action_buy = None
            logger.info(f'Latest T: {latest_T}.')
            logger.info(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #4 Selling for {self.ticker.symbol}: {e}")

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
            logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH':
                    logger.info(f'Detected first swingpoint. HH')
                    last_candle = swing_point.price_object
                    last_sp = swing_point
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.info(f'First swingpoint not HH. Strategy not valid.')
                    break
                # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                most_recent_label = 'HH'
                swing_point_counter += 1
            elif swing_point_counter >1:
                if swing_point.label == 'HL' and most_recent_label == 'HH':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.info(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    T_prev.append(swing_point.candle_count_since_last_swing_point)
                    most_recent_label = 'HL'
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'HH':
                    logger.info(f'Found a prior {swing_point.label}.')
                    most_recent_label = 'HH'
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'LL' or swing_point.label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.info(f'Found a prior {swing_point.label}. So downtrend / uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if len(T_prev) > 0:
            max_T = max(T_prev)
            if max_T < latest_T and len(T_prev) > 1:
                # The most recent downward rally is longer than the longest downward rally during the down trend.
                # And we have at least 2 sections of downward movement during the most recent upward trend.
                logger.info(f'Latest downswing LONGER than longest down swing during up trend. Strategy valid.')
                action_buy = False

            else:
                # The most recent downward rally is shorter than the longest downward rally during the down trend.
                logger.info(f'Latest downswing shorter than longest down swing during up trend. Strategy not valid.')
                action_buy = None
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                 later_candle=latest_price)
            final_downswing_size = round((swing_point.price - latest_price.close_price) / swing_point.price, 3)
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev), 'max_T': str(max(T_prev)),
                    'count_T_prev': str(len(T_prev)),
                    'prior_trend_duration': str(prior_trend_duration),
                    'recent_swing_points': recent_swing_points,  # recent_swing_points not as a string as it gets removed and accessed if present.
                    'final_downswing_size': str(final_downswing_size),
                    'duration_after_latest_sp' : str(duration_after_latest_sp),
                    }
            logger.info(f'Max T during prior series of swings: {max_T}.')
        else:
            data = {'latest_T': str(latest_T), 'T_prev': str(T_prev)}
            action_buy = None
        logger.info(f'Latest T: {latest_T}.')
        logger.info(f'........')
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
        lh_prices = []
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'LL':
                    logger.info(f'Detected first swingpoint. LL')
                    last_candle = swing_point.price_object
                    last_sp = swing_point
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.info(f'First swingpoint not LL. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.label == 'LH':
                    # Swing point is a high on the down trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.info(f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    #most_recent_swing_label = swing_point.label
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                    lh_prices.append(float(swing_point.price_object.high_price))
                    recent_swing_points.append(swing_point)
                elif swing_point.label == 'LL':
                    logger.info(f'Found a prior {swing_point.label}.')
                    if swing_point.label == 'LL':
                        section_count += 1
                        most_recent_duration = swing_point.candle_count_since_last_swing_point
                        #most_recent_swing_label = swing_point.swing_point_label
                        recent_swing_points.append(swing_point)
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.label == 'HH' or swing_point.label == 'HL':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.info(f'Found a prior {swing_point.label}. So downtrend / uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if T_most_recent is not None:
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                 later_candle=latest_price)
            final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
            most_recent_lh_price = lh_prices[0]
            if latest_price.close_price > most_recent_lh_price:
                confirmed = True
                when_confirmed = ""
            else:
                confirmed = False
                when_confirmed = f'When price rises above {most_recent_lh_price}.'

            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),
                    'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration), 'recent_swing_points' : recent_swing_points,
                    'final_upswing_size' : str(final_upswing_size), 'duration_after_latest_sp' : str(duration_after_latest_sp), 'most_recent_lh_price' : most_recent_lh_price,
                    'confirmed': confirmed, 'when_confirmed': when_confirmed}
            logger.info(f'T_most_recent during prior series of swings: {T_most_recent}.')
            if T_most_recent < latest_T and section_count > 1:
                action_buy = True
        else:
            data = {'latest_T': str(latest_T),'section_count': str(section_count),}
            action_buy = None
        logger.info(f'Latest T: {latest_T}.')
        logger.info(f'........')
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
        hl_prices = []
        for swing_point in swing_point_query:
            # Check first is a LL
            logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH':
                    logger.info(f'Detected first swingpoint. HH')
                    last_candle = swing_point.price_object
                    last_sp = swing_point
                    recent_swing_points.append(swing_point)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.info(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                most_recent_swing_point_label = 'HH'
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
            elif swing_point_counter > 1:
                if swing_point.label == 'HL':
                    # Swing point is a low on the up trend.
                    # Save the number of days that that it took to reach this swing point.
                    logger.info(
                        f'Found a prior {swing_point.label}. Days to this point = {swing_point.candle_count_since_last_swing_point}')
                    # Only save the most recent elapsed time.
                    most_recent_duration = swing_point.candle_count_since_last_swing_point
                    most_recent_swing_point_label = 'HL'
                    recent_swing_points.append(swing_point)
                    hl_prices.append(float(swing_point.price_object.low_price))
                elif swing_point.label == 'HH':
                    logger.info(f'Found a prior {swing_point.label}.')
                    if swing_point.label == 'HH' and most_recent_swing_point_label == 'HL':
                        section_count += 1
                        recent_swing_points.append(swing_point)
                        if T_most_recent is None:
                            T_most_recent = most_recent_duration
                elif swing_point.label == 'LL' or swing_point.label == 'LH':
                    # This must be the start of the prior down / up trend.
                    # Stop checking further swing points.
                    logger.info(f'Found a prior {swing_point.label}. So uptrend started here.')
                    first_candle = swing_point.price_object
                    recent_swing_points.append(swing_point)
                    break
                swing_point_counter += 1
        if T_most_recent is not None:
            final_downswing_size = round((swing_point.price - latest_price.close_price) / swing_point.price,3)
            prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                 later_candle=latest_price)
            min_hl_price = min(hl_prices)
            if latest_price.close_price < min_hl_price:
                confirmed = True
                when_confirmed = ""
            else:
                confirmed = False
                when_confirmed = f'When price falls below {min_hl_price}.'

            data = {'latest_T': str(latest_T), 'T_most_recent': str(T_most_recent),'section_count': str(section_count), 'prior_trend_duration' : str(prior_trend_duration),
                    'recent_swing_points' : recent_swing_points, 'final_downswing_size' : str(final_downswing_size), 'min_hl_price' : min_hl_price,
                    'duration_after_latest_sp' : str(duration_after_latest_sp), 'confirmed' : str(confirmed), 'when_confirmed' : when_confirmed}
            logger.info(f'T_most_recent during prior series of swings: {T_most_recent}.')
            if T_most_recent < latest_T and section_count > 1:
                action_buy = False
        else:
            data = {'latest_T': str(latest_T),'section_count': str(section_count),}
            action_buy = None
        logger.info(f'Latest T: {latest_T}.')
        logger.info(f'........')
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
        bottom_dates = []
        if latest_price is not None:
            latest_close_price = latest_price.close_price
        else:
            # We have no price data for this ticker, so strategy cannot be detected.
            data = {}
            action_buy = None
            logger.info(f'Latest T: {latest_T}.')
            logger.info(f'........')
            return action_buy, data
        for swing_point in swing_point_query:
            # Check first is a LL or HH
            logger.info(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'LL' and latest_close_price > swing_point.price:
                    logger.info(f'Detected first swingpoint is LL and latest close price is higher.')
                    last_candle = swing_point.price_object
                    last_low = last_candle.low_price
                    last_sp = swing_point
                    recent_swing_points.append(swing_point)
                    most_recent_low_swing_point_candle = last_candle
                    most_recent_low = float(swing_point.price_object.low_price)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.info(f'First swingpoint not LL. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
                most_recent_date = swing_point.date
                most_recent_label = 'LL'
                bottom_prices = []
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
                        logger.info(
                            f'Found a prior {swing_point.label} before a {most_recent_label}. Found a top.')
                        bottom_dates.append(most_recent_date.isoformat())
                        bottoms += 1
                        bottom_prices.append(most_recent_low)
                    else:
                        logger.info(
                            f'Successive highs. Not creating / continuing valid bottoms pattern.')
                        break
                elif swing_point.label == 'LL' or swing_point.label == 'HL':
                    if (most_recent_label == 'LH' or most_recent_label == 'HH'):
                        # This is potentially another bottom.
                        logger.info(f'Found a prior {swing_point.label} after a {most_recent_label}.')
                        # Test if the bottom is within the threshold to be considered at the same level as the last low.
                        low_price_percent_variance = abs(swing_point.price_object.low_price - last_low)*100/last_low
                        recent_swing_points.append(swing_point)
                        most_recent_low = float(swing_point.price_object.low_price)
                        most_recent_label = swing_point.label
                        if low_price_percent_variance < max_variance_percent:
                            logger.info(f'Low is within threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            most_recent_low_swing_point_candle = swing_point.price_object
                        else:
                            logger.info(
                                f'Low is outside threshold {low_price_percent_variance} vs max_variance_percent of {max_variance_percent}.')
                            break
                    else:
                        logger.info(
                            f'Successive lows. Not creating / continuing valid bottoms pattern.')
                        break
                swing_point_counter += 1
        # Check we have at least a double bottom and the peaks are at least significantly above the low
        if bottoms > 1 and max_top/last_low > (1 + (peak_percent_above_bottom/100)):
            bottom_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
            duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                 later_candle=latest_price)
            bottom_price = min(bottom_prices)
            if latest_close_price > max_top:
                confirmed = True
                when_confirmed = ""
            else:
                confirmed = False
                when_confirmed = f'Confirmed when price exceeds {max_top}.'
            data = {'latest_T': str(latest_T), 'bottoms': str(bottoms), 'bottom_duration' : str(bottom_duration), 'recent_swing_points' : recent_swing_points,
                    'duration_after_latest_sp' : str(duration_after_latest_sp), 'confirmed' : str(confirmed), 'when_confirmed' : when_confirmed,
                    'bottom_price' : bottom_price, 'bottoms_dates' : str(bottom_dates)}
            logger.info(f'Multiple bottoms detected: {bottoms}.')
            # Temporarily set any double bottoms to be defined as a buy.
            #action_buy = True
            if latest_price.close_price > max_top:
                logger.info(f'Latest close ({latest_price.close_price}) is above max_top ({max_top}).')
                action_buy = True
        else:
            data = {}
            action_buy = None
        logger.info(f'Latest T: {latest_T}.')
        logger.info(f'........')
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
        tops_dates = []
        top_prices = []
        recent_swing_points = []
        if latest_price is not None:
            latest_close_price = latest_price.close_price
        else:
            # We have no price data for this ticker, so strategy cannot be detected.
            data = {}
            action_buy = None
            logger.info(f'Latest T: {latest_T}.')
            logger.info(f'........')
            return action_buy, data
        for swing_point in swing_point_query:
            # Check first is a HH
            logger.error(
                f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}". candle_count_since_last_swing_point:"{str(swing_point.candle_count_since_last_swing_point)}".')
            if swing_point_counter == 1:
                if swing_point.label == 'HH' and latest_close_price < swing_point.price:
                    logger.error(f'Detected first swingpoint is HH and latest close price is lower.')
                    last_candle = swing_point.price_object
                    last_high = last_candle.high_price
                    last_sp = swing_point
                    recent_swing_points.append(swing_point)
                    most_recent_low_swing_point_candle = last_candle
                    most_recent_price = float(swing_point.price_object.high_price)
                else:
                    # This strategy cannot be true. End review of swing points.
                    logger.error(f'First swingpoint not HH. Strategy not valid.')
                    break
                    # Now need to determine the elapsed days since this LL or HH.
                latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                swing_point_counter += 1
                most_recent_date = swing_point.date
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
                        tops_dates.append(most_recent_date.isoformat())
                        top_prices.append(most_recent_price)
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
                        most_recent_price = float(swing_point.price_object.high_price)
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
            duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                                                                 later_candle=latest_price)
            top_price = max(top_prices)
            if latest_close_price < min_bottom:
                confirmed = True
                when_confirmed = ""
            else:
                confirmed = False
                when_confirmed = f'Confirmed when price falls below {min_bottom}.'

            data = {'latest_T': str(latest_T), 'tops': str(tops), 'top_duration': str(top_duration),
                    'recent_swing_points': recent_swing_points, 'duration_after_latest_sp' : str(duration_after_latest_sp),
                    'confirmed' : str(confirmed), 'when_confirmed' : when_confirmed, 'tops_dates' : str(tops_dates), 'top_price' : str(top_price)}
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

class GannPointNineBuy(BaseStrategy):
    name="Gann's Buying point #9"

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
            sections = 0
            pattern_detected = False
            most_recent_hh_price = None
            for swing_point in swing_point_query:
                # Check first is a HL
                logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'HL':
                        logger.error(f'Detected first swingpoint. HL')
                        last_sp = swing_point
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'First swingpoint not HL. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.
                    #latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                    most_recent_label = 'HL'
                    swing_point_counter += 1
                elif swing_point_counter >1:
                    if swing_point.label == 'HH' and most_recent_label == 'HL':
                        # Swing point is a high.
                        # Save the number of days that that it took to reach this swing point.
                        logger.error(f'Found a prior {swing_point.label}.')
                        #T_prev.append(swing_point.candle_count_since_last_swing_point)
                        most_recent_label = 'HH'
                        recent_swing_points.append(swing_point)
                        sections += 1
                        if most_recent_hh_price is None:
                            most_recent_hh_price = swing_point.price
                    elif swing_point.label == 'HL' and most_recent_label == 'HH':
                        logger.error(f'Found a prior {swing_point.label}. Another peak in the sequence.')
                        most_recent_label = 'HH'
                        recent_swing_points.append(swing_point)
                    else :
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        logger.error(f'Found a prior {swing_point.label}. Sections: {sections}.')
                        first_sp = swing_point
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if sections > 0:
                # At least 2 sections in most recent trend. So should analyse price movements from most recent swingpoint to most recent price
                # Does the most recent HH get broken?
                # Yes, then do we have 2 down candles followed by up candle.
                # Then check we have no close with a lower low.
                logger.error(f'Found sufficient sections to analyse recent price action.')
                # Get the candles to be analysed.
                prices = DailyPrice.objects.filter(ticker=self.ticker, datetime__gt=last_sp.price_object.datetime).order_by('datetime')
                hh_price_exceeded = False
                # Initialize variables
                prev_high_price = None
                prev_low_price = None
                pattern = []  # To track the pattern (higher, lower, lower, higher)
                individual_candles = []
                duration_to_start = 0
                for price in prices:
                    print(price.datetime, 'most_recent_hh_price:', most_recent_hh_price, 'price.close_price:', price.close_price)
                    duration_to_start += 1
                    if hh_price_exceeded == False and price.close_price > most_recent_hh_price:
                        # price has closed above the previous HH swingpoint.
                        hh_price_exceeded = True
                        prev_low_price = price.low_price
                        logger.error(f'Price has exceeded the previous HH. Checking individual price candles.')
                    elif hh_price_exceeded == True:
                        # Price exceeded the prior HH. Start looking for the pattern.
                        #if price.high_price > prev_high_price: and price.low_price > prev_low_price:
                        print(price.datetime, 'prev_low_price:', prev_low_price, 'price.low_price:',
                              price.low_price)
                        if prev_low_price is None or price.low_price > prev_low_price:
                            # Higher candle
                            if len(pattern) == 3:
                                logger.error(f"Continuous H still true...")
                                pattern_detected = True
                                individual_candles.append({'datetime': price.datetime.isoformat(),
                                                                        'low_price': float(price.low_price), 'high_price': float(price.high_price), 'colour' : 'green' })
                                ind_candle_count += 1
                            elif len(pattern) < 3:
                                # Expected a lower candle, got a higher.
                                pattern_detected = False
                                logger.error(f"Got Higher, expected a lower. Restarting search for pattern.")
                                start_candle = {}
                                individual_candles = []
                                ind_candle_count = 0
                                pattern = []
                            if len(pattern) == 0:
                                pattern = ['higher']
                                logger.error(f'First H found... pattern: {pattern}')
                                # Save current candle as the peak in case the pattern is found
                                start_candle = new_start
                                individual_candles.append({'datetime': price.datetime.isoformat(),
                                                                        'low_price': float(price.low_price), 'high_price': float(price.high_price), 'colour' : 'green' })
                                ind_candle_count += 1
                            prev_low_price = price.low_price
                        elif price.low_price < prev_low_price:
                            # Lower candle
                            if (len(pattern) == 1 and pattern[0] == 'higher') or (len(pattern) == 2 and pattern[1] == 'lower'):
                                pattern.append('lower')
                                logger.error(f'L found... pattern: {pattern}')
                                # Save current candle as might be the 2-day retracement candle.
                                individual_candles.append({'datetime': price.datetime.isoformat(),
                                                                        'low_price': float(price.low_price), 'high_price': float(price.high_price), 'colour' : 'red' })
                                ind_candle_count += 1
                            elif pattern_detected:
                                pattern_detected = False
                                logger.error(f"Found L when expecting continuous H's. Pattern broken.")
                                start_candle = {}
                                individual_candles = []
                                ind_candle_count = 0
                                pattern = []
                            prev_low_price = price.low_price
                        else:
                            # Either high or low not matching expected pattern.
                            pattern_detected = False
                            logger.error(f"Either low is unchanged. Pattern broken.")
                            start_candle = {}
                            individual_candles = []
                            ind_candle_count = 0
                            pattern = []
                            prev_low_price = price.low_price
                    new_start = {'datetime': price.datetime.isoformat(),
                                 'low_price': float(price.low_price), 'high_price': float(price.high_price), 'colour' : 'green' }

            if sections > 0 and pattern_detected == True:
                logger.error(f'Strategy confirmed to be valid.')
                action_buy = True

                #prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
                #final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
                #duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                #                                                     later_candle=latest_price)
                data = {'duration_to_start' : duration_to_start- ind_candle_count, 'ind_candle_count' : str(ind_candle_count - 3), 'start_candle': start_candle, 'individual_candles': individual_candles,
                        'recent_swing_points' : recent_swing_points,} # recent_swing_points not as a string as it gets removed and accessed if present.
            else:
                data = {}
                action_buy = None
            logger.error(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #9 Buying for {self.ticker.symbol}: {e}")

class GannPointNineSell(BaseStrategy):
    name="Gann's Selling point #9"

    def check_criteria(self):
        try:
            data = {}
            action_buy = None
            # Access the latest DailyPrice (or other relevant price model) for the ticker
            swing_point_query = SwingPoint.objects.filter(ticker=self.ticker).order_by('-date')

            latest_price = DailyPrice.objects.filter(ticker=self.ticker).order_by('-datetime').first()
            swing_point_counter = 1
            T_prev = []
            #latest_T = 0
            recent_swing_points = []
            sections = 0
            pattern_detected = False
            most_recent_ll_price = None
            for swing_point in swing_point_query:
                # Check first is a LH
                logger.error(f'Swing point for "{str(self.ticker.symbol)}" at "{str(swing_point.date)}". swing_point_label:"{str(swing_point.label)}".')
                if swing_point_counter == 1:
                    if swing_point.label == 'LH':
                        logger.error(f'Detected first swingpoint. LH')
                        last_sp = swing_point
                        recent_swing_points.append(swing_point)
                    else:
                        # This strategy cannot be true. End review of swing points.
                        logger.error(f'First swingpoint not LH. Strategy not valid.')
                        break
                    # Now need to determine the elapsed days since this LL or HH.
                    #latest_T = instance_difference_count(self.ticker, swing_point.price_object)
                    most_recent_label = 'LH'
                    swing_point_counter += 1
                elif swing_point_counter >1:
                    if swing_point.label == 'LL' and most_recent_label == 'LH':
                        # Swing point is a low.
                        # Save the number of days that that it took to reach this swing point.
                        logger.error(f'Found a prior {swing_point.label}.')
                        #T_prev.append(swing_point.candle_count_since_last_swing_point)
                        most_recent_label = 'LL'
                        recent_swing_points.append(swing_point)
                        sections += 1
                        if most_recent_ll_price is None:
                            most_recent_ll_price = swing_point.price
                    elif swing_point.label == 'LH' and most_recent_label == 'LL':
                        logger.error(f'Found a prior {swing_point.label}. Another trough in the sequence.')
                        most_recent_label = 'LL'
                        recent_swing_points.append(swing_point)
                    else :
                        # This must be the start of the prior up trend.
                        # Stop checking further swing points.
                        logger.error(f'Found a prior {swing_point.label}. Sections: {sections}.')
                        first_sp = swing_point
                        recent_swing_points.append(swing_point)
                        break
                    swing_point_counter += 1
            if sections > 0:
                # At least 2 sections in most recent trend. So should analyse price movements from most recent swingpoint to most recent price
                # Does the most recent HH get broken?
                # Yes, then do we have 2 down candles followed by up candle.
                # Then check we have no close with a lower low.
                logger.error(f'Found sufficient sections to analyse recent price action.')
                # Get the candles to be analysed.
                prices = DailyPrice.objects.filter(ticker=self.ticker, datetime__gte=last_sp.price_object.datetime).order_by('datetime')
                ll_price_exceeded = False
                # Initialize variables
                prev_high_price = None
                prev_low_price = None
                pattern = []  # To track the pattern (higher, lower, lower, higher)
                individual_candles = []
                duration_to_start = 0
                for price in prices:
                    print(price.datetime, 'most_recent_ll_price:', most_recent_ll_price, 'price.low_price:',price.low_price )
                    duration_to_start += 1
                    if ll_price_exceeded == False and price.low_price < most_recent_ll_price:
                        # price has closed above the previous HH swingpoint.
                        ll_price_exceeded = True
                        prev_low_price = price.low_price
                        prev_high_price = price.high_price
                        logger.error(f'Price is below the previous LL. Checking individual price candles.')
                    elif ll_price_exceeded == True:
                        # Price went below prior LL. Start looking for the pattern.
                        print(price.datetime, 'prev_high_price:', prev_high_price, 'price.high_price:', price.high_price)
                        if prev_high_price is None or price.high_price < prev_high_price:
                            # Lower candle
                            if len(pattern) < 3:
                                logger.error(f"Expected a higher, got a lower price.")
                                pattern_detected = False
                                start_candle = {}
                                individual_candles = []
                                ind_candle_count = 0
                                pattern = []
                            elif len(pattern) == 3:
                                logger.error(f"Continuous L still true...")
                                pattern_detected = True
                                individual_candles.append({'datetime' : price.datetime.isoformat(), 'low_price': float(price.low_price), 'high_price' : float(price.high_price), 'colour' : 'red' })
                                ind_candle_count += 1
                            if len(pattern) == 0:
                                pattern = ['lower']
                                logger.error(f'First L found... pattern: {pattern}')
                                start_candle = new_start
                                individual_candles.append({'datetime' : price.datetime.isoformat(), 'low_price': float(price.low_price), 'high_price' : float(price.high_price), 'colour' : 'red' })
                                ind_candle_count = 1
                                # Save current candle as the peak in case the pattern is found
                            prev_high_price = price.high_price
                        elif price.high_price > prev_high_price:
                            # Higher candle
                            if (len(pattern) == 1 and pattern[0] == 'lower') or (len(pattern) == 2 and pattern[1] == 'higher'):
                                pattern.append('higher')
                                logger.error(f'H found... pattern: {pattern}')
                                # Save current candle as might be the 2-day retracement candle.
                                individual_candles.append({'datetime': price.datetime.isoformat(), 'low_price': float(price.low_price),'high_price': float(price.high_price), 'colour' : 'green' })
                                ind_candle_count += 1
                            elif pattern_detected:
                                pattern_detected = False
                                logger.error(f"Found H when expecting continuous L's. Pattern broken.")
                                pattern = []
                                start_candle = {}
                                individual_candles = []
                                ind_candle_count = 0
                            prev_high_price = price.high_price
                        else:
                            # Either high or low not matching expected pattern.
                            pattern_detected = False
                            logger.error(f"High is unchanged. Pattern broken.")
                            start_candle = {}
                            individual_candles = []
                            ind_candle_count = 0
                            pattern = []
                            prev_high_price = price.high_price
                    new_start = {'datetime' : price.datetime.isoformat(), 'low_price':float( price.low_price), 'high_price' : float(price.high_price), 'colour' : 'red' }
            if sections > 0 and pattern_detected == True:
                logger.error(f'Strategy confirmed to be valid.')
                action_buy = False

                #prior_trend_duration = instance_difference_count(self.ticker, first_candle, later_candle=last_candle)
                #final_upswing_size = round((latest_price.close_price - swing_point.price) / swing_point.price, 3) - 1
                #duration_after_latest_sp = instance_difference_count(self.ticker, last_sp.price_object,
                #                                                     later_candle=latest_price)
                data = {'duration_to_start' : duration_to_start-ind_candle_count, 'ind_candle_count' : str(ind_candle_count-3), 'start_candle': start_candle, 'individual_candles': individual_candles,
                        'recent_swing_points' : recent_swing_points,} # recent_swing_points not as a string as it gets removed and accessed if present.
            else:
                data = {}
                action_buy = None
            logger.error(f'........')
            return action_buy, data
        except Exception as e:
            print(f"Error in Gann #9 Sellining for {self.ticker.symbol}: {e}")


from django.utils import timezone

def process_trading_opportunities_single_ticker(ticker_symbol, strategies):
    # What to pass to function for 'strategies' i.e. a list with the strategy classes that should be tested:
    # strategies = [GannPointFourBuy2, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy, GannPointEightSell,
    #                   GannPointThreeBuy, GannPointThreeSell, GannPointOneBuy, GannPointOneSell, GannPointNineBuy, GannPointNineSell]  #

    ticker = Ticker.objects.get(symbol=ticker_symbol)

    try:
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
                if 'peak_before_two_day_retracement' in data:
                    peak_before_two_day_retracement = data['peak_before_two_day_retracement']
                    low_after_two_day_retracement = data['low_after_two_day_retracement']
                    del data['peak_before_two_day_retracement']
                    del data['low_after_two_day_retracement']
                if 'confirmed' in data:
                    # Strategy has set the flag to indicate whether the Trading signal is confirmed or tentative.
                    confirmed = data['confirmed']
                    print('type(confirmed):',type(confirmed))
                    del data['confirmed']
                else:
                    confirmed = True
                if existing_tradingopp is not None:
                    logger.error(f'Existing TradingOpp being updated (id={existing_tradingopp.id})...')
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

                    # Create a new TradingOpp instance.
                    trading_opp = TradingOpp.objects.create(
                        ticker=ticker,
                        strategy=strategy_instance,
                        datetime_identified=timezone.now(),
                        metrics_snapshot=data, # Capture relevant metrics
                        count = 1,
                        action_buy = action_buy,
                        confirmed = confirmed,
                    )
                    if recent_swing_points_exist == True:
                        for swing_point in recent_swing_points:
                            trading_opp.swing_points.add(swing_point)
                    trading_opp.save()
                    logger.error(f'Created new TradingOpp. id:{trading_opp.id}.')
            else:
                # The strategy is not valid for the ticker.
                # Check if there was an active TradingOpp for this Ticker / strategy and set is_active=0
                if existing_tradingopp is not None:
                    existing_tradingopp.is_active = False
                    existing_tradingopp.datetime_invalidated = datetime.now()
                    if action_buy is not None:
                        existing_tradingopp.action_buy = action_buy
                    existing_tradingopp.save()
        logger.error(f'Finished process_trading_opportunities_single_ticker().')
    except Exception as e:
        message = f'Error occured in process_trading_opportunities_single_ticker(). Current ticker: {ticker.symbol}. Error: {e}'
        nj_param = Params.objects.get(key='night_job_end_dt')
        end_time = datetime.now()
        nj_param.value = end_time
        nj_param.save()
        nj_param = Params.objects.get(key='night_job_status_message')
        nj_param.value = message
        nj_param.save()
        logger.error(message)

def process_trading_opportunities():
    logger.error(f'Starting process_trading_opportunities()...')
    tickers = Ticker.objects.all()
    #tickers = Ticker.objects.filter(symbol="LUV")
    #strategies = [TAEStrategy, TwoPeriodCumRSI, DoubleSevens]  # List of strategy classes
    strategies = [GannPointFourBuy2, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy, GannPointEightSell,
                  GannPointThreeBuy, GannPointThreeSell, GannPointOneBuy, GannPointOneSell, GannPointNineBuy, GannPointNineSell]  # List of strategy classes

    try:
        for ticker in tickers:

            process_trading_opportunities_single_ticker(ticker.symbol, strategies)
        logger.error(f'Finished process_trading_opportunities().')
    except Exception as e:
        print(f"Error in process_trading_opportunities. Current ticker: {ticker.symbol}: {e}")

