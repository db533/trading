from .models import Ticker, DailyPrice
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import pytz
from datetime import datetime, timedelta, timezone, date

def display_local_time():
    # Get the current datetime in UTC
    utc_now = datetime.utcnow()

    # Convert UTC datetime to the local timezone
    local_timezone = pytz.timezone('Europe/Riga')  # Replace with your local timezone
    local_datetime = utc_now.replace(tzinfo=pytz.utc).astimezone(local_timezone)

    # Format and print the local datetime
    local_datetime_str = local_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f'Current datetime: {local_datetime_str}')

def compute_ma_200_trend_strength(ticker, data_points):
    ma_200_trend_strength = None
    if len(data_points) >= 200:
        closing_prices = [dp.close_price for dp in
                          reversed(data_points)]  # Reverse to get the values in ascending order of date
        series = pd.Series(closing_prices)

        # Calculate moving average of the latest 200 data points
        moving_avg = series.mean()

        # Get the last closing price
        last_closing_price = float(closing_prices[-1])

        try:
            ma_200_trend_strength = abs(last_closing_price - moving_avg) / last_closing_price
            if last_closing_price < moving_avg:
                ma_200_trend_strength = -ma_200_trend_strength
        except ZeroDivisionError:
            ma_200_trend_strength = None
            print('Divide by zero error as last_closing_price =', last_closing_price)
    else:
        print('To few data points to compute 200 day MA:', len(data_points))
    ticker.ma_200_trend_strength = ma_200_trend_strength
    print('ma_200_trend_strength:',ma_200_trend_strength)
    return ticker

def compute_atr(ticker, last_30_data_points):
    closing_prices = [dp.close_price for dp in
                      reversed(last_30_data_points)]  # Reverse to get the values in ascending order of date
    opening_prices = [dp.open_price for dp in
                      reversed(last_30_data_points)]  # Reverse to get the values in ascending order of date
    volumes = [dp.volume for dp in
               reversed(last_30_data_points)]  # Reverse to get the values in ascending order of date
    closing_series = pd.Series(closing_prices)
    opening_series = pd.Series(opening_prices)
    volumes_series = pd.Series(volumes)
    # print('closing_series:',closing_series)
    # print('opening_series:', opening_series)
    trade_range = abs(closing_series - opening_series)
    trade_range = trade_range * volumes_series
    # print('trade_range:', trade_range)
    atr = trade_range.sum() / volumes_series.sum()
    # print('atr:', trade_range.mean())
    ticker.atr = atr
    print('atr:', atr)
    return ticker

def compute_average_volume(ticker, last_100_data_points):
    volumes = [dp.volume for dp in
               reversed(last_100_data_points)]  # Reverse to get the values in ascending order of date
    volumes_series = pd.Series(volumes)
    # print('closing_series:',closing_series)
    # print('opening_series:', opening_series)
    average_volume=round(volumes_series.mean()/1000)

    ticker.avg_volume_100_days = average_volume
    print('average_volume (k):', average_volume)
    return ticker

def compute_rsi(data_points, period=14):
    # Get closing prices in ascending order of date
    closing_prices = [dp.close_price for dp in reversed(data_points)]
    # Get opening prices in ascending order of date
    opening_prices = [dp.open_price for dp in reversed(data_points)]

    # Convert to Pandas Series
    closing_series = pd.Series(closing_prices)
    opening_series = pd.Series(opening_prices)
    #print('opening_series:', opening_series)
    #print('closing_series:',closing_series)


    # Compute the daily gains and losses
    daily_changes = closing_series - opening_series
    #print('daily_changes:', daily_changes)
    gains = daily_changes.where(daily_changes > 0, 0)
    losses = -daily_changes.where(daily_changes < 0, 0)
    #print('gains:', gains)
    #print('losses:', losses)

    # Calculate the rolling average of gains and losses over the specified period
    #avg_gains = gains.rolling(window=period, min_periods=1).mean()
    #avg_losses = losses.rolling(window=period, min_periods=1).mean()
    avg_gains = gains.mean()
    avg_losses = losses.mean()
    #print('avg_gains:', avg_gains)
    #print('avg_losses:', avg_losses)

    # Calculate the relative strength
    if avg_losses > 0:
        RS = avg_gains / avg_losses
        RSI = 100 - (100 / (1 + RS))
    else:
        RSI = 100

    return RSI

def compute_period_high(ticker, data_points):
    # Get closing prices in ascending order of date
    closing_prices = [dp.close_price for dp in reversed(data_points)]

    # Convert to Pandas Series
    closing_series = pd.Series(closing_prices)
    #print('closing_series:',closing_series)

    max_close = closing_series.max()
    ticker.seven_day_max = max_close

    return ticker

def compute_period_low(ticker, data_points):
    # Get closing prices in ascending order of date
    closing_prices = [dp.close_price for dp in reversed(data_points)]

    # Convert to Pandas Series
    closing_series = pd.Series(closing_prices)
    # print('closing_series:',closing_series)

    min_close = closing_series.min()
    ticker.seven_day_min = min_close

    return ticker

def update_sr_level_data(ticker):
    print('update_sr_level_data()...')
    daily_prices_query = DailyPrice.objects.filter(ticker=ticker, level__isnull=False).only('datetime', 'level', 'level_strength')
    latest_candle = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
    latest_close_price = latest_candle.close_price if latest_candle else None
    print('latest_close_price:', latest_close_price)

    # Computing the number of days from datetime to today for each DailyPrice instance
    current_date = date.today()
    daily_prices = []
    smallest_range_to_level = 1000
    nearest_level_value=None
    days_since_last_tested = None

    if latest_close_price is not None and nearest_level_value is not None:
        for dp in daily_prices_query:
            days_difference = (current_date - dp.datetime.date()).days
            print('dp.level:',dp.level)
            print('close_price_percentage:',(abs(dp.level-latest_close_price) / latest_close_price) * 100)

            if latest_close_price and latest_close_price != 0:
                close_price_percentage = (abs(dp.level-latest_close_price) / latest_close_price) * 100
                if close_price_percentage < smallest_range_to_level:
                    smallest_range_to_level = close_price_percentage
                    nearest_level_value = dp.level
                    days_since_last_tested = days_difference
                    #if dp.close_price < dp.level:
                    #    smallest_level_type = 'Resistance'
                    #else:
                    #    smallest_level_type = 'Support'
            else:
                close_price_percentage = None

        if ticker.last_high_low != None:
            if smallest_range_to_level < 2:
                # Price is close to support / resistance level, so look back to the most recent high / low to determine if this is a support / resistance.
                if latest_close_price < ticker.last_high_low:
                    smallest_level_type = 'Support'
                else:
                    smallest_level_type = 'Resistance'
            else:
                # Price is far from support / resistance level, so just compare the close price to the support / resistance level.
                if smallest_range_to_level == 100:
                    smallest_level_type = None
                if latest_close_price > nearest_level_value:
                    smallest_level_type = 'Support'
                else:
                    smallest_level_type = 'Resistance'
        else:
            if latest_close_price < nearest_level_value:
                smallest_level_type = 'Support'
            else:
                smallest_level_type = 'Resistance'

        ticker.nearest_level_value = nearest_level_value
        ticker.nearest_level_type = smallest_level_type
        ticker.nearest_level_days_since_retest = days_since_last_tested
        ticker.nearest_level_percent_distance = smallest_range_to_level
        ticker.patterns_detected = latest_candle.patterns_detected
        ticker.bullish_detected = latest_candle.bullish_detected
        ticker.bullish_reversal_detected = latest_candle.bullish_reversal_detected
        ticker.bearish_detected = latest_candle.bearish_detected
        ticker.reversal_detected = latest_candle.reversal_detected

        ticker.save()
    else:
        print('No daily prices. Cannot compute metrics.')
    return ticker

def test_tae_strategy(ticker):
    trend_weight = 40
    sr_level_weight = 40
    candle_weight = 20
    ma_strength_threshold = 0.065

    strategy_score=0
    ma_strength = ticker.ma_200_trend_strength
    dist_to_sr_level = ticker.nearest_level_percent_distance
    sr_level_type = ticker.nearest_level_type
    candle_bullish = ticker.bullish_detected
    candle_bullish_reversal = ticker.bullish_reversal_detected
    candle_bearish = ticker.bearish_detected

    # Is ticker in a general upward trend?
    if ma_strength is not None:
        if ma_strength > ma_strength_threshold and sr_level_type == 'Support' and (candle_bullish or candle_bullish_reversal) and not (candle_bearish):
            strategy_score += (ma_strength-ma_strength_threshold)*trend_weight
            strategy_score += float((candle_bullish + candle_bullish_reversal)) * candle_weight
            strategy_score += float((max(5-float(dist_to_sr_level),0))*sr_level_weight/5)

        if ma_strength < -ma_strength_threshold and sr_level_type == 'Resistance' and (candle_bearish) and not (candle_bullish or candle_bullish_reversal):
            strategy_score +=  -(ma_strength + ma_strength_threshold) * trend_weight
            strategy_score += float((candle_bearish)) * candle_weight
            strategy_score += float((max(5 - float(dist_to_sr_level), 0)) * sr_level_weight / 5)

        ticker.tae_strategy_score = strategy_score
    else:
        ticker.tae_strategy_score = 0
    return ticker

def update_ticker_metrics(ticker_symbol="All"):
    display_local_time()

    print('Updating metrics:')
    if ticker_symbol == 'All':
        ticker_list = Ticker.objects.filter(is_daily=True)
    else:
        ticker_list = Ticker.objects.filter(symbol=ticker_symbol).filter(is_daily=True)
    for ticker in ticker_list:
        print('Ticker:', ticker.symbol)

        # Recompute the support / resistance levels.
        ticker = update_sr_level_data(ticker)

        last_200_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:200]  # Get the last 200 data points
        last_100_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:100]  # Get the last 100 data points
        last_30_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:30]  # Get the last 30 data points
        last_2_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:2]  # Get the last 2 data points
        prior_2_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[1:3]
        last_7_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[
                             :7]  # Get the last 2 data points

        if len(last_200_data_points) > 3:
            ticker = compute_ma_200_trend_strength(ticker,last_200_data_points)
            ticker = compute_atr(ticker, last_30_data_points)
            ticker = compute_average_volume(ticker, last_100_data_points)
            current_2_day_rsi = compute_rsi(last_2_data_points, period=2)
            #print('current_2_day_rsi:',current_2_day_rsi)
            prior_2_day_rsi = compute_rsi(prior_2_data_points, period=2)
            cumulative_two_period_two_day_rsi = (current_2_day_rsi + prior_2_day_rsi)/2
            ticker.cumulative_two_period_two_day_rsi = cumulative_two_period_two_day_rsi
            ticker = compute_period_high(ticker, last_7_data_points)
            ticker = compute_period_low(ticker, last_7_data_points)

            # Now update strategy fulfillment:
            ticker = test_tae_strategy(ticker)

            ticker.save()
