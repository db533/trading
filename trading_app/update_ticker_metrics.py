from .models import Ticker, DailyPrice
import pandas as pd
import datetime
import pytz

def display_local_time():
    # Get the current datetime in UTC
    utc_now = datetime.datetime.utcnow()

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


def update_ticker_metrics():
    display_local_time()

    print('Updating metrics:')
    for ticker in Ticker.objects.filter(is_daily=True):
        print('Ticker:', ticker.symbol)
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

            ticker.save()
