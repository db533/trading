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

def update_ticker_metrics():
    display_local_time()

    print('Updating metrics:')
    for ticker in Ticker.objects.filter(is_daily=True):
        print('Ticker:', ticker.symbol)
        last_200_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:200]  # Get the last 200 data points
        last_100_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:100]  # Get the last 200 data points
        last_30_data_points = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime')[:30]  # Get the last 200 data points

        ticker = compute_ma_200_trend_strength(ticker,last_200_data_points)
        ticker = compute_atr(ticker, last_30_data_points)
        ticker.save()
