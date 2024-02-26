from django.shortcuts import render
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import yfinance as yf
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import time
import json

from django.shortcuts import render
from .models import *
from .forms import *
from datetime import datetime, timedelta, timezone, date
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from candlestick import candlestick
from . import db_candlestick
from . import update_ticker_metrics
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db.models import Q
from time import sleep
from .tasks import background_manual_category_download, background_manual_ticker_download

logger = logging.getLogger('django')
from rest_framework.response import Response

# https://manojadhikari.medium.com/track-email-opened-status-django-rest-framework-5fcd1fbdecfb
from rest_framework.views import APIView
from django.contrib.sessions.models import Session

BASE_DIR = Path(__file__).resolve().parent.parent
#print('BASE_DIR:',BASE_DIR)

def display_local_time():
    # Get the current datetime in UTC
    utc_now = datetime.utcnow()

    # Convert UTC datetime to the local timezone
    local_timezone = pytz.timezone('Europe/Riga')  # Replace with your local timezone
    local_datetime = utc_now.replace(tzinfo=pytz.utc).astimezone(local_timezone)

    # Format and print the local datetime
    local_datetime_str = local_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
    print(f'Current datetime: {local_datetime_str}')
    return local_datetime


# Create your views here.
def index(request):
    # Generate counts of some of the main objects
    context = {
    }
    # Render the HTML template index.html with the data in the context variable
    return render(request, 'index.html', context=context)

from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from rest_framework.authtoken.models import Token

@csrf_exempt
def get_auth_token(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            token, created = Token.objects.get_or_create(user=user)
            return JsonResponse({'token': token.key})
        else:
            return JsonResponse({'error': 'Invalid credentials'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

from django.contrib.auth import authenticate, login
from django.http import JsonResponse

def login_view(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid credentials'})

from django.db.models import OuterRef, Subquery, F

@login_required
def ticker_config(request):
    if request.method == 'POST':
        category_form = CategorySelectForm(request.POST)
        if category_form.is_valid():
            selected_categories = category_form.cleaned_data['categories']
            include_not_defined = request.POST.get('categories') == 'not_defined'
            uptrend = category_form.cleaned_data['uptrend']
            downtrend = category_form.cleaned_data['downtrend']
            swing_trend = category_form.cleaned_data['swing_trend']
            tae_score = category_form.cleaned_data['tae_score']
            two_period_cum_rsi = category_form.cleaned_data['two_period_cum_rsi']
        else:
            selected_categories = TickerCategory.objects.all()
            include_not_defined = True
            uptrend = False
            downtrend = False
            swing_trend = False
            tae_score = False
            two_period_cum_rsi = False

    else:
        category_form = CategorySelectForm()
        selected_categories = TickerCategory.objects.all()
        include_not_defined = True
        uptrend = False
        downtrend = False
        swing_trend = False
        tae_score = False
        two_period_cum_rsi = False

    tickers_q = Q(categories__in=selected_categories)

    if include_not_defined:
        tickers_q |= Q(categories=None)

    if uptrend:
        tickers_q &= Q(ma_200_trend_strength__gt=0)
        order_by = '-ma_200_trend_strength'
    elif downtrend:
        tickers_q &= Q(ma_200_trend_strength__lt=0)
        order_by = 'ma_200_trend_strength'
    elif swing_trend:
        tickers_q &= ~Q(swing_point_current_trend=0) & ~Q(swing_point_current_trend__isnull=True)
        order_by = '-ma_200_trend_strength'
    elif tae_score:
        tickers_q &= Q(tae_strategy_score__gt=0)
        order_by = '-ma_200_trend_strength'
    elif two_period_cum_rsi:
        tickers_q &= (
            (Q(cumulative_two_period_two_day_rsi__lt=10) & Q(ma_200_trend_strength__gt=0)) |
            (Q(cumulative_two_period_two_day_rsi__gt=70) & Q(ma_200_trend_strength__lt=0))
        )
        order_by = 'cumulative_two_period_two_day_rsi'
    else:
        order_by = 'symbol'

    tickers = Ticker.objects.filter(tickers_q).distinct().order_by(order_by)

    latest_daily_prices = DailyPrice.objects.filter(ticker=OuterRef('pk')).order_by('-datetime')

    tickers = tickers.annotate(
        latest_candle_close_price=Subquery(latest_daily_prices.values('close_price')[:1]),
        latest_candle_bearish_detected = Subquery(latest_daily_prices.values('bearish_detected')[:1]),
        latest_candle_bullish_detected = Subquery(latest_daily_prices.values('bullish_detected')[:1]),
        latest_candle_reversal_detected = Subquery(latest_daily_prices.values('reversal_detected')[:1]),
        latest_candle_bullish_reversal_detected=Subquery(latest_daily_prices.values('bullish_reversal_detected')[:1]),
    )

    tickers_with_data = []
    print('Computing data for ticker config listing...')
    hourly_price_query_count = 0
    for ticker in tickers:
        # Fetching the most recent DailyPrice's close_price
        #latest_candle = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
        #latest_close_price = latest_candle.close_price if latest_candle else None
        latest_candle = {
            'close_price': ticker.latest_candle_close_price,
            'bearish_detected': ticker.latest_candle_bearish_detected,
            'bullish_detected': ticker.latest_candle_bullish_detected,
            'reversal_detected': ticker.latest_candle_reversal_detected,
            'bullish_reversal_detected': ticker.latest_candle_bullish_reversal_detected,
        }
        #print('tickers_config. ',ticker.symbol,'ticker.swing_point_current_trend:',ticker.swing_point_current_trend)

        # Increment hourly_price_query_count if 15 or 5 min updates are desired.
        if ticker.is_fifteen_min:
            hourly_price_query_count += 4
        if ticker.is_five_min:
            hourly_price_query_count += 12

        tickers_with_data.append({
            'ticker': ticker,
            'latest_candle': latest_candle,
            # 'smallest_level_type' : smallest_level_type,
            # 'smallest_range_to_level' : smallest_range_to_level,
            # 'sr_level' : sr_level
        })

    yahoo_update_rate_percent = round(hourly_price_query_count * 100 / 200)

    return render(request, 'ticker_config.html', {
        'tickers_with_data': tickers_with_data,
        'yahoo_update_rate_percent': yahoo_update_rate_percent,
        'hourly_price_query_count': hourly_price_query_count,
        'category_form': category_form
    })

def add_ticker(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            print('Adding new ticker:')
            ticker_symbol = form.cleaned_data['symbol']
            existing_ticker = Ticker.objects.filter(symbol=ticker_symbol)
            if len(existing_ticker) == 0:
                print('ticker_symbol:',ticker_symbol)
                form.save()
            else:
                print('Ticker already exists. Not saving.')
            new_ticker=Ticker.objects.get(symbol=ticker_symbol)
            existing_daily_prices = DailyPrice.objects.filter(ticker=new_ticker)
            if len(existing_daily_prices) == 0:
                print('Retrieving daily prices as new ticker added.')
                download_prices(timeframe="Daily", ticker_symbol=ticker_symbol, trigger='User')
                print('Updating metrics as new ticker added.')
                update_ticker_metrics.update_ticker_metrics()
            return redirect('ticker_config')
    else:
        form = TickerForm()

    return render(request, 'add_config.html', {'form': form})


from django.shortcuts import render, get_object_or_404, redirect
from .models import Ticker
from .forms import TickerForm
from django.utils import timezone


def get_price_data(ticker, interval, start_time, finish_time):
    try:
        # Ensure start_time and finish_time are timezone-aware
        start_time = start_time.replace(tzinfo=timezone.utc)
        finish_time = finish_time.replace(tzinfo=timezone.utc)

        data = yf.Ticker(ticker.symbol).history(interval=interval, start=start_time, end=finish_time)
        if not data.empty:

            data['Ticker'] = ticker.symbol  # Add 'Ticker' column with the symbol

            # Create a 'Datetime' column from the index
            data['Datetime'] = data.index

            # Convert the datetime to a naive datetime
            #data['Datetime'] = data['Datetime'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)
            # Set the timezone to your project's timezone (e.g., UTC)
            #data['Datetime'] = data['Datetime']

            # Set the timezone to your project's timezone (e.g., UTC)
            #data['Datetime'] = data['Datetime'].apply(timezone.make_aware, timezone=timezone.utc)
            # Ensure the datetime index is in UTC timezone
            #data.index = data.index.tz_convert(timezone.utc)
            data = data.tz_localize(None)

            data['PercentChange'] = data['Close'].pct_change() * 100  # Multiply by 100 to get a percentage
            data.at[data.index[0], 'PercentChange'] = 0

            # Reorder the columns
            data = data[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume','PercentChange']]

            time.sleep(1)
            print(data)
    except Exception as e:
        print(f"Error fetching data for {ticker.symbol}: {e}")
        data = pd.DataFrame(columns=['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])

    return data

def get_missing_dates(ticker, interval, start_day, finish_day):
    # Get the list of dates missing in DailyPrice for the given ticker within the date range
    if interval == '1D':
        existing_dates = DailyPrice.objects.filter(ticker=ticker, datetime__range=(start_day, finish_day)).values_list('datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='D')
        missing_dates = [date for date in all_dates if date not in existing_dates]
    if interval == '15m':
        existing_dates = FifteenMinPrice.objects.filter(ticker=ticker, datetime__range=(start_day, finish_day)).values_list(
            'datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='15T')
        missing_dates = [date for date in all_dates if date not in existing_dates]
        #print('all_dates:',all_dates)
    if interval == '5m':
        existing_dates = FiveMinPrice.objects.filter(ticker=ticker,
                                                        datetime__range=(start_day, finish_day)).values_list(
            'datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='5T')
        missing_dates = [date for date in all_dates if date not in existing_dates]
        print('all_dates:',all_dates)
        print('missing_dates:',missing_dates)
    return missing_dates

def add_candle_data(price_history, candlestick_functions, column_names):
    for candlestick_func, column_name in zip(candlestick_functions, column_names):
        price_history = candlestick_func(price_history, target=column_name, ohlc=['Open', 'High', 'Low', 'Close'])
        price_history[column_name].fillna(False, inplace=True)
    return price_history

def add_db_candle_data(price_history, db_candlestick_functions, db_column_names):
    for db_candlestick_func, column_name in zip(db_candlestick_functions, db_column_names):
        price_history = db_candlestick_func(price_history, target=column_name, ohlc=['Open', 'High', 'Low', 'Close'])
        price_history[column_name].fillna(False, inplace=True)
        #print('column_name:',column_name)
        #print(price_history[column_name])
    return price_history

# List of candlestick functions to replace 'candlestick.xxx'
candlestick_functions = [candlestick.bullish_engulfing, candlestick.bullish_harami, candlestick.hammer, candlestick.inverted_hammer,
                         candlestick.hanging_man, candlestick.shooting_star, candlestick.bearish_engulfing, candlestick.bearish_harami,
                         candlestick.dark_cloud_cover, candlestick.gravestone_doji, candlestick.dragonfly_doji, candlestick.doji_star,
                         candlestick.piercing_pattern, candlestick.morning_star, candlestick.morning_star_doji,
                         #candlestick.evening_star, candlestick.evening_star_doji
                         ]

# List of column names to replace 'xxx' in price_history
column_names = ['bullish_engulfing', 'bullish_harami', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star', 'bearish_engulfing', 'bearish_harami',
                'dark_cloud_cover', 'gravestone_doji','dragonfly_doji', 'doji_star', 'piercing_pattern', 'morning_star', 'morning_star_doji',
                #'evening_star', 'evening_star_doji'
                ]

# List of candlestick functions to replace 'candlestick.xxx'
db_candlestick_functions = [db_candlestick.three_white_soldiers,db_candlestick.bullish_kicker, db_candlestick.bearish_kicker]

# List of column names to replace 'xxx' in price_history
db_column_names = ['three_white_soldiers','bullish_kicker', 'bearish_kicker'
                   ]

pattern_types = {
    'bullish': ['bullish_engulfing', 'bullish_harami', 'hammer', 'inverted_hammer', 'hanging_man',
                'three_white_soldiers', 'bullish_kicker'],
    'bearish': ['bearish_engulfing', 'bearish_harami', 'dark_cloud_cover', 'gravestone_doji', 'bearish_kicker', 'shooting_star'],
    'reversal': ['dragonfly_doji', 'doji_star', 'piercing_pattern'],
    'bullish_reversal': ['morning_star', 'morning_star_doji'],
    'bearish_reversal': [], }

def count_patterns(df, pattern_types):
    # Initialize new columns with zeros
    for pattern_type in pattern_types.keys():
        df[pattern_type] = 0

    # Loop through each pattern_type and its corresponding column names
    for pattern_type, columns in pattern_types.items():
        # Sum the boolean values across the relevant columns and assign to the new column
        df[pattern_type] = df[columns].sum(axis=1)

    # Create the 'patterns' column using a vectorized operation
    all_columns = [col for columns in pattern_types.values() for col in columns]

    bool_df = df[all_columns]
    df['patterns_detected'] = bool_df.dot(bool_df.columns + ' ').str.strip()

def find_levels(df, columns=['Open', 'Close'], window=20, retest_threshold_percent=0.01):
#def find_levels(df, columns=['Close'], window=20, retest_threshold_percent=0.001):
    support = {}
    resistance = {}
    sr_level = {}
    last_high_low_level=None
    retests = {}   # Keep track of retest counts and last retest datetime for each level
    window_last_high = 5

    # First pass to identify initial support and resistance levels
    for i in range(window, len(df) - window):
        combined_window = pd.concat([df[col][i - window:i + window] for col in columns])
        min_val = combined_window.min()
        max_val = combined_window.max()

        current_close = df.iloc[i]['Close']
        current_open = df.iloc[i]['Open']

        if current_close == min_val:
            support[current_close] = df.index[i]
            sr_level[current_close] = df.index[i]

        elif current_open == min_val:
            support[current_open] = df.index[i]
            sr_level[current_open] = df.index[i]

        if current_close == max_val:
            resistance[current_close] = df.index[i]
            sr_level[current_close] = df.index[i]

        elif current_open == max_val:
            resistance[current_open] = df.index[i]
            sr_level[current_open] = df.index[i]

    # Now look for a min/max for the most recent high/low:
    for i in range(window_last_high, len(df) - window_last_high):
        combined_last_low_high_window = pd.concat(
            [df[col][i - window_last_high:i + window_last_high] for col in columns])
        last_min_val = combined_last_low_high_window.min()
        last_max_val = combined_last_low_high_window.max()

        current_close = df.iloc[i]['Close']
        current_open = df.iloc[i]['Open']
        if current_close == last_min_val:
            last_high_low_level = current_close

        elif current_open == last_min_val:
            last_high_low_level = current_open

        if current_close == last_max_val:
            last_high_low_level = current_close

        elif current_open == last_max_val:
            last_high_low_level = current_open

    # Helper function to check if two levels are within threshold of each other
    def within_threshold(level1, level2, threshold_percent):
        close_levels = []
        for future_level in level2:
            if abs(level1 - future_level) <= level1 * threshold_percent:
                print('Future level close to level1:',future_level)
                close_levels.append(future_level)
        #print('Level1:', level1, 'Level2:',level2, 'abs(level1 - level2):',abs(level1 - level2),'threshold_value:',level1 * threshold_percent, 'Within threshold?',abs(level1 - level2) <= level1 * threshold_percent)
        return close_levels

    # Remove sr_levels that are within threshold of a previous support level
    print('sr_level:',sr_level)
    sr_keys_sorted = sorted(sr_level, key=sr_level.get)  # sort by datetime
    #print('sr_keys_sorted:',sr_keys_sorted)
    for i in range(len(sr_keys_sorted)):
        level1 = sr_keys_sorted[i]
        if level1 in sr_level:
            #print('Level1:', level1)
            compared_levels = sr_keys_sorted[i + 1:]
            close_levels = within_threshold(level1, compared_levels, retest_threshold_percent)
            if len(close_levels) > 0:
                #print('close_levels:', close_levels)
                all_levels = [level1] + close_levels
                #print('all_levels:', all_levels)
                avg_level = sum(all_levels) / len(all_levels)
                print('level1 changed from',level1,'to avg_level', avg_level)
                datetime_for_avg_level = sr_level[level1]  # store datetime associated with original level
                del sr_level[level1]
                for level in close_levels:
                    if level in sr_level:
                        del sr_level[level]
                sr_level[avg_level] = datetime_for_avg_level
                sr_keys_sorted[i] = avg_level  # Update the current key with average level

    # Second pass to update levels if they are breached and track retests
    for i in range(len(df)):
        current_datetime = df.index[i]
        current_close = df.iloc[i]['Close']
        current_open = df.iloc[i]['Open']

        # Check if current price breaches or retests any prior support levels
        for level in list(sr_level.keys()):
            level_datetime = sr_level[level]

            if level_datetime < current_datetime:
                threshold = level * retest_threshold_percent  # Calculate threshold as a % of the level
                #if abs(level - current_close) <= threshold or abs(level - current_open) <= threshold:
                if abs(level - current_close) <= threshold:
                    # Handle retests
                    retests.setdefault(level, {'count': 0, 'last_retest': None})
                    retests[level]['count'] += 1
                    retests[level]['last_retest'] = current_datetime
                    print(f"Support/resistance level {level} retested at {current_datetime}.")

    return sr_level, retests, last_high_low_level

def add_levels_to_price_history(df, sr_levels, retests):
    # Initialize new columns with default values
    df['level'] = None
    df['level_type'] = 0
    df['level_strength'] = 0
    #print('Adding levels to price_history...')

    # Update 'level', 'level_type', and 'level_strength' based on the retests dictionary and the support levels
    for level, level_datetime in sr_levels.items():
        #print('Found support level:', level, level_datetime)
        df.at[level_datetime, 'level'] = level
        df.at[level_datetime, 'level_type'] = 1
        if level in retests:
            #print('Found retest of support level.')
            count = retests[level]['count']
            df.at[level_datetime, 'level_strength'] = count + 1
        else:
            df.at[level_datetime, 'level_strength'] = 1

    return df


def add_ema_and_trend(price_history):
    # Calculate the Exponential Moving Average for 200 data points
    price_history['EMA_200'] = price_history['Close'].ewm(span=200, adjust=False).mean()

    # Calculate the Exponential Moving Average for 50 data points
    price_history['EMA_50'] = price_history['Close'].ewm(span=50, adjust=False).mean()

    # Infer trend based on the 2 EMA calculations
    def infer_trend(row):
        if row['EMA_50'] > row['EMA_200']:
            return 1
        elif row['EMA_50'] < row['EMA_200']:
            return -1
        else:
            return 0

    price_history['Trend'] = price_history.apply(infer_trend, axis=1)

    return price_history

from .price_download import download_prices

@login_required
def edit_ticker(request, ticker_id):
    # Retrieve the Ticker instance to be edited or create a new one if it doesn't exist
    ticker = get_object_or_404(Ticker, id=ticker_id)
    support = {}
    resistance = {}
    retests = {}

    if request.method == 'POST':
        form = TickerForm(request.POST, instance=ticker)
        print('Checking', ticker.symbol)
        if form.is_valid():
            if ticker.is_daily:
                existing_prices = DailyPrice.objects.all()
                if len(existing_prices) == 0:
                    print('About to call download_prices(timeframe="Daily")...')
                    download_prices(timeframe="Daily", ticker_symbol=ticker.symbol, trigger='User')
            if ticker.is_fifteen_min:
                print('About to call download_prices(timeframe="15 mins")...')
                download_prices(timeframe="15 mins", ticker_symbol=ticker.symbol, trigger='User')
            if ticker.is_five_min:
                print('About to call download_prices(timeframe="5 mins")...')
                download_prices(timeframe="5 mins", ticker_symbol=ticker.symbol, trigger='User')
            form.save()

            return redirect('ticker_config')  # Redirect back to the configuration page
    else:
        form = TickerForm(instance=ticker)

    return render(request, 'edit_ticker.html', {'form': form, 'ticker': ticker})


from django.shortcuts import render
import pytz

@login_required
def daily_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    daily_prices = DailyPrice.objects.filter(ticker=ticker).order_by('datetime')


    riga_tz = pytz.timezone('Europe/Riga')  # Define the timezone object

    for daily_price in daily_prices:
        daily_price.datetime = timezone.localtime(daily_price.datetime, riga_tz)  # Use the defined timezone object
        daily_price.sum_reversals = daily_price.reversal_detected + daily_price.bearish_reversal_detected + daily_price.bullish_reversal_detected
        daily_price.swing_direction = 0
        if daily_price.swing_point_label == "HH":
            daily_price.swing_direction = 1
        if daily_price.swing_point_label == "LH":
            daily_price.swing_direction = -1
        if daily_price.swing_point_label == "HL":
            daily_price.swing_direction = 1
        if daily_price.swing_point_label == "LL":
            daily_price.swing_direction = -1

    return render(request, 'price_list_v2.html', {'ticker': ticker, 'candles': daily_prices, 'heading_text' : 'Daily'})

@login_required
def fifteen_min_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    fifteen_min_prices = FifteenMinPrice.objects.filter(ticker=ticker).order_by('datetime')

    riga_tz = pytz.timezone('Europe/Riga')  # Define the timezone object

    for fifteen_min_price in fifteen_min_prices:
        fifteen_min_price.datetime = timezone.localtime(fifteen_min_price.datetime, riga_tz)
        fifteen_min_price.sum_reversals = fifteen_min_price.reversal_detected + fifteen_min_price.bearish_reversal_detected + fifteen_min_price.bullish_reversal_detected\

    return render(request, 'price_list_v2.html', {'ticker': ticker, 'candles': fifteen_min_prices, 'heading_text' : '15 Minute'})

@login_required
def five_min_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    five_min_prices = FiveMinPrice.objects.filter(ticker=ticker).order_by('datetime')

    riga_tz = pytz.timezone('Europe/Riga')  # Define the timezone object

    for five_min_price in five_min_prices:
        five_min_price.datetime = timezone.localtime(five_min_price.datetime, riga_tz)
        five_min_price.sum_reversals = five_min_price.reversal_detected + five_min_price.bearish_reversal_detected + five_min_price.bullish_reversal_detected\

    return render(request, 'price_list_v2.html', {'ticker': ticker, 'candles': five_min_prices, 'heading_text' : '5 Minute'})

@login_required
def update_metrics_view(request, ticker_symbol):
    update_ticker_metrics.update_ticker_metrics(ticker_symbol=ticker_symbol)
    return HttpResponseRedirect(reverse('ticker_config'))  # Redirect to admin dashboard or any other desired URL

@login_required
def ticker_detail(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    daily_prices_query = DailyPrice.objects.filter(ticker=ticker, level__isnull=False).only('datetime', 'level', 'level_strength')
    swing_daily_prices_query = DailyPrice.objects.filter(ticker=ticker, swing_point_label__isnull=False)\
        .exclude(swing_point_label="").only('datetime', 'swing_point_label').order_by('datetime')

    # Fetching the most recent DailyPrice's close_price
    latest_candle = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
    if latest_candle is not None:
        latest_close_price = latest_candle.close_price if latest_candle else None
        patterns_detected = latest_candle.patterns_detected
        if latest_candle.bullish_detected:
            bullish_detected = 'True'
        else:
            bullish_detected = 'False'
        if latest_candle.bullish_reversal_detected:
            bullish_reversal_detected = 'True'
        else:
            bullish_reversal_detected = 'False'
        if latest_candle.bearish_detected:
            bearish_detected = 'True'
        else:
            bearish_detected = 'False'
        if latest_candle.reversal_detected:
            reversal_detected = 'True'
        else:
            reversal_detected = 'False'

        # Computing the number of days from datetime to today for each DailyPrice instance
        current_date = date.today()
        daily_prices = []
        smallest_range_to_level = 100
        for dp in daily_prices_query:
            days_difference = (current_date - dp.datetime.date()).days
            if latest_close_price and latest_close_price != 0:
                close_price_percentage = (abs(dp.level-latest_close_price) / latest_close_price) * 100
                if close_price_percentage < smallest_range_to_level:
                    smallest_range_to_level = close_price_percentage
            else:
                close_price_percentage = None
            daily_prices.append(
                {'daily_price' : dp,
                 'days_from_today' : days_difference,
                 'close_price_percentage' : close_price_percentage}
            )

        # Add the swing direction to the swingpoint entries
        swing_points = []
        for sp in swing_daily_prices_query:
            if sp.swing_point_label[0] == "H":
                swing_direction = 1
            else:
                swing_direction = -1
            swing_points.append(
                {'swing_point' : sp,
                 'swing_direction' : swing_direction,
                 }
            )

    else:
        daily_prices = {}
        swing_points = {}
        latest_close_price = 'False'
        patterns_detected  = 'False'
        bullish_detected = 'False'
        bullish_reversal_detected = 'False'
        bearish_detected = 'False'
        reversal_detected = 'False'


    context = {
        'ticker': ticker,
        'daily_prices': daily_prices,
        'swing_points' : swing_points,
        'close_price': latest_close_price,
        #'smallest_level_type' : smallest_level_type,
        #'smallest_range_to_level' : smallest_range_to_level,
        'patterns_detected' : patterns_detected,
        'bullish_detected' : bullish_detected,
        'bullish_reversal_detected' : bullish_reversal_detected,
        'bearish_detected' : bearish_detected,
        'reversal_detected' : reversal_detected,
    }
    return render(request, 'ticker_detail.html', context)

from django.shortcuts import redirect
from django.http import HttpResponseNotFound

@login_required
def ticker_delete(request, ticker_id):
    try:
        ticker = Ticker.objects.get(id=ticker_id)
    except Ticker.DoesNotExist:
        return HttpResponseNotFound("Ticker not found.")

    # Manually deleting associated objects if CASCADE isn't set on the ForeignKey in your models.
    # If CASCADE is set, these deletions would be automatic, and you won't need these lines.
    #DailyPrice.objects.filter(ticker=ticker).delete()
    #FifteenMinsPrice.objects.filter(ticker=ticker).delete()
    #FiveMinsPrice.objects.filter(ticker=ticker).delete()

    ticker.delete()

    return redirect('ticker_config')

# View to refresh price data for a specific ticker
def manual_download(request, ticker_id, timeframe):

    valid_timeframes = {"day" : "Daily", "15mins" : "15 mins", "5mins" : "5 mins"}
    if timeframe not in valid_timeframes.keys():
        print("manual_download() called with invalid timeframe:", timeframe)
    else:
        timeframe_label=valid_timeframes[timeframe]
        ticker = get_object_or_404(Ticker, id=ticker_id)
        download_prices(timeframe=timeframe_label, ticker_symbol=ticker.symbol, trigger='User')
    return redirect('ticker_config')

def manual_category_download(request, category_name):
    # Retrieve all tickers that are in the given category.
    try:
        tickers_for_throtlling = 195
        logger.error(f'manual_category_download() starting for stocks in category "{str(category_name)}"...')
        logger.error(f'category: {str(category_name)}')
        category_name = category_name.replace('%20',' ')
        category_name = category_name.replace('%2520', ' ')
        logger.error(f'Cleaned category: {str(category_name)}')
        tickers = Ticker.objects.filter(categories__name=category_name)
        ticker_count = Ticker.objects.filter(categories__name=category_name).count()
        logger.error(f'ticker_count: {str(ticker_count)}')
        if ticker_count > tickers_for_throtlling:
            logger.error(f'Rate throttling will occur.')
            throttling = True
        else:
            logger.error(f'No rate throttling needed.')
            throttling = False

        # Iterate through all retrieved tickers and download prices.
        for ticker in tickers:
            background_manual_ticker_download(ticker.symbol, throttling)
            logger.error(f'{str(ticker.symbol)} price download requested in background...')
        logger.error(f'manual_category_download() completed. All price downloads created as background tasks.')
        logger.error(
            f'=========================================================================================')
        #time.sleep(15)
        #logger.error(f'Waited 15 seconds.')
    except Exception as e:
        logger.error(f'Error occured in manual_category_download(). {e}')
    return redirect('ticker_config')

# View to refresh price data for a specific ticker
def manual_category_download_original(request, category_name):
    # Retrieve all tickers that are in the given category.
    tickers_for_throtlling = 195
    logger.error(f'manual_category_download() starting for stocks in category "{str(category_name)}"...')

    logger.error(f'category: {str(category_name)}')
    category_name = category_name.replace('%20',' ')
    category_name = category_name.replace('%2520', ' ')
    logger.error(f'Cleaned category: {str(category_name)}')
    tickers = Ticker.objects.filter(categories__name=category_name)
    ticker_count = Ticker.objects.filter(categories__name=category_name).count()
    logger.error(f'ticker_count: {str(ticker_count)}')
    if ticker_count > tickers_for_throtlling:
        logger.error(f'Rate throttling will occur.')
    else:
        logger.error(f'No rate throttling needed.')

    # Iterate through all retrieved tickers and download prices.
    for ticker in tickers:
        start_time = display_local_time()  # record the start time of the loop
        logger.error(f'ticker.symbol: {str(ticker.symbol)}')
        download_prices(timeframe='Daily', ticker_symbol=ticker.symbol, trigger='User')

        end_time = display_local_time()  # record the end time of the loop
        elapsed_time = end_time - start_time  # calculate elapsed time
        logger.error(f'elapsed_time.total_seconds(): {str(elapsed_time.total_seconds())} secs...')
        logger.error(f'elapsed_time.total_seconds() < 20: {str(elapsed_time.total_seconds() < 20)} secs...')
        logger.error(f'ticker_count > tickers_for_throtlling: {str(ticker_count > tickers_for_throtlling)} secs...')

        if elapsed_time.total_seconds() < 20 and ticker_count > tickers_for_throtlling:
            pause_duration = 20 - elapsed_time.total_seconds()
            print('Rate throttling for', pause_duration, 'secs...')
            logger.error(f'Rate throttling for {str(pause_duration)} secs...')
            sleep(pause_duration)

    # Redirect to the 'ticker_config' URL pattern.
    return redirect('ticker_config')

from django.contrib import messages
from django.http import Http404
from .models import DailyPrice, Ticker

def delete_daily_price(request, symbol=None):
    if request.method == "POST":  # Ensure requests are POST for data modifications
        try:
            if symbol:  # if symbol provided, delete where symbol matches
                ticker_instance = Ticker.objects.get(symbol=symbol)
                DailyPrice.objects.filter(ticker=ticker_instance).delete()
                messages.success(request, f"DailyPrice instances with symbol {symbol} deleted successfully!")
            else:  # if symbol not provided, delete all
                DailyPrice.objects.all().delete()
                messages.success(request, "All DailyPrice instances deleted successfully!")
        except Ticker.DoesNotExist:
            raise Http404("Ticker with provided symbol does not exist")
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
        return redirect('ticker_config')  # Redirect to a confirmation page or the main page
    else:
        raise Http404("Invalid request method")  # Prevent deletion on GET request

STRATEGY_METRICS_LABELS = {
    "Gann's Buying point #4": {
        'latest_T': 'Time since last swing point',
        'T_prev': 'Duration of previous upswings',
        'count_T_prev' : 'Count of upswings',
        'max_T' : 'Duration of longest upswing',
        'prior_trend_duration' : 'Duration of downtrend',
        'final_upswing_size' : '% up from low',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Selling point #4": {
        'latest_T': 'Time since last swing point',
        'T_prev': 'Duration of previous downswings',
        'count_T_prev' : 'Count of downswings',
        'max_T' : 'Duration of longest downswing',
        'prior_trend_duration' : 'Duration of uptrend',
        'final_downswing_size' : '% down from high',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Buying point #3": {
        'P_prev': 'Previous price rises during downtrend',
        'max_P': 'Largest price rise during downtrend',
        'sections': 'Number of sections in downtrend',
        'downtrend_price_movement': 'Price fall during downtrend',
        'prior_trend_duration': 'Duration of downtrend',
        'latest_P': 'Price rise from low',
        'retracement_P': 'Price fall during retracement',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Selling point #3": {
        'P_prev': 'Previous price falls during uptrend',
        'min_P': 'Largest price fall during uptrend',
        'sections': 'Number of sections in uptrend',
        'uptrend_price_movement': 'Price rise during uptrend',
        'prior_trend_duration': 'Duration of uptrend',
        'latest_P': 'Price fall from high',
        'retracement_P': 'Price rise during retracement',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Buying point #5": {
      'latest_T': 'Time since last swing point',
      'T_most_recent': 'Duration of most recent upswing',
      'section_count': 'Number of sections in prior downtrend',
      'prior_trend_duration': 'Duration of downtrend',
      'final_upswing_size': '% up from low',
      'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Selling point #5": {
      'latest_T': 'Time since last swing point',
      'T_most_recent': 'Duration of most recent downswing',
      'section_count': 'Number of sections in prior uptrend',
      'prior_trend_duration': 'Duration of uptrend',
      'final_downswing_size': '% down from high',
      'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Buying point #8": {
      'latest_T': 'Time since last swing point',
      'bottoms': 'Number of troughs',
      'bottom_duration': 'Duration between first and last trough',
      'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Selling point #8": {
      'latest_T': 'Time since last swing point',
      'tops': 'Number of peaks',
      'tops_duration': 'Duration between first and last peak',
      'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Buying point #1" : {
        'sp_price_diff_vs_prior_high' : 'Price difference between pullback and prior HL',
        'price_retracement' : 'Price retracement from HH',
        'retracement_as_percent' : 'Retracement as % of rise to HH',
        'elapsed__duration' : 'Duration between prior HL and recent swingpoint',
        'rise_after_retracement' : 'Price rise after retracement swingpoint',
        'rise_after_retracement_percent_of_retracement' : 'Rise after retracement as % of retracement',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Selling point #1": {
        'sp_price_diff_vs_prior_low': 'Price difference between pullback and prior LH',
        'price_retracement': 'Price retracement from LL',
        'retracement_as_percent': 'Retracement as % of fall to LL',
        'elapsed__duration': 'Duration between prior LH and recent swingpoint',
        'fall_after_retracement': 'Price fall after retracement swingpoint',
        'fall_after_retracement_percent_of_retracement': 'Fall after retracement as % of retracement',
        'duration_after_latest_sp' : 'Duration since last SP',
    },
    "Gann's Buying point #9": {
        'start_candle': 'Candle prior to 2-day price action',
        'individual_candles': 'Individual candles',
        'ind_candle_count': 'Duration of successive growth since 2-day price action',
        'duration_to_start': 'Duration till 2-day action',
    },
    "Gann's Selling point #9": {
        'start_candle': 'Candle prior to 2-day price action',
        'individual_candles': 'Individual candles',
        'ind_candle_count': 'Duration of successive falls since 2-day price action',
        'duration_to_start': 'Duration till 2-day action',
    }

}

def translate_metrics(opp):
    strategy_name = opp.strategy.name  # Assuming `name` is the identifier for the strategy.
    metrics_labels = STRATEGY_METRICS_LABELS.get(strategy_name, {})
    translated_metrics = {
        metrics_labels.get(key, key): value  # Fallback to the original key if no label is found.
        for key, value in opp.metrics_snapshot.items()
    }
    return translated_metrics

@login_required
def trading_opps_view(request):
    # Get all active TradingOpp instances
    query = TradingOpp.objects.filter(is_active=True).select_related('ticker', 'strategy')

    action_param = request.GET.get('action', 'all')  # 'buy', 'sell', or 'all'
    category_param = request.GET.get('category', 'all')  # category ID or 'all'

    # Filter by action_buy if applicable
    if action_param.lower() == 'buy':
        query = query.filter(action_buy=True)
    elif action_param.lower() == 'sell':
        query = query.filter(action_buy=False)

    # Filter by TickerCategory if applicable
    if category_param != 'all':
        query = query.filter(ticker__categories__id=category_param)

    # Group by Ticker
    ticker_opps = {}
    for opp in query:
        ticker = opp.ticker
        opp.translated_metrics = translate_metrics(opp)  # Assign translated metrics to each opp
        if ticker not in ticker_opps:
            ticker_opps[ticker] = []
        ticker_opps[ticker].append(opp)

    context = {
        'ticker_opps': ticker_opps,
        'categories': TickerCategory.objects.all(),  # Assuming you want to list all categories for filtering
    }

    return render(request, 'trading_opp_list.html', context)

from collections import defaultdict
from datetime import datetime

from django.shortcuts import redirect, get_object_or_404
from django.views.decorators.http import require_POST

@require_POST
def update_tradingopp(request, opp_id):
    opp = get_object_or_404(TradingOpp, id=opp_id)

    # Check and convert if there is a value passed from the form for stop_loss_price
    stop_loss_price = request.POST.get('stop_loss_price')
    if stop_loss_price:
        opp.stop_loss_price = float(stop_loss_price)  # Convert to float

    # Check and convert if there is a value passed from the form for profit_taker_price
    profit_taker_price = request.POST.get('profit_taker_price')
    if profit_taker_price:
        opp.profit_taker_price = float(profit_taker_price)  # Convert to float

    # Check if 'schedule' was checked
    if 'schedule' in request.POST:
        latest_daily_price = DailyPrice.objects.filter(ticker=opp.ticker).order_by('-datetime').first()
        if latest_daily_price:
            # Create a Trade instance linked to this TradingOpp
            Trade.objects.create(
                tradingopp=opp,
                date=timezone.now(),  # Use timezone.now() to get the current date and time
                action='1' if opp.action_buy else '0',  # Set action based on action_buy of TradingOpp
                planned=True,  # Mark the Trade as planned
                price = latest_daily_price.close_price,  # Set the price to the close_price of the latest DailyPrice
                units = 0,
                currency = 'USD',
                rate_to_eur = 0.9213,
                commission = 1,
            )

    # Ensure conversion or computation logic is correctly handled in the model's save method or elsewhere as needed
    opp.save()

    return redirect('trading_opps_sorted')  # Ensure this name matches your URL pattern name for the listing view

def trading_opps_sorted_view(request):
    # Fetch query parameters for filtering
    action_param = request.GET.get('action', 'all')  # 'buy', 'sell', or 'all'
    category_param = request.GET.get('category', 'all')  # category ID or 'all'

    # Start with all active TradingOpps
    query = TradingOpp.objects.filter(is_active=True).select_related('ticker', 'strategy').order_by('-datetime_identified')

    # Filter by action_buy if applicable
    if action_param.lower() == 'buy':
        query = query.filter(action_buy=True)
    elif action_param.lower() == 'sell':
        query = query.filter(action_buy=False)

    # Filter by TickerCategory if applicable
    if category_param != 'all':
        query = query.filter(ticker__categories__id=category_param)

    # Group TradingOpps by date, ignoring time
    grouped_trading_opps = defaultdict(list)
    for opp in query:
        opp.translated_metrics = translate_metrics(opp)  # Assuming this function exists
        date_key = opp.datetime_identified.date()  # Extract date part
        grouped_trading_opps[date_key].append(opp)

    context = {
        'grouped_trading_opps': dict(grouped_trading_opps),
        'categories': TickerCategory.objects.all(),
    }

    # This return statement is now outside and at the end of the function, ensuring an HttpResponse is always returned
    return render(request, 'trading_opp_sorted_list.html', context)

class BaseGraphCustomizer:
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date,strategy_data, offset_up, offset_down):
        # Base customization logic (if any)
        pass


class GannOneBuyCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannOneBuyCustomizer()...')
        # Extract min_P from trading_opp's metrics_snapshot
        sp_price_diff_vs_prior_high = float(trading_opp.metrics_snapshot.get('sp_price_diff_vs_prior_high'))
        print('sp_price_diff_vs_prior_high:', sp_price_diff_vs_prior_high)
        price_retracement = float(trading_opp.metrics_snapshot.get('price_retracement'))
        print('price_retracement:', price_retracement)
        elapsed__duration = float(trading_opp.metrics_snapshot.get('elapsed__duration'))
        print('elapsed__duration:', elapsed__duration)
        rise_after_retracement = float(trading_opp.metrics_snapshot.get('rise_after_retracement'))
        print('rise_after_retracement:', rise_after_retracement)
        duration_after_latest_sp = trading_opp.metrics_snapshot.get('duration_after_latest_sp')
        if duration_after_latest_sp is not None:
            duration_after_latest_sp = int(duration_after_latest_sp)
            print('duration_after_latest_sp:', duration_after_latest_sp)
        else:
            duration_after_latest_sp = '?'
        swing_point_price_list = []
        swing_point_date_list = []
        swing_point_counter = 0
        # Get all swing_points into lists
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.price:', swing_point.price)
            swing_point_price_list.append(float(swing_point.price))
            swing_point_date_list.append(swing_point.date)

        # Draw horizontal line to left of retracement sp
        self.draw_horizontal_line(ax, swing_point_date_list[-3], swing_point_date_list[-1], swing_point_price_list[-1])
        label_price = swing_point_price_list[-3]  + (swing_point_price_list[-1]- swing_point_price_list[-3]) /2
        label_date = swing_point_date_list[-3]
        ax.text(label_date, label_price, f"{round(sp_price_diff_vs_prior_high, 2)}", fontsize=9, ha='center', va='bottom')

        # Draw horizontal line to right of peak sp
        self.draw_horizontal_line(ax, swing_point_date_list[-2], swing_point_date_list[-1], swing_point_price_list[-2])
        label_price = swing_point_price_list[-1] + (swing_point_price_list[-2] - swing_point_price_list[-1]) / 2
        label_date = swing_point_date_list[-1]
        ax.text(label_date, label_price, f"{round(price_retracement, 2)}", fontsize=9, ha='center',
                va='bottom')

        # Draw vertical line from retracement sp
        self.draw_vertical_line(ax, swing_point_date_list[-1], swing_point_price_list[-1], swing_point_price_list[-3] + offset_down)
        label_price = swing_point_price_list[-3]
        label_date = swing_point_date_list[-3] + (swing_point_date_list[-1] - swing_point_date_list[-3])/2
        ax.text(label_date, label_price, f"t={round(elapsed__duration,0)}", fontsize=9, ha='center',
                va='bottom')

        # Draw a line from most recent sp up to HH level. Add time to most recent price.
        self.draw_vertical_line(ax, swing_point_date_list[-1], swing_point_price_list[-1],swing_point_price_list[-2] + offset_up)
        self.draw_vertical_line(ax, most_recent_date, float(most_recent_price),swing_point_price_list[-3] + offset_down)
        label_price = swing_point_price_list[-3] + (offset_down * 2)
        label_date = swing_point_date_list[-1] + (most_recent_date - swing_point_date_list[-1]) / 2
        ax.text(label_date, label_price, f"t={duration_after_latest_sp}", fontsize=9, ha='center',
                va='bottom')

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

    def draw_horizontal_line(self, ax, date1, date2, price):
        # Draw a line between (date1, price) and (date2, price)
        ax.plot([date1, date2], [price, price], color='orange', linestyle='--')

class GannOneSellCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannOneSellCustomizer()...')

        # Get metrics from trading_opp's metrics_snapshot
        sp_price_diff_vs_prior_low = float(trading_opp.metrics_snapshot.get('sp_price_diff_vs_prior_low'))
        print('sp_price_diff_vs_prior_low:', sp_price_diff_vs_prior_low)
        price_retracement = float(trading_opp.metrics_snapshot.get('price_retracement'))
        print('price_retracement:', price_retracement)
        elapsed__duration = float(trading_opp.metrics_snapshot.get('elapsed__duration'))
        print('elapsed__duration:', elapsed__duration)
        fall_after_retracement = float(trading_opp.metrics_snapshot.get('fall_after_retracement'))
        print('fall_after_retracement:', fall_after_retracement)
        duration_after_latest_sp = float(trading_opp.metrics_snapshot.get('duration_after_latest_sp'))
        print('duration_after_latest_sp:', duration_after_latest_sp)

        swing_point_price_list = []
        swing_point_date_list = []
        swing_point_counter = 0
        # Get all swing_points into lists
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.price:', swing_point.price)
            swing_point_price_list.append(float(swing_point.price))
            swing_point_date_list.append(swing_point.date)

        # Draw horizontal line to left of retracement sp
        self.draw_horizontal_line(ax, swing_point_date_list[-3], swing_point_date_list[-1], swing_point_price_list[-1])
        label_price = swing_point_price_list[-1]
        label_date = swing_point_date_list[-3]
        ax.text(label_date, label_price, f"{round(sp_price_diff_vs_prior_low, 2)}", fontsize=9, ha='center', va='bottom')

        # Draw horizontal line to right of peak sp
        self.draw_horizontal_line(ax, swing_point_date_list[-2], swing_point_date_list[-1], swing_point_price_list[-2])
        label_price = swing_point_price_list[-2]
        label_date = swing_point_date_list[-1]
        ax.text(label_date, label_price, f"+{round(price_retracement, 2)}", fontsize=9, ha='center',
                va='bottom')

        # Draw vertical line from retracement sp
        self.draw_vertical_line(ax, swing_point_date_list[-1], swing_point_price_list[-1], swing_point_price_list[-3] + offset_up)
        label_price = swing_point_price_list[-3] + offset_down
        label_date = swing_point_date_list[-3] + (swing_point_date_list[-1] - swing_point_date_list[-3])/2
        ax.text(label_date, label_price, f"t={round(elapsed__duration,0)}", fontsize=9, ha='center',
                va='bottom')

        # Draw a line from most recent sp down to LL level. Add time to most recent price.
        self.draw_vertical_line(ax, swing_point_date_list[-1], swing_point_price_list[-1] + offset_down,
                            min(swing_point_price_list[-1] + offset_up, swing_point_price_list[-1]))
        self.draw_vertical_line(ax, most_recent_date, float(most_recent_price),swing_point_price_list[-3])
        label_price = swing_point_price_list[-3] + offset_down
        label_date = swing_point_date_list[-1] + (most_recent_date - swing_point_date_list[-1]) / 2
        ax.text(label_date, label_price, f"t={duration_after_latest_sp}", fontsize=9, ha='center',
                va='bottom')

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

    def draw_horizontal_line(self, ax, date1, date2, price):
        # Draw a line between (date1, price) and (date2, price)
        ax.plot([date1, date2], [price, price], color='orange', linestyle='--')

class GannThreeBuyCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannThreeBuyCustomizer()...')
        # Extract min_P from trading_opp's metrics_snapshot
        max_P = float(trading_opp.metrics_snapshot.get('max_P'))
        print('max_P:', max_P)
        #label_offset_multiplier = 0.5 # Multiplier of min value to offset price change label below the orange lines

        swing_point_price_list = []
        swing_point_date_list = []
        swing_point_counter = 0
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.price:',swing_point.price)
            swing_point_price_list.append(float(swing_point.price))
            swing_point_date_list.append(swing_point.date)
            if swing_point.label == 'LH':
                prior_price_trend = swing_point_price_list[-2]
                new_price_trend = float(swing_point.price)
                price_change = new_price_trend - prior_price_trend
                if abs(abs(price_change) - abs(max_P))< 0.01:
                    # This is the largest price change in the prior upswing. Add lines.
                    prior_date_trend = swing_point_date_list[-2]
                    new_date_trend = swing_point.date
                    self.draw_horizontal_line(ax, prior_date_trend, new_date_trend, prior_price_trend)
                    self.draw_horizontal_line(ax, prior_date_trend, new_date_trend, new_price_trend)

                    # Locate label for price movement:
                    # Place label at x% lower than the low price vs the overall price movement.
                    #label_price = min(prior_price_trend,new_price_trend) * (1 - label_offset_multiplier)
                    label_price = prior_price_trend #+ (prior_price_trend- new_price_trend) /2
                    label_date = prior_date_trend + (new_date_trend - prior_date_trend) / 2
                    ax.text(label_date, label_price, f"{round(price_change,2)}", fontsize=9, ha='center', va='bottom')

                print('Found lh_swing_point. new_price_trend:',new_price_trend,'prior_price_trend:',prior_price_trend,'price_change:',price_change)
            #last_swing_point = swing_point

        # Now loop back from the end of the list of swing points to add lines for the recent price movement.
        prior_price_primary = swing_point_price_list[-3]
        next_price_primary = swing_point_price_list[-2]
        prior_date_primary = swing_point_date_list[-3]
        new_date_primary = swing_point_date_list[-2]
        self.draw_horizontal_line(ax, prior_date_primary, new_date_primary, prior_price_primary)
        self.draw_horizontal_line(ax, prior_date_primary, new_date_primary, next_price_primary)

        # Locate label for price movement:
        # Place label at x% lower than the low price vs the overall price movement.
        #label_price = min(prior_price_primary, next_price_primary) * (1 - label_offset_multiplier)
        label_price = prior_price_primary #+ (prior_price_primary - next_price_primary) / 2
        label_date = prior_date_primary + (new_date_primary - prior_date_primary) / 2
        ax.text(label_date, label_price, f"{round(next_price_primary-prior_price_primary, 2)}", fontsize=9, ha='center', va='bottom')

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

    def draw_horizontal_line(self, ax, date1, date2, price):
        # Draw a line between (date1, price) and (date2, price)
        ax.plot([date1, date2], [price, price], color='orange', linestyle='--')

class GannThreeSellCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannThreeSellCustomizer()...')
        # Extract min_P from trading_opp's metrics_snapshot
        min_P = float(trading_opp.metrics_snapshot.get('min_P'))
        print('min_P:', min_P)
        #label_offset_multiplier = 0.5 # Multiplier of min value to offset price change label below the orange lines

        # Filter swing points to find the one with 'LH' label and matching candle_count_since_last_swing_point
        hh_swing_point = None
        # print(f'Before loop: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')
        swing_point_price_list = []
        swing_point_date_list = []
        swing_point_counter = 0
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.price:',swing_point.price)
            # print(f'Loop start: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')

            # For most recent swing point, compute the location of the text label for the time after this swing point.
            # Overwriting the prior value to leave just the last swingpoint value
            #mid_date_current = swing_point.date + (most_recent_date - swing_point.date) / 2
            #print('swing_point.date:', swing_point.date, 'most_recent_date:', most_recent_date, 'mid_date_current:',
            #      mid_date_current)
            swing_point_price_list.append(float(swing_point.price))
            swing_point_date_list.append(swing_point.date)
            if swing_point.label == 'HL':
                prior_price_trend = swing_point_price_list[-2]
                new_price_trend = float(swing_point.price)
                price_change = new_price_trend - prior_price_trend
                if abs(abs(price_change) - abs(min_P))< 0.01:
                    # This is the largest price change in the prior upswing. Add lines.
                    prior_date_trend = swing_point_date_list[-2]
                    new_date_trend = swing_point.date
                    self.draw_horizontal_line(ax, prior_date_trend, new_date_trend, prior_price_trend)
                    self.draw_horizontal_line(ax, prior_date_trend, new_date_trend, new_price_trend)

                    # Locate label for price movement:
                    # Place label at x% lower than the low price vs the overall price movement.
                    #label_price = min(prior_price_trend,new_price_trend) * (1 - label_offset_multiplier)
                    label_price = new_price_trend #+ (prior_price_trend- new_price_trend) /2
                    label_date = prior_date_trend + (new_date_trend - prior_date_trend) / 2
                    ax.text(label_date, label_price, f"{round(price_change,2)}", fontsize=9, ha='center', va='bottom')

                #hh_swing_point = swing_point
                print('Found hl_swing_point. new_price_trend:',new_price_trend,'prior_price_trend:',prior_price_trend,'price_change:',price_change)
            #last_swing_point = swing_point

        # Now loop back from the end of the list of swing points to add lines for the recent price movement.
        prior_price_primary = swing_point_price_list[-3]
        next_price_primary = swing_point_price_list[-2]
        prior_date_primary = swing_point_date_list[-3]
        new_date_primary = swing_point_date_list[-2]
        self.draw_horizontal_line(ax, prior_date_primary, new_date_primary, prior_price_primary)
        self.draw_horizontal_line(ax, prior_date_primary, new_date_primary, next_price_primary)

        # Locate label for price movement:
        # Place label at x% lower than the low price vs the overall price movement.
        #label_price = min(prior_price_primary, next_price_primary) * (1 - label_offset_multiplier)
        label_price = next_price_primary #+ (prior_price_primary - next_price_primary) / 2
        label_date = prior_date_primary + (new_date_primary - prior_date_primary) / 2
        ax.text(label_date, label_price, f"{round(next_price_primary-prior_price_primary, 2)}", fontsize=9, ha='center', va='bottom')

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

    def draw_horizontal_line(self, ax, date1, date2, price):
        # Draw a line between (date1, price) and (date2, price)
        ax.plot([date1, date2], [price, price], color='orange', linestyle='--')

class GannFourBuyCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date,strategy_data, offset_up, offset_down):
        print('Starting GannFourBuyCustomizer()...')
        # Extract max_T from trading_opp's metrics_snapshot
        max_T = int(trading_opp.metrics_snapshot.get('max_T'))
        print('max_T:',max_T)

        # Filter swing points to find the one with 'LH' label and matching candle_count_since_last_swing_point
        lh_swing_point = None
        #print(f'Before loop: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')
        for swing_point in swing_points:
            print('swing_point.label:',swing_point.label, ' swing_point.candle_count_since_last_swing_point:',swing_point.candle_count_since_last_swing_point )
            #print(f'Loop start: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')

            # For most recent swing point, compute the location of the text label for the time after this swing point.
            # Overwriting the prior value to leave just the last swingpoint value
            mid_date_current = swing_point.date + (most_recent_date - swing_point.date) / 2
            print('swing_point.date:', swing_point.date, 'most_recent_date:', most_recent_date, 'mid_date_current:',mid_date_current)
            if swing_point.label == 'LH' and int(swing_point.candle_count_since_last_swing_point) == max_T:
                lh_swing_point = swing_point
                print('Found lh_swing_point.')
            last_swing_point = swing_point

        if lh_swing_point:
            # Find the preceding swing point (if exists)
            preceding_swing_point = None
            lh_index = list(swing_points).index(lh_swing_point)
            if lh_index > 0:
                preceding_swing_point = swing_points[lh_index - 1]
                print('Found preceding_swing_point.')

            # Find min price for drawing vertical lines
            price_list = [swing_point.price for swing_point in swing_points]
            price_list.append(most_recent_price)
            min_price = min(price_list)
            print('min_price:', min_price)
            max_price = max(price_list)

            try:
                # Draw vertical lines
                if preceding_swing_point:
                    self.draw_vertical_line(ax, preceding_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, lh_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, most_recent_date, max_price, min_price)
                self.draw_vertical_line(ax, last_swing_point.date, max_price, min_price)

                # Add text annotation
                if preceding_swing_point and int(lh_swing_point.candle_count_since_last_swing_point) > 3:
                    mid_date = lh_swing_point.date + (preceding_swing_point.date - lh_swing_point.date) / 2
                    ax.text(mid_date, min_price, f"t={max_T}", fontsize=9, ha='center', va='bottom')
                # Add text label for time since the last low to current candle.
                latest_T = strategy_data['latest_T']
                print('latest_T:',latest_T)
                if int(latest_T) > 3:
                    ax.text(mid_date_current, min_price, f"t={latest_T}", fontsize=9, ha='center', va='bottom')
            except Exception as e:
                print(f"Error drawing lines and adding labels: {e}")

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

class GannFourSellCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date,strategy_data, offset_up, offset_down):
        print('Starting GannFourSellCustomizer()...')
        # Extract max_T from trading_opp's metrics_snapshot
        max_T = int(trading_opp.metrics_snapshot.get('max_T'))
        print('max_T:',max_T)

        # Filter swing points to find the one with 'HL' label and matching candle_count_since_last_swing_point
        hl_swing_point = None
        #print(f'Before loop: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')
        for swing_point in swing_points:
            print('swing_point.label:',swing_point.label, ' swing_point.candle_count_since_last_swing_point:',swing_point.candle_count_since_last_swing_point )
            #print(f'Loop start: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')

            # For most recent swing point, compute the location of the text label for the time after this swing point.
            # Overwriting the prior value to leave just the last swingpoint value
            mid_date_current = swing_point.date + (most_recent_date - swing_point.date) / 2
            print('swing_point.date:', swing_point.date, 'most_recent_date:', most_recent_date, 'mid_date_current:',mid_date_current)
            last_swing_point = swing_point
            if swing_point.label == 'HL' and int(swing_point.candle_count_since_last_swing_point) == max_T:
                hl_swing_point = swing_point
                print('Found hl_swing_point.')

        if hl_swing_point:
            # Find the preceding swing point (if exists)
            preceding_swing_point = None
            hl_index = list(swing_points).index(hl_swing_point)
            if hl_index > 0:
                preceding_swing_point = swing_points[hl_index - 1]
                print('Found preceding_swing_point.')

            # Find min price for drawing vertical lines
            price_list = [swing_point.price for swing_point in swing_points]
            price_list.append(most_recent_price)
            min_price = min(price_list)
            print('min_price:', min_price)
            max_price = max(price_list)

            try:
                # Draw vertical lines
                if preceding_swing_point:
                    self.draw_vertical_line(ax, preceding_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, hl_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, last_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, most_recent_date, max_price, min_price)

                # Add text annotation
                if preceding_swing_point and int(hl_swing_point.candle_count_since_last_swing_point) > 3:
                    mid_date = preceding_swing_point.date + ( hl_swing_point.date - preceding_swing_point.date) / 2
                    ax.text(mid_date, min_price, f"t={max_T}", fontsize=9, ha='center', va='bottom')
                # Add text label for time since the last low to current candle.
                latest_T = strategy_data['latest_T']
                print('latest_T:',latest_T)
                if int(latest_T) > 3:
                    ax.text(mid_date_current, min_price, f"t={latest_T}", fontsize=9, ha='center', va='bottom')
            except Exception as e:
                print(f"Error drawing lines and adding labels: {e}")

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

class GannFiveBuyCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date,strategy_data, offset_up, offset_down):
        print('Starting GannFiveBuyCustomizer()...')
        # Filter swing points to find the one with 'LH' label and matching candle_count_since_last_swing_point
        lh_swing_point = None
        # print(f'Before loop: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')
        max_lh_price = int(trading_opp.metrics_snapshot.get('max_lh_price'))
        print('max_lh_price:', max_lh_price)
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.candle_count_since_last_swing_point:',
                  swing_point.candle_count_since_last_swing_point)
            # print(f'Loop start: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')

            # For most recent swing point, compute the location of the text label for the time after this swing point.
            # Overwriting the prior value to leave just the last swingpoint value
            mid_date_current = swing_point.date + (most_recent_date - swing_point.date) / 2
            print('swing_point.date:', swing_point.date, 'most_recent_date:', most_recent_date, 'mid_date_current:',
                  mid_date_current)

            if swing_point.label == 'LH':
                lh_swing_point = swing_point
                print('Found lh_swing_point.')
            last_swing_point = swing_point

        if lh_swing_point:
            # Find the preceding swing point (if exists)
            preceding_swing_point = None
            lh_index = list(swing_points).index(lh_swing_point)
            if lh_index > 0:
                preceding_swing_point = swing_points[lh_index - 1]
                print('Found preceding_swing_point.')

            # Find min price for drawing vertical lines
            price_list = [swing_point.price for swing_point in swing_points]
            price_list.append(most_recent_price)
            min_price = min(price_list)
            print('min_price:', min_price)
            max_price = max(price_list)

            try:
                # Draw vertical lines
                if preceding_swing_point:
                    self.draw_vertical_line(ax, preceding_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, lh_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, most_recent_date, max_price, min_price)
                self.draw_vertical_line(ax, last_swing_point.date, max_price, min_price)

                # Add text annotation
                if preceding_swing_point and int(lh_swing_point.candle_count_since_last_swing_point) > 3:
                    mid_date = preceding_swing_point.date + ( lh_swing_point.date - preceding_swing_point.date) / 2
                    ax.text(mid_date, min_price, f"t={lh_swing_point.candle_count_since_last_swing_point}", fontsize=9, ha='center', va='bottom')
                # Add text label for time since the last low to current candle.
                latest_T = strategy_data['latest_T']
                print('latest_T:', latest_T)
                if int(latest_T) > 3:
                    ax.text(mid_date_current, min_price, f"t={latest_T}", fontsize=9, ha='center', va='bottom')
            except Exception as e:
                print(f"Error drawing lines and adding labels: {e}")

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

class GannFiveSellCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannFiveBuyCustomizer()...')
        # Filter swing points to find the one with 'LH' label and matching candle_count_since_last_swing_point
        hl_swing_point = None
        # print(f'Before loop: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.candle_count_since_last_swing_point:',
                  swing_point.candle_count_since_last_swing_point)
            # print(f'Loop start: mid_date_current = {mid_date_current}, most_recent_date = {most_recent_date}')

            # For most recent swing point, compute the location of the text label for the time after this swing point.
            # Overwriting the prior value to leave just the last swingpoint value
            mid_date_current = swing_point.date + (most_recent_date - swing_point.date) / 2
            print('swing_point.date:', swing_point.date, 'most_recent_date:', most_recent_date, 'mid_date_current:',
                  mid_date_current)

            if swing_point.label == 'HL':
                hl_swing_point = swing_point
                print('Found hl_swing_point.')
            last_swing_point = swing_point

        if hl_swing_point:
            # Find the preceding swing point (if exists)
            preceding_swing_point = None
            hl_index = list(swing_points).index(hl_swing_point)
            if hl_index > 0:
                preceding_swing_point = swing_points[hl_index - 1]
                print('Found preceding_swing_point.')

            # Find min price for drawing vertical lines
            price_list = [swing_point.price for swing_point in swing_points]
            price_list.append(most_recent_price)
            min_price = min(price_list)
            print('min_price:', min_price)
            max_price = max(price_list)

            try:
                # Draw vertical lines
                if preceding_swing_point:
                    self.draw_vertical_line(ax, preceding_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, hl_swing_point.date, max_price, min_price)
                self.draw_vertical_line(ax, most_recent_date, max_price, min_price)
                self.draw_vertical_line(ax, last_swing_point.date, max_price, min_price)

                # Add text annotation
                if preceding_swing_point and int(hl_swing_point.candle_count_since_last_swing_point) > 3:
                    mid_date = preceding_swing_point.date + (hl_swing_point.date - preceding_swing_point.date) / 2
                    ax.text(mid_date, min_price, f"t={hl_swing_point.candle_count_since_last_swing_point}", fontsize=9,
                            ha='center', va='bottom')
                # Add text label for time since the last low to current candle.
                latest_T = strategy_data['latest_T']
                print('latest_T:', latest_T)
                if int(latest_T) > 3:
                    ax.text(mid_date_current, min_price, f"t={latest_T}", fontsize=9, ha='center', va='bottom')
            except Exception as e:
                print(f"Error drawing lines and adding labels: {e}")

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')

class GannEightCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date,strategy_data, offset_up, offset_down):
        print('Starting GannEightCustomizer()...')

        duration_after_latest_sp = int(trading_opp.metrics_snapshot.get('duration_after_latest_sp'))
        print('duration_after_latest_sp:', duration_after_latest_sp)

        # Find min / max price for drawing lines
        price_list = [swing_point.price for swing_point in swing_points]
        min_price = min(price_list)
        print('min_price:', min_price)
        max_price = max(price_list)

        # Find min / max date for drawing horizontal lines
        date_list = [swing_point.date for swing_point in swing_points]
        date_list.append(most_recent_date)
        min_date = min(date_list)
        print('min_date:', min_date)
        max_date = max(date_list)

        bottom_price = trading_opp.metrics_snapshot.get('bottom_price')
        top_price = trading_opp.metrics_snapshot.get('top_price')
        if bottom_price is not None:
            min_price = float(bottom_price)
        if top_price is not None:
            max_price = float(top_price)

        try:
            # Draw horizontal lines
            self.draw_horizontal_line(ax, min_date, max_date, min_price)
            self.draw_horizontal_line(ax, min_date, max_date, max_price)
        except Exception as e:
            print(f"Error drawing lines: {e}")


        # Draw a line from most recent sp up to HH level. Add time to most recent price.
        print('date_list[-1]:', date_list[-1], 'date_list[-2]:', date_list[-2])
        self.draw_vertical_line(ax, date_list[-2], price_list[-1], float(most_recent_price))
        self.draw_vertical_line(ax, most_recent_date, float(most_recent_price), price_list[-1])
        label_price = float(price_list[-1]) + offset_up
        label_date = date_list[-2] + (most_recent_date - date_list[-2]) / 2
        ax.text(label_date, label_price, f"t={round(duration_after_latest_sp, 0)}", fontsize=9, ha='center',
                va='bottom')
        if float(most_recent_price) > price_list[-1] :
            # Buy strategy.
            pass
        else:
            # Sell strategy.
            pass

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')


    def draw_horizontal_line(self, ax, date1, date2, price):
        # Draw a line between (date1, price) and (date2, price)
        ax.plot([date1, date2], [price, price], color='orange', linestyle='--')

class GannNineCustomizer(BaseGraphCustomizer):
    def customize_graph(self, ax, trading_opp, swing_points, most_recent_price, most_recent_date, strategy_data, offset_up, offset_down):
        print('Starting GannNineCustomizer()...')
        # Extract min_P from trading_opp's metrics_snapshot
        start_candle = trading_opp.metrics_snapshot.get('start_candle')
        start_candle_datetime = datetime.fromisoformat(start_candle["datetime"])
        start_candle_low = float(start_candle["low_price"])
        start_candle_high = float(start_candle["high_price"])
        start_candle_colour = start_candle["colour"]
        duration_to_start = int(trading_opp.metrics_snapshot.get('duration_to_start'))
        print('duration_to_start:', duration_to_start)
        if start_candle_colour == 'green':
            start_candle_price = start_candle_high
        else:
            start_candle_price = start_candle_low
        print('start_candle:', start_candle)
        individual_candles = trading_opp.metrics_snapshot.get('individual_candles')
        print('individual_candles:', individual_candles)

        swing_point_price_list = []
        swing_point_date_list = []
        swing_point_counter = 0
        # Get all swing_points into lists
        for swing_point in swing_points:
            print('swing_point.label:', swing_point.label, ' swing_point.price:', swing_point.price)
            swing_point_price_list.append(float(swing_point.price))
            swing_point_date_list.append(swing_point.date)

        ax.plot([start_candle_datetime], [start_candle_price], marker='o', color=start_candle_colour, linestyle='None')  # Mark the most recent price with a red dot
        # If there are previous points, draw a dotted line from the last swing point to the new point
        ax.plot([swing_point_date_list[-1], start_candle_datetime], [swing_point_price_list[-1], start_candle_price], start_candle_colour, linestyle='--')  # Dotted line

        candle_price_list = []
        for candle in individual_candles:
            print('candle:',candle)
            candle_datetime = datetime.fromisoformat(candle["datetime"])
            candle_low = float(candle["low_price"])
            candle_high = float(candle["high_price"])
            candle_colour = candle["colour"]
            self.draw_candle_line(ax, candle_datetime, candle_low, candle_high, candle_colour)
            if start_candle_colour == 'green':
                candle_price_list.append(candle_high)
            else:
                candle_price_list.append(candle_low)

        min_price = min(min(swing_point_price_list),min(candle_price_list))
        max_price = max(max(swing_point_price_list), max(candle_price_list))
        # Add vertical lines to show elapsed time since last swingpoint

        self.draw_vertical_line(ax, swing_point_date_list[-1], min_price, max_price)
        self.draw_vertical_line(ax, start_candle_datetime, min_price, max_price)

        # Add text annotation

        label_date = swing_point_date_list[-1] + (start_candle_datetime - swing_point_date_list[-1]) / 2
        if start_candle_colour == 'green':
            label_price = max_price
        else:
            label_price = min_price
        ax.text(label_date, label_price, f"t={duration_to_start}", fontsize=9,ha='center', va='bottom')

    def draw_candle_line(self, ax, date, low_price, high_price, colour):
        # Draw a line for a candle between (date, low_price) and (date, high_price)
        ax.plot([date, date], [low_price, high_price], color=colour, linestyle='-')

    def draw_vertical_line(self, ax, date, start_price, min_price):
        # Draw a line between (date, start_price) and (date, min_price)
        ax.plot([date, date], [start_price, min_price], color='orange', linestyle='--')


# Add more customizers for other strategies
def get_graph_customizer(trading_strategy):
    print('About to select Customizer for strategy:',trading_strategy.name)
    customizers = {
        "Gann's Buying point #4": GannFourBuyCustomizer(),
        "Gann's Selling point #4": GannFourSellCustomizer(),
        "Gann's Buying point #5": GannFiveBuyCustomizer(),
        "Gann's Selling point #5": GannFiveSellCustomizer(),
        "Gann's Buying point #8": GannEightCustomizer(),
        "Gann's Selling point #8": GannEightCustomizer(),
        "Gann's Buying point #3": GannThreeBuyCustomizer(),
        "Gann's Selling point #3": GannThreeSellCustomizer(),
        "Gann's Buying point #1": GannOneBuyCustomizer(),
        "Gann's Selling point #1": GannOneSellCustomizer(),
        "Gann's Buying point #9": GannNineCustomizer(),
        "Gann's Selling point #9": GannNineCustomizer(),
        # Map more strategies to their customizers
    }
    return customizers.get(trading_strategy.name, BaseGraphCustomizer())

from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import io
from django.db.models import Q

def generate_swing_point_graph_view(request, opp_id):
    # Fetch the TradingOpp instance by ID
    opp = TradingOpp.objects.get(id=opp_id)
    ticker = opp.ticker
    strategy = opp.strategy
    print('Creating graph for ticker',ticker.symbol)
    # Access the metrics_snapshot directly
    metrics_snapshot = opp.metrics_snapshot
    # Directly access SwingPoints associated with this TradingOpp
    swing_points = opp.swing_points.all()

    if swing_points.exists():
        # Use the first swing point to determine the content_type
        first_swing_point = swing_points.first()
        last_swing_point = swing_points.last()
        content_type = first_swing_point.content_type

        # Determine the Price model class based on the content_type
        model = content_type.model_class()

        # Assuming 'opp' is your TradingOpp instance and 'model' is the model class you are querying
        if opp.datetime_invalidated:
            # If datetime_invalidated is not None, filter for datetime less than datetime_invalidated
            most_recent_price_instance = model.objects.filter(
                Q(ticker=opp.ticker),
                Q(datetime__lt=opp.datetime_invalidated)
            ).order_by('-datetime').first()
        else:
            # Fetch the most recent price for this Ticker from the determined Price model
            most_recent_price_instance = model.objects.filter(ticker=opp.ticker).order_by('-datetime').first()

        if most_recent_price_instance:
            most_recent_price = most_recent_price_instance.close_price
            most_recent_date = most_recent_price_instance.datetime

    # Prepare data for plotting
    dates = [swing_point.date for swing_point in swing_points]
    prices = [float(swing_point.price) for swing_point in swing_points]
    labels = [swing_point.label for swing_point in swing_points]

    # Plotting logic
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.plot(dates, prices, marker='o', linestyle='-')

    # Calculate an offset as a function of the price range. This is a simple example; adjust the factor as needed.
    price_range = max(prices) - min(prices)
    label_offset_up = 0.05 # 5% upwards for labels above the datapoint
    label_offset_down = 0.10  # 10% downwards for labels above the datapoint
    offset_up = price_range * label_offset_up
    offset_down = price_range * -label_offset_down

    for date, price, label in zip(dates, prices, labels):
        if label.endswith('H'):
            va_align = 'bottom'
            y_pos = price + offset_up  # Move the label up
        elif label.endswith('L'):
            va_align = 'top'
            y_pos = price + offset_down  # Move the label down
        else:
            va_align = 'center'
            y_pos = price  # Keep the label at its current position

        ax.text(date, y_pos, f"{label}\n{price:.2f}", fontsize=9, ha='center', va=va_align)

    print('About to check if',strategy.id, 'is in the list to exclude line to latest price.')
    print("int(strategy.id) not in [14, 15]:",int(strategy.id) not in [14, 15])
    if int(strategy.id) not in [14, 15]:  # 14 = "Gann's Buying point #9", 15 = "Gann's Selling point #9"
        # Only add most recent price line and label for strategies that will not customize this.
        # Determine label placement based on comparison of the most recent price to the last swing point's price
        print(strategy.id,'is not in list [14, 15].')

        # Adding a line to the most recent close price.
        if dates and prices:
            if float(most_recent_price) > last_swing_point.price:
                colour = 'green'
            else:
                colour = 'red'
            ax.plot([most_recent_date], [float(most_recent_price)], marker='o', color=colour,
                    linestyle='None')  # Mark the most recent price with a red dot
            # If there are previous points, draw a dotted line from the last swing point to the new point
            ax.plot([dates[-1], most_recent_date], [prices[-1], float(most_recent_price)], colour,
                    linestyle='--')  # Dotted line

        last_swing_price = float(swing_points.last().price)  # Assuming swing_points is ordered by date

        if float(most_recent_price) < last_swing_price:
            va_align = 'top'
            y_pos = float(most_recent_price) + offset_down  # Move the label down
        else:
            va_align = 'bottom'
            y_pos = float(most_recent_price) + offset_up  # Move the label up

        # Annotate the most recent price point with its price, adjusting placement
        ax.text(most_recent_date, y_pos, f"{most_recent_price:.2f}", fontsize=9, ha='center', va=va_align)
    else:
        print(strategy.id, 'is present list [14, 15]. Excluding line to latest price.')

    # Select the appropriate graph customizer based on the TradingStrategy
    trading_strategy = opp.strategy
    customizer = get_graph_customizer(trading_strategy)

    # Apply customizations
    customizer.customize_graph(ax, opp, swing_points, most_recent_price, most_recent_date,metrics_snapshot, offset_up, offset_down)

    ax.set_xticks([])  # Optionally hide x-axis labels
    ax.set_yticks([])  # Optionally hide y-axis labels
    ax.autoscale(enable=True, axis='both', tight=None)
    ax.margins(x=0.1, y=0.4)

    # Save to a BytesIO buffer
    buffer = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buffer)
    plt.close(fig)  # Close the figure to release memory

    # IMPORTANT: Reset buffer position to the start
    buffer.seek(0)  # Reset the buffer's position to the start

    # Serve the image
    response = HttpResponse(buffer.getvalue(), content_type='image/png')
    response['Content-Length'] = str(len(response.content))

    # Set headers to prevent caching
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from django.http import HttpResponse
from .models import Ticker, DailyPrice

def generate_ticker_graph_view(request, ticker_symbol):
    ticker = Ticker.objects.get(symbol=ticker_symbol)
    daily_prices = DailyPrice.objects.filter(ticker=ticker).order_by('datetime')
    swing_point_query = SwingPoint.objects.filter(ticker=ticker).order_by('date')
    trading_opp_query = TradingOpp.objects.filter(ticker=ticker).order_by('datetime_identified')

    dates = [price.datetime for price in daily_prices]
    lows = [price.low_price for price in daily_prices]
    highs = [price.high_price for price in daily_prices]
    sp_dates = [sp.date for sp in swing_point_query]
    sp_price = [sp.price for sp in swing_point_query]
    trading_opp_dates = [to.datetime_identified.date() for to in trading_opp_query]
    trading_opp_action_buy = [to.action_buy for to in trading_opp_query]

    # Convert daily_prices to a dictionary for easy lookup
    date_to_price_map = {price.datetime: (price.low_price, price.high_price) for price in daily_prices}

    # Method to identify the first record of each month
    month_starts_indices = []
    previous_month = None
    sp_indices = []
    bar_colours = []
    to_counter = 0
    if len(trading_opp_dates) > to_counter:
        current_to_date = trading_opp_dates[to_counter] - timedelta(days=1)
    else:
        current_to_date = None
    logger.error(f'to_counter: {to_counter}. current_to_date: {current_to_date}')
    for i, date in enumerate(dates):
        if date.month != previous_month:
            month_starts_indices.append(i)
            previous_month = date.month
        if date in sp_dates:
            sp_indices.append(i)
        if current_to_date is not None:
            if date.date() == current_to_date:
                # Get the index for the date in the list of trading_opps
                to_action = trading_opp_action_buy[to_counter]
                if to_action == 0:
                    bar_colours.append('red')
                elif to_action == 1:
                    bar_colours.append('green')
                prior_to_date = current_to_date
                while True:
                    to_counter += 1
                    if len(trading_opp_dates) > to_counter:
                        current_to_date = trading_opp_dates[to_counter] - timedelta(days=1)
                        if prior_to_date < current_to_date:
                            break
                    else:
                        current_to_date = None
                        break
                logger.error(
                    f'len(trading_opp_dates): {len(trading_opp_dates)} to_counter: {to_counter}. current_to_date: {current_to_date}')
            else:
                bar_colours.append('black')
        else:
            bar_colours.append('black')
    # Check if the first label is too close to the second label
    # Adjust this condition based on your specific requirements
    if month_starts_indices[1] - month_starts_indices[0] < 5:  # Example threshold
        month_starts_indices.pop(0)  # Drop the first month start if too close to the second

    month_starts_dates = [dates[i] for i in month_starts_indices]
    labels = [date.strftime('%Y-%m') for date in month_starts_dates]

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_title(f'Price Range for {ticker.company_name} ({ticker_symbol})')

    # Adjust the swing points plotting logic
    sp_x_positions = []
    sp_y_positions = []
    for sp in swing_point_query:
        if sp.date in date_to_price_map:
            low_price, high_price = date_to_price_map[sp.date]
            # Determine if the swing point is closer to the high or the low for the day
            if abs(sp.price - high_price) < abs(sp.price - low_price):
                sp_y_positions.append(high_price)
            else:
                sp_y_positions.append(low_price)
            sp_x_positions.append(dates.index(sp.date))


    # Plot adjustments and the rest of your plotting logic...
    for i, (low, high) in enumerate(zip(lows, highs)):
        if bar_colours[i] == 'black':
            linewidth = 1
        else:
            linewidth = 2
        ax.plot([i, i], [low, high], color=bar_colours[i], linewidth=linewidth)

    # Plot swing points with corrected y-positions
    ax.plot(sp_x_positions, sp_y_positions, 'o', color='blue', linestyle='-')  # Example: red markers for visibility

    ax.set_xticks(month_starts_indices)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    plt.tight_layout()

    buffer = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buffer)
    plt.close(fig)

    buffer.seek(0)

    response = HttpResponse(buffer.getvalue(), content_type='image/png')
    response['Content-Length'] = str(len(response.content))
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

@login_required
def trading_opps_filtered(request):
    form = TickerSymbolForm()
    trading_opps = None
    ticker = None

    if 'symbol' in request.GET:
        form = TickerSymbolForm(request.GET)
        if form.is_valid():
            symbol = form.cleaned_data['symbol']
            ticker = Ticker.objects.filter(symbol__iexact=symbol).first()
            if ticker:
                trading_opps = TradingOpp.objects.filter(ticker=ticker).all()
                for opp in trading_opps:
                    opp.translated_metrics = translate_metrics(opp)  # Assign translated metrics to each opp
            else:
                trading_opps = []

    return render(request, 'trading_opps_filtered.html', {
        'form': form,
        'trading_opps': trading_opps,
        'ticker' : ticker
    })

from django.shortcuts import render
from background_task.models import Task

@login_required()
def task_queue_view(request):
    tasks = Task.objects.all().values('id', 'task_name', 'task_params', 'run_at')
    count = tasks.count()
    return render(request, 'task_queue.html', {'tasks': tasks, 'count': count})

@login_required()
def trading_opps_with_trades_view(request):
    # Fetch TradingOpp instances that have at least one Trade linked to them
    trading_opps = TradingOpp.objects.filter(trades__isnull=False).distinct().order_by('-id')

    # You could also prepare trades data if needed for the template or handle it directly within the template

    context = {
        'trading_opps': trading_opps,
    }

    return render(request, 'trading_opps_with_trades.html', context)

from django.http import HttpResponseRedirect
from django.urls import reverse
from .models import Trade
from django.views.decorators.csrf import csrf_exempt  # Only if you're bypassing CSRF temporarily

@csrf_exempt  # Consider proper CSRF protection in production
def update_trades(request):
    if request.method == 'POST':
        for key in request.POST:
            if key.startswith('date_'):
                trade_id = key.split('_')[1]
                trade = Trade.objects.get(id=trade_id)
                trade.date = request.POST.get(key)
                trade.action = request.POST.get(f'action_{trade_id}')
                trade.price = request.POST.get(f'price_{trade_id}', 0)
                trade.units = request.POST.get(f'units_{trade_id}', 0)
                trade.planned = f'planned_{trade_id}' in request.POST
                trade.save()
        return HttpResponseRedirect(reverse('trading_opps_with_trades'))  # Redirect back to the list
    return HttpResponseRedirect(reverse('home'))  # Redirect somewhere relevant if not a POST request
