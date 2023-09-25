from django.shortcuts import render
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import yfinance as yf
import pandas as pd
import time
import json


from django.shortcuts import render
from .models import *
from .forms import *
from rest_framework import status
from django.template.loader import get_template
from django.core.mail import EmailMultiAlternatives
from django.contrib.auth.models import User
from django.db.models import Sum, Max, Count
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from datetime import datetime, timedelta, timezone, date
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from candlestick import candlestick
from . import db_candlestick
from . import update_ticker_metrics
from django.http import HttpResponseRedirect
from django.urls import reverse

from rest_framework.response import Response

# https://manojadhikari.medium.com/track-email-opened-status-django-rest-framework-5fcd1fbdecfb
from rest_framework.views import APIView
from django.contrib.sessions.models import Session

BASE_DIR = Path(__file__).resolve().parent.parent
#print('BASE_DIR:',BASE_DIR)


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


from django.shortcuts import render, redirect
from .models import Ticker
from .forms import TickerForm

@login_required
def ticker_config(request):
    tickers = Ticker.objects.all().order_by('symbol')  # Order by symbol in ascending order

    tickers_with_data = []
    print('Computing data for tickers...')
    for ticker in tickers:
        # Fetching the most recent DailyPrice's close_price
        latest_candle = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
        latest_close_price = latest_candle.close_price if latest_candle else None
        daily_prices_query = DailyPrice.objects.filter(ticker=ticker, level__isnull=False).only('datetime', 'level',
                                                                                                'level_strength')
        current_date = date.today()
        smallest_range_to_level = 100
        smallest_level_type = ''
        sr_level = None
        print('Ticker:', ticker.symbol)
        print('latest_close_price:', latest_close_price)
        for dp in daily_prices_query:
            days_difference = (current_date - dp.datetime.date()).days
            if latest_close_price and latest_close_price != 0:
                print('dp.level:', dp.level)
                close_price_percentage = (abs(dp.level-latest_close_price) / latest_close_price) * 100
                print('close_price_percentage:', close_price_percentage)
                if close_price_percentage < smallest_range_to_level:
                    smallest_range_to_level = close_price_percentage
                    print('smallest_range_to_level is now:', smallest_range_to_level)
                    sr_level = dp.level
                    print('sr_level:', sr_level)
                    if dp.close_price < dp.level:
                        smallest_level_type = 'Resistance'
                    else:
                        smallest_level_type = 'Support'
            else:
                close_price_percentage = None

        #if smallest_range_to_level > 2:
            # We are more than 1% from a support / resistance level
        smallest_level_type = ''
        if ticker.last_high_low != None:
            if smallest_range_to_level < 2:
                if latest_close_price < ticker.last_high_low:
                    smallest_level_type = 'Support'
                else:
                    smallest_level_type = 'Resistance'
            else:
                if latest_close_price > sr_level:
                    smallest_level_type = 'Support'
                else:
                    smallest_level_type = 'Resistance'

        tickers_with_data.append({
            'ticker': ticker,
            'latest_candle': latest_candle,
            'smallest_level_type' : smallest_level_type,
            'smallest_range_to_level' : smallest_range_to_level,
            'sr_level' : sr_level
        })

    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('ticker_config')
    else:
        form = TickerForm()

    return render(request, 'ticker_config.html', {'form': form, 'tickers_with_data': tickers_with_data})

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
db_candlestick_functions = [db_candlestick.three_white_soldiers,
                         ]

# List of column names to replace 'xxx' in price_history
db_column_names = ['three_white_soldiers',
                ]

pattern_types = {'bullish' : ['bullish_engulfing', 'bullish_harami', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star', 'three_white_soldiers'],
                 'bearish' : ['bearish_engulfing', 'bearish_harami','dark_cloud_cover', 'gravestone_doji'],
                 'reversal' : ['dragonfly_doji', 'doji_star', 'piercing_pattern'],
                 'bullish_reversal' : ['morning_star', 'morning_star_doji'],
                 'bearish_reversal' : [],}


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
                start_day = timezone.now() - timedelta(days=365)
                finish_day = timezone.now()
                interval = '1D'

                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)

                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day,' to ', finish_day)

                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day)

                    #price_history = candlestick.bullish_engulfing(price_history, target='bullish_engulfing', ohlc=['Open','High','Low','Close'])
                    #price_history['bullish_engulfing'].fillna(False, inplace=True)
                    price_history = add_candle_data(price_history, candlestick_functions, column_names)
                    #price_history = db_candlestick.three_white_soldiers(price_history, target='three_white_soldiers',
                    #                                                    ohlc=['Open', 'High', 'Low', 'Close'])
                    #price_history['three_white_soldiers'].fillna(False, inplace=True)
                    price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)

                    count_patterns(price_history, pattern_types)
                    sr_levels, retests, last_high_low_level = find_levels(price_history, window=20)
                    ticker.last_high_low = last_high_low_level
                    ticker.save()
                    price_history = add_levels_to_price_history(price_history, sr_levels, retests)
                    price_history = add_ema_and_trend(price_history)

                    # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                    for index, row in price_history.iterrows():
                        if not DailyPrice.objects.filter(ticker=ticker, datetime=row['Datetime']).exists():
                            daily_price = DailyPrice(
                                ticker=ticker,
                                datetime=row['Datetime'],
                                open_price=row['Open'],
                                high_price=row['High'],
                                low_price=row['Low'],
                                close_price=row['Close'],
                                percent_change=row['PercentChange'],
                                volume=row['Volume'],
                                patterns_detected = row['patterns_detected'],
                                bullish_detected=row['bullish'],
                                bearish_detected=row['bearish'],
                                reversal_detected=row['reversal'],
                                bullish_reversal_detected=row['bullish_reversal'],
                                bearish_reversal_detected=row['bearish_reversal'],
                                level=row['level'],
                                level_type=row['level_type'],
                                level_strength=row['level_strength'],
                                ema_200=row['EMA_200'],
                                ema_50=row['EMA_50'],
                                trend=row['Trend'],
                            )
                        else:
                            daily_price = DailyPrice.objects.get(ticker=ticker, datetime=row['Datetime'])
                            daily_price.patterns_detected = row['patterns_detected']
                            daily_price.bullish_detected = row['bullish']
                            daily_price.bearish_detected = row['bearish']
                            daily_price.reversal_detected = row['reversal']
                            daily_price.bullish_reversal_detected = row['bullish_reversal']
                            daily_price.bearish_reversal_detected = row['bearish_reversal']
                            daily_price.level = row['level']
                            daily_price.level_type = row['level_type']
                            daily_price.level_strength = row['level_strength']
                            daily_price.ema_200 = row['EMA_200']
                            daily_price.ema_50 = row['EMA_50']
                            daily_price.trend = row['Trend']
                        daily_price.save()

            if ticker.is_fifteen_min:
                start_day = timezone.now() - timedelta(days=7)
                finish_day = timezone.now()
                interval = '15m'
                print('Checking 15 min data for', ticker.symbol)

                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)

                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day, ' to ', finish_day)

                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day)
                    price_history = add_candle_data(price_history, candlestick_functions, column_names)
                    price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)
                    count_patterns(price_history, pattern_types)

                    # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                    for index, row in price_history.iterrows():
                        if not FifteenMinPrice.objects.filter(ticker=ticker, datetime=row['Datetime']).exists():
                            fifteenmin_price = FifteenMinPrice(
                                ticker=ticker,
                                datetime=row['Datetime'],
                                open_price=row['Open'],
                                high_price=row['High'],
                                low_price=row['Low'],
                                close_price=row['Close'],
                                percent_change=row['PercentChange'],
                                volume=row['Volume'],
                                patterns_detected=row['patterns_detected'],
                                bullish_detected=row['bullish'],
                                bearish_detected=row['bearish'],
                                reversal_detected=row['reversal'],
                                bullish_reversal_detected=row['bullish_reversal'],
                                bearish_reversal_detected=row['bearish_reversal'],
                            )
                        else:
                            fifteenmin_price=FifteenMinPrice.objects.get(ticker=ticker, datetime=row['Datetime'])
                            fifteenmin_price.patterns_detected = row['patterns_detected']
                            fifteenmin_price.bullish_detected = row['bullish']
                            fifteenmin_price.bearish_detected = row['bearish']
                            fifteenmin_price.reversal_detected = row['reversal']
                            fifteenmin_price.bullish_reversal_detected = row['bullish_reversal']
                            fifteenmin_price.bearish_reversal_detected = row['bearish_reversal']
                        fifteenmin_price.save()
            if ticker.is_five_min:
                start_day = timezone.now() - timedelta(days=5)
                finish_day = timezone.now()
                interval = '5m'
                print('Checking 5 min data for', ticker.symbol)

                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)

                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day, ' to ', finish_day)

                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day)
                    price_history = add_candle_data(price_history, candlestick_functions, column_names)
                    price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)
                    count_patterns(price_history, pattern_types)

                    # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                    for index, row in price_history.iterrows():
                        if not FiveMinPrice.objects.filter(ticker=ticker, datetime=row['Datetime']).exists():
                            fivemin_price = FiveMinPrice(
                                ticker=ticker,
                                datetime=row['Datetime'],
                                open_price=row['Open'],
                                high_price=row['High'],
                                low_price=row['Low'],
                                close_price=row['Close'],
                                percent_change=row['PercentChange'],
                                volume=row['Volume'],
                                patterns_detected=row['patterns_detected'],
                                bullish_detected=row['bullish'],
                                bearish_detected=row['bearish'],
                                reversal_detected=row['reversal'],
                                bullish_reversal_detected=row['bullish_reversal'],
                                bearish_reversal_detected=row['bearish_reversal'],
                            )
                        else:
                            fivemin_price = FiveMinPrice.objects.get(ticker=ticker, datetime=row['Datetime'])
                            fivemin_price.patterns_detected = row['patterns_detected']
                            fivemin_price.bullish_detected = row['bullish']
                            fivemin_price.bearish_detected = row['bearish']
                            fivemin_price.reversal_detected = row['reversal']
                            fivemin_price.bullish_reversal_detected = row['bullish_reversal']
                            fivemin_price.bearish_reversal_detected = row['bearish_reversal']
                        fivemin_price.save()
            if ticker.is_one_min:
                pass
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
def update_metrics_view(request):
    update_ticker_metrics.update_ticker_metrics()
    return HttpResponseRedirect(reverse('ticker_config'))  # Redirect to admin dashboard or any other desired URL


def ticker_detail(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    daily_prices_query = DailyPrice.objects.filter(ticker=ticker, level__isnull=False).only('datetime', 'level', 'level_strength')

    # Fetching the most recent DailyPrice's close_price
    latest_candle = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
    latest_close_price = latest_candle.close_price if latest_candle else None
    # Computing the number of days from datetime to today for each DailyPrice instance
    current_date = date.today()
    daily_prices = []
    smallest_range_to_level = 100
    smallest_level_type = ''
    for dp in daily_prices_query:
        days_difference = (current_date - dp.datetime.date()).days
        if latest_close_price and latest_close_price != 0:
            close_price_percentage = (abs(dp.level-latest_close_price) / latest_close_price) * 100
            if close_price_percentage < smallest_range_to_level:
                smallest_range_to_level = close_price_percentage
                #if dp.close_price < dp.level:
                #    smallest_level_type = 'Resistance'
                #else:
                #    smallest_level_type = 'Support'
        else:
            close_price_percentage = None

        daily_prices.append({
            'daily_price': dp,
            'days_from_today': days_difference,
            'close_price_percentage': close_price_percentage,
            'latest_candle' : latest_candle,
        })
    if ticker.last_high_low != None:
        if smallest_range_to_level < 2:
            # Price is close to support / resistance level, so look back to the most recent high / low to determine if this is a support / resistance.
            if latest_close_price < ticker.last_high_low:
                smallest_level_type = 'Support'
            else:
                smallest_level_type = 'Resistance'
        else:
            # Price is far from support / resistance level, so just compare the close price to the support / resistance level.
            if latest_close_price > sr_level:
                smallest_level_type = 'Support'
            else:
                smallest_level_type = 'Resistance'

    context = {
        'ticker': ticker,
        'daily_prices': daily_prices,
        'close_price': latest_close_price,
        'smallest_level_type' : smallest_level_type,
        'smallest_range_to_level' : smallest_range_to_level,
    }
    return render(request, 'ticker_detail.html', context)

