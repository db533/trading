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
from datetime import datetime, timedelta, timezone
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from candlestick import candlestick
from . import db_candlestick

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
    tickers = Ticker.objects.all()

    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('ticker_config')

    else:
        form = TickerForm()

    return render(request, 'ticker_config.html', {'form': form, 'tickers': tickers})

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
            data['Datetime'] = data['Datetime'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)

            # Set the timezone to your project's timezone (e.g., UTC)
            #data['Datetime'] = data['Datetime'].apply(timezone.make_aware, timezone=timezone.utc)
            # Ensure the datetime index is in UTC timezone
            #data.index = data.index.tz_convert(timezone.utc)
            data = data.tz_localize(None)

            # Reorder the columns
            data = data[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

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

@login_required
def edit_ticker(request, ticker_id):
    # Retrieve the Ticker instance to be edited or create a new one if it doesn't exist
    ticker = get_object_or_404(Ticker, id=ticker_id)

    if request.method == 'POST':
        form = TickerForm(request.POST, instance=ticker)
        print('Checking', ticker.symbol)
        if form.is_valid():
            if ticker.is_daily:
                start_day = timezone.now() - timedelta(days=31)
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
                                volume=row['Volume'],
                                bullish_engulfing=row['bullish_engulfing'],
                                bullish_harami=row['bullish_harami'],
                                hammer=row['hammer'],
                                inverted_hammer=row['inverted_hammer'],
                                hanging_man=row['hanging_man'],
                                shooting_star=row['shooting_star'],
                                bearish_engulfing=row['bearish_engulfing'],
                                bearish_harami=row['bearish_harami'],
                                dark_cloud_cover=row['dark_cloud_cover'],
                                gravestone_doji=row['gravestone_doji'],
                                dragonfly_doji=row['dragonfly_doji'],
                                doji_star=row['doji_star'],
                                piercing_pattern=row['piercing_pattern'],
                                morning_star=row['morning_star'],
                                morning_star_doji=row['morning_star_doji'],
                                three_white_soldiers=row['three_white_soldiers']
                            )
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

                    price_history = db_candlestick.three_white_soldiers(price_history, target='three_white_soldiers',
                                                                        ohlc=['Open', 'High', 'Low', 'Close'])
                    price_history['three_white_soldiers'].fillna(False, inplace=True)


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
                                volume=row['Volume'],
                                bullish_engulfing=row['bullish_engulfing'],
                                bullish_harami=row['bullish_harami'],
                                hammer=row['hammer'],
                                inverted_hammer=row['inverted_hammer'],
                                hanging_man=row['hanging_man'],
                                shooting_star=row['shooting_star'],
                                bearish_engulfing=row['bearish_engulfing'],
                                bearish_harami=row['bearish_harami'],
                                dark_cloud_cover=row['dark_cloud_cover'],
                                gravestone_doji=row['gravestone_doji'],
                                dragonfly_doji=row['dragonfly_doji'],
                                doji_star=row['doji_star'],
                                piercing_pattern=row['piercing_pattern'],
                                morning_star=row['morning_star'],
                                morning_star_doji=row['morning_star_doji'],
                                three_white_soldiers=row['three_white_soldiers']
                            )
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

                    price_history = db_candlestick.three_white_soldiers(price_history, target='three_white_soldiers',
                                                                        ohlc=['Open', 'High', 'Low', 'Close'])
                    price_history['three_white_soldiers'].fillna(False, inplace=True)

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
                                volume=row['Volume'],
                                bullish_engulfing=row['bullish_engulfing'],
                                bullish_harami=row['bullish_harami'],
                                hammer=row['hammer'],
                                inverted_hammer=row['inverted_hammer'],
                                hanging_man=row['hanging_man'],
                                shooting_star=row['shooting_star'],
                                bearish_engulfing=row['bearish_engulfing'],
                                bearish_harami=row['bearish_harami'],
                                dark_cloud_cover=row['dark_cloud_cover'],
                                gravestone_doji=row['gravestone_doji'],
                                dragonfly_doji=row['dragonfly_doji'],
                                doji_star=row['doji_star'],
                                piercing_pattern=row['piercing_pattern'],
                                morning_star=row['morning_star'],
                                morning_star_doji=row['morning_star_doji'],
                                three_white_soldiers=row['three_white_soldiers']
                            )
                            fivemin_price.save()
            if ticker.is_one_min:
                pass
            form.save()

            return redirect('ticker_config')  # Redirect back to the configuration page
    else:
        form = TickerForm(instance=ticker)

    return render(request, 'edit_ticker.html', {'form': form, 'ticker': ticker})


from django.shortcuts import render
from .models import Ticker, DailyPrice

@login_required
def daily_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    daily_prices = DailyPrice.objects.filter(ticker=ticker)

    return render(request, 'price_list.html', {'ticker': ticker, 'candles': daily_prices, 'heading_text' : 'Daily'})

@login_required
def fifteen_min_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    fifteen_min_prices = FifteenMinPrice.objects.filter(ticker=ticker)

    return render(request, 'price_list.html', {'ticker': ticker, 'candles': fifteen_min_prices, 'heading_text' : '15 Minute'})

@login_required
def five_min_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    five_min_prices = FiveMinPrice.objects.filter(ticker=ticker)

    return render(request, 'price_list.html', {'ticker': ticker, 'candles': five_min_prices, 'heading_text' : '5 Minute'})