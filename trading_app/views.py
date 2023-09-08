from django.shortcuts import render
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import yfinance as yf
import pandas as pd
import time

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
from datetime import datetime, timedelta
from django.utils import timezone


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
        data = yf.Ticker(ticker.symbol).history(interval=interval, start=start_time, end=finish_time)
        if not data.empty:
            data['Ticker'] = ticker.symbol  # Add 'Ticker' column with the symbol

            # Create a 'Datetime' column from the index
            data['Datetime'] = data.index

            # Convert the datetime to a naive datetime
            data['Datetime'] = data['Datetime'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)

            # Set the timezone to your project's timezone (e.g., UTC)
            data['Datetime'] = data['Datetime'].apply(timezone.make_aware, timezone=timezone.utc)

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
    print('missing_dates:',missing_dates)
    return missing_dates

def edit_ticker(request, ticker_id):
    # Retrieve the Ticker instance to be edited or create a new one if it doesn't exist
    ticker = get_object_or_404(Ticker, id=ticker_id)

    if request.method == 'POST':
        form = TickerForm(request.POST, instance=ticker)
        if form.is_valid():
            if ticker.is_daily:
                start_day = timezone.now() - timedelta(days=31)
                finish_day = timezone.now()
                interval = '1D'
                print('Checking', ticker.symbol)

                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)

                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day,' to ', finish_day)

                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day)

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
                                volume=row['Volume']
                            )
                            daily_price.save()

            elif ticker.is_fifteen_min:
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
                                volume=row['Volume']
                            )
                            fifteenmin_price.save()
            elif ticker.is_five_min:
                pass
            elif ticker.is_one_min:
                pass
            form.save()

            return redirect('ticker_config')  # Redirect back to the configuration page
    else:
        form = TickerForm(instance=ticker)

    return render(request, 'edit_ticker.html', {'form': form, 'ticker': ticker})


from django.shortcuts import render
from .models import Ticker, DailyPrice


def daily_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    daily_prices = DailyPrice.objects.filter(ticker=ticker)

    return render(request, 'daily_price_list.html', {'ticker': ticker, 'daily_prices': daily_prices})

def fifteen_min_price_list(request, ticker_id):
    ticker = get_object_or_404(Ticker, id=ticker_id)
    fifteen_min_prices = FifteenMinPrice.objects.filter(ticker=ticker)

    return render(request, 'fifteen_min_price_list.html', {'ticker': ticker, 'fifteen_min_prices': fifteen_min_prices})

