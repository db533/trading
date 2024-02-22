from background_task import background
import logging
from .models import Ticker, DailyPrice
from datetime import datetime, timedelta, timezone, date
import pytz
from .price_download import download_prices, download_daily_ticker_price
from time import sleep
import time

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
    return local_datetime

def background_manual_category_download(category_name):
    # Retrieve all tickers that are in the given category.
    try:
        tickers_for_throtlling = 195
        logger.error(f'background_manual_category_download() starting for stocks in category "{str(category_name)}"...')
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
        logger.error(f'background_manual_category_download() completed. All price downloads created as background tasks.')
        logger.error(
            f'=========================================================================================')
        #time.sleep(15)
        #logger.error(f'Waited 15 seconds.')
    except Exception as e:
        logger.error(f'Error occured in background_manual_category_download(). {e}')

from django.utils.timezone import now
from datetime import timedelta

@background(schedule=5)
def background_manual_ticker_download(ticker_symbol,throttling):
    # Retrieve a particular ticker price date and throttle if requested.
    try:
        logger.error(f'background_manual_ticker_download() starting for ticker "{str(ticker_symbol)}". throttling={str(throttling)}...')
        ticker = Ticker.objects.get(symbol=ticker_symbol)

        # Attempt to get the most recent DailyPrice instance for the Ticker
        try:
            latest_daily_price = DailyPrice.objects.filter(ticker=ticker).latest('datetime')
            last_update_time = latest_daily_price.datetime
            logger.error(f'last_update_time: "{str(last_update_time)}"')

            # Calculate the time difference between now and the last update
            current_time = now()
            logger.error(f'current_time: "{str(current_time)}"')
            time_difference = current_time - last_update_time
            logger.error(f'time_difference: "{str(time_difference)}"')

            # Check if 18 hours have passed since the last update
            if time_difference < timedelta(hours=18):
                logger.error(f'Less than 18 hours since the last update. Aborting task for "{ticker_symbol}".')
                return
            else:
                logger.error(f'More than 18 hours since the last update. Scheduling price download for "{ticker_symbol}".')
        except DailyPrice.DoesNotExist:
            # If no DailyPrice exists, proceed with the download
            logger.error(f'No DailyPrice record found for ticker "{ticker_symbol}". Proceeding with download.')

        start_time = display_local_time()  # record the start time of the loop
        download_daily_ticker_price(timeframe='Daily', ticker_symbol=ticker.symbol, trigger='User')

        end_time = display_local_time()  # record the end time of the loop
        elapsed_time = end_time - start_time  # calculate elapsed time
        logger.error(f'elapsed_time.total_seconds(): {str(elapsed_time.total_seconds())} secs...')

        if elapsed_time.total_seconds() < 20 and throttling == True:
            pause_duration = 20 - elapsed_time.total_seconds()
            logger.error(f'Rate throttling for {str(pause_duration)} secs...')
            sleep(pause_duration)
        logger.error(f'background_manual_ticker_download() completed.')
    except Exception as e:
        logger.error(f'Error occured in background_manual_ticker_download(): {e}')