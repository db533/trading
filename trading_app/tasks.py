from background_task import background
import logging
from .models import Ticker, DailyPrice, Params
from datetime import datetime, timedelta, timezone, date
import pytz
from .price_download import download_prices, download_daily_ticker_price
from time import sleep
import time
from .update_strategies import *
from .update_ticker_metrics import *


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
        logger.info(f'background_manual_category_download() starting for stocks in category "{str(category_name)}"...')
        logger.info(f'category: {str(category_name)}')
        category_name = category_name.replace('%20',' ')
        category_name = category_name.replace('%2520', ' ')
        logger.info(f'Cleaned category: {str(category_name)}')
        tickers = Ticker.objects.filter(categories__name=category_name)
        ticker_count = Ticker.objects.filter(categories__name=category_name).count()
        logger.info(f'ticker_count: {str(ticker_count)}')
        if ticker_count > tickers_for_throtlling:
            logger.info(f'Rate throttling will occur.')
            throttling = True
        else:
            logger.info(f'No rate throttling needed.')
            throttling = False

        # Iterate through all retrieved tickers and download prices.
        for ticker in tickers:
            background_manual_ticker_download(ticker.symbol, throttling)
            logger.info(f'{str(ticker.symbol)} price download requested in background...')
        logger.info(f'background_manual_category_download() completed. All price downloads created as background tasks.')

        logger.info(
            f'=========================================================================================')
        #time.sleep(15)
        #logger.info(f'Waited 15 seconds.')
    except Exception as e:
        message = f'Error occured in background_manual_category_download(). {e}'
        nj_param = Params.objects.get(key='night_job_end_dt')
        end_time = datetime.now()
        nj_param.value = end_time
        nj_param.save()
        nj_param = Params.objects.get(key='night_job_status_message')
        nj_param.value = message
        nj_param.save()
        logger.error(message)

from django.utils.timezone import now
from datetime import timedelta

@background(schedule=5)
def background_manual_ticker_download(ticker_symbol,throttling):
    # Retrieve a particular ticker price date and throttle if requested.
    strategies = [GannPointFourBuy2, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy,
                  GannPointEightSell,
                  GannPointThreeBuy, GannPointThreeSell, GannPointOneBuy, GannPointOneSell, GannPointNineBuy,
                  GannPointNineSell]

    try:
        logger.info(f'background_manual_ticker_download() starting for ticker "{str(ticker_symbol)}". throttling={str(throttling)}...')
        ticker = Ticker.objects.get(symbol=ticker_symbol)

        # Attempt to get the most recent DailyPrice instance for the Ticker
        try:
            latest_daily_price = DailyPrice.objects.filter(ticker=ticker).latest('datetime')
            last_update_time = latest_daily_price.datetime
            logger.info(f'last_update_time: "{str(last_update_time)}"')

            # Calculate the time difference between now and the last update
            current_time = now()
            logger.info(f'current_time: "{str(current_time)}"')
            time_difference = current_time - last_update_time
            logger.info(f'time_difference: "{str(time_difference)}"')

            # Check if 18 hours have passed since the last update
            if time_difference < timedelta(hours=28): # Adding 10 hours for the time difference
                logger.info(f'Less than 18 hours since the last update. Aborting task for "{ticker_symbol}".')
                return
            else:
                logger.info(f'More than 18 hours since the last update. Scheduling price download for "{ticker_symbol}".')
        except DailyPrice.DoesNotExist:
            # If no DailyPrice exists, proceed with the download
            logger.error(f'No DailyPrice record found for ticker "{ticker_symbol}". Proceeding with download.')

        start_time = display_local_time()  # record the start time of the loop

        # Download ticker prices
        logger.info(f'=====================================================================')
        logger.info(f'1. Downloading latest prices from background_manual_ticker_download()...')
        download_daily_ticker_price(timeframe='Daily', ticker_symbol=ticker.symbol, trigger='User')

        # Look for trading opportunities for this ticker
        logger.info(f'=====================================================================')
        logger.info(f'2. Updating metrics from background_manual_ticker_download()...')
        update_single_ticker_metrics(ticker.symbol)

        logger.info(f'=====================================================================')
        logger.info(f'3. Looking for valid trading strategies from background_manual_ticker_download()...')
        process_trading_opportunities_single_ticker(ticker.symbol, strategies)
        logger.info(f'Finished process_trading_opportunities_single_ticker() from background_manual_ticker_download() in tasks.py')

        end_time = display_local_time()  # record the end time of the loop
        elapsed_time = end_time - start_time  # calculate elapsed time
        logger.info(f'elapsed_time.total_seconds(): {str(elapsed_time.total_seconds())} secs...')

        if elapsed_time.total_seconds() < 20 and throttling == True:
            pause_duration = 20 - elapsed_time.total_seconds()
            logger.info(f'Rate throttling for {str(pause_duration)} secs...')
            sleep(pause_duration)
        logger.info(f'background_manual_ticker_download() completed.')
    except Exception as e:
        message = f'Error occured in background_manual_ticker_download(): {e}'
        nj_param = Params.objects.get(key='night_job_end_dt')
        end_time = datetime.now()
        nj_param.value = end_time
        nj_param.save()
        nj_param = Params.objects.get(key='night_job_status_message')
        nj_param.value = message
        nj_param.save()
        logger.error(message)


@background(schedule=5)
def delete_ticker(ticker_symbol):
    ticker_list = Ticker.objects.filter(symbol=ticker_symbol)
    for ticker in ticker_list:
        ticker.delete()
        # Assuming `logger` is correctly defined/imported elsewhere
        logger.info(f'Deleted ticker {ticker_symbol}.')  # Changed to info and fixed indentation

@background(schedule=5)
def background_update_ticker_strategies(ticker_symbol):
    strategies = [GannPointFourBuy2, GannPointFourSell, GannPointFiveBuy, GannPointFiveSell, GannPointEightBuy,
                  GannPointEightSell,
                  GannPointThreeBuy, GannPointThreeSell, GannPointOneBuy, GannPointOneSell, GannPointNineBuy,
                  GannPointNineSell]
    logger.info(
        f'background_update_ticker_strategies() starting for ticker "{str(ticker_symbol)}"...')
    ticker = Ticker.objects.get(symbol=ticker_symbol)
    process_trading_opportunities_single_ticker(ticker.symbol, strategies)
    logger.info(
        f'background_update_ticker_strategies() finished for ticker "{str(ticker_symbol)}".')

