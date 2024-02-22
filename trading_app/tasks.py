from background_task import background
import logging
from .models import Ticker
from datetime import datetime, timedelta, timezone, date
import pytz
from .price_download import download_prices
from time import sleep

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



@background(schedule=5)
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
        logger.error(f'background_manual_category_download() completed.')
    except Exception as e:
        logger.error(f'Error occured in background_manual_category_download(). {e}')