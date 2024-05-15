import logging
from .update_strategies import *

logging.basicConfig(level=logging.INFO)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import yfinance as yf
from time import sleep
from .models import Ticker, DailyPrice, FifteenMinPrice, FiveMinPrice, TickerCategory, SwingPoint, Params, Trade

import pandas as pd
import pytz
from datetime import datetime, timedelta, timezone, date, time
from candlestick import candlestick
from . import db_candlestick
from django.utils import timezone
import logging
from decimal import Decimal
import math
from django.contrib.contenttypes.models import ContentType
import time

default_logger = logging.getLogger('django')
scheduled_logger = logging.getLogger('scheduled_tasks')

def format_elapsed_time(start_time, end_time):
    """
    Calculate elapsed time and format it as hours, minutes, and seconds.

    Parameters:
    - start_time: The start time (as returned by time.time()).
    - end_time: The end time (as returned by time.time()).

    Returns:
    - A string formatted as "X hours, Y minutes, Z seconds".
    """
    elapsed_seconds = int(end_time - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours} : {minutes}  : {seconds}"

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


def check_for_nat(logger, df):
    # Checking both 'Datetime' and 'Datetime_TZ' columns for NaT
    nat_indexes = df[df['Datetime'].isna()].index.tolist()
    nat_indexes_tz = df[df['Datetime_TZ'].isna()].index.tolist()

    # If NaT values are found, log the index values
    if nat_indexes:
        logger.info(f"NaT found in Datetime at index positions: {nat_indexes}")
    else:
        logger.info("No NaT values found in the Datetime column.")
    if nat_indexes_tz:
        logger.info(f"NaT found in Datetime_TZ at index positions: {nat_indexes_tz}")
    else:
        logger.info("No NaT values found in the Datetime_TZ column.")

    return nat_indexes

def get_largest_index_value(df):
    """
    Returns the largest index value of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame whose largest index value is required.

    Returns:
        The largest index value of the DataFrame.
    """
    if df.index.empty:
        return None  # Return None if the DataFrame is empty
    else:
        return df.index.max()

def get_price_data(ticker, interval, start_time, finish_time, logger):
    # Fetching existing data from the database
    try:
        existing_data = DailyPrice.objects.filter(ticker=ticker).values()
        if len(existing_data) > 0:
            existing_data_retrieved = True
        else:
            existing_data_retrieved = False
        if existing_data_retrieved == True:
            step = 1
            logger.info(f'get_price_data(). Retrieved existing data.')
            existing_df = pd.DataFrame.from_records(existing_data)
            #logger.info(
            #    f'step: {step} existing_df before any changes: existing_df.index[0]:{existing_df.index[0]} existing_df.iloc[0].to_dict(): {existing_df.iloc[0].to_dict()}')
            #logger.info(f'tz_info: {tz_info}')
            existing_df['Datetime_TZ'] = pd.to_datetime(existing_df['datetime_tz'])
            existing_df['Datetime'] = pd.to_datetime(existing_df['datetime'])
            existing_df['Open'] = existing_df['open_price'].astype(float)
            existing_df['High'] = existing_df['high_price'].astype(float)
            existing_df['Low'] = existing_df['low_price'].astype(float)
            existing_df['Close'] = existing_df['close_price'].astype(float)
            existing_df['Volume'] = existing_df['volume']

            # Check if there is at least one NaT in 'Datetime_TZ'
            #print('Step 1')
            if existing_df['Datetime_TZ'].isna().any():
                # Replace 'Datetime_TZ' column values with 'Datetime' column values
                #print('Step 2')
                existing_df['Datetime_TZ'] = existing_df['Datetime']

            #print('existing_df.tail(5) before set_index:')
            #print(existing_df.tail(5))

            #print('Step 3')
            try:
                existing_df['Datetime_index'] = existing_df['Datetime']
                existing_df = existing_df.set_index('Datetime_index')
            except Exception as e:
                logger.error(f"1. Error during loop: {e}")
            #print('Step 4')
            step = 1.5
            #logger.info(f'step = {step} Existing data:')
            #logger.info(f'existing_df.columns:{existing_df.columns}')
            #logger.info(f'existing_df.iloc[0].to_dict():{str(existing_df.iloc[0].to_dict())}')
            existing_df = existing_df.drop(columns=['datetime', 'datetime_tz', 'ticker_id', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
            step=1.7
            #logger.info(f'step = {step} Existing data:')
            #logger.info(f'existing_df.columns:{existing_df.columns}')
            #print('existing_df.head(5) after set_index:')
            #print(existing_df.head(5))
            #print('existing_df.tail(5) after set_index:')
            #print(existing_df.tail(5))
            # Filter out records that are newer than start_time
            existing_df = existing_df[existing_df.index < pd.to_datetime(start_time)]
            step = 2
            #logger.info(f'step = {step} Existing data:')
            #logger.info(f'len(existing_df):{len(existing_df)}')
            #logger.info(f'existing_df.columns:{existing_df.columns}')
            #logger.info(f'existing_df.index[0]:{existing_df.index[0]}')
            #logger.info(f'existing_df.iloc[0].to_dict():{str(existing_df.iloc[0].to_dict())}')
            tzinfo = []
            datetime_tzinfo = []
            for x in range(len(existing_df)):
                tzinfo.append(existing_df.iloc[x]["Datetime_TZ"])
                datetime_tzinfo.append(existing_df.iloc[x]["Datetime"])
            #logger.info(f'len(tzinfo): {len(tzinfo)}')
            #logger.info(f'existing_df["Datetime_TZ"] values: {tzinfo}')
            for handler in logger.handlers:
                handler.flush()
            #logger.info(f'len(datetime_tzinfo): {len(datetime_tzinfo)}')
            #logger.info(f'existing_df["Datetime"] values: {datetime_tzinfo}')

    except Exception as e:
        print(f"Error fetching existing data for {ticker.symbol}: {e}")
        existing_df = pd.DataFrame(columns=['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
    logger.info(f'get_price_data(). Existing data retrieval is complete.')

    try:
        # Ensure start_time and finish_time are timezone-aware
        start_time = start_time.replace(tzinfo=timezone.utc)
        finish_time = finish_time.replace(tzinfo=timezone.utc)
        step = 3
        data = yf.Ticker(ticker.symbol).history(interval=interval, start=start_time, end=finish_time)
        #logger.info(f'step = {step} get_price_data(). data retrieval performed.')
        #logger.info(f'data as received from Yahoo: data.index[0]:{data.index[0]} data.iloc[0].to_dict(): {data.iloc[0].to_dict()}')
        if not data.empty:
            print('Retrieve new price data records...')

            # Create a 'Datetime' column from the index
            data['Datetime'] = data.index.tz_convert('UTC').tz_localize(None)
            data['Datetime_TZ'] = data.index.tz_convert('UTC').tz_localize(None)
            #data.index = data.index.tz_convert('UTC')  # Making index tz-naive
            data.index = data.index.tz_convert('UTC').tz_localize(None)
            #logger.info(f'step = {step}')
            #logger.info(f'data["Datetime"].dtype): {data["Datetime"].dtype}')
            #logger.info(f'data["Datetime_TZ"].dtype): {data["Datetime_TZ"].dtype}')

            # Convert the first row to a dictionary
            step = 4
            data_len=len(data)
            #logger.info(f'step: {step} Data downloaded from Yahoo:')
            #logger.info(f'data.index[0]: {data.index[0]}')
            #logger.info(f'data.iloc[0].to_dict(): {data.iloc[0].to_dict()}')
            #logger.info(f'data_len: {data_len}')
            #logger.info(f'data.index[data_len-1]: {data.index[data_len-1]}')
            #logger.info(f'data.iloc[data_len-1].to_dict(): {data.iloc[data_len-1].to_dict()}')
            #logger.info(f'get_largest_index_value(data): {get_largest_index_value(data)}')
            #logger.info(f'data["Datetime"].dtype): {data["Datetime"].dtype}')
            #logger.info(f'data["Datetime_TZ"].dtype): {data["Datetime_TZ"].dtype}')

            #logger.info(f'[B] About to check for NaT in data in Datetime and Datetime_TZ...')
            #check_for_nat(logger, data)
            step = 5
        else:
            logger.info(f'get_price_data(). Retrieved data is empty.')
            step = 6
        step = 7
        if existing_data_retrieved == True:
            #logger.info(f'step = {step}')
            #logger.info(f'existing_df["Datetime"].dtype): {existing_df["Datetime"].dtype}')
            #logger.info(f'existing_df["Datetime_TZ"].dtype): {existing_df["Datetime_TZ"].dtype}')
            existing_df.index = existing_df.index.tz_convert('UTC').tz_localize(None)
            step = 8
            existing_df['Datetime'] = existing_df['Datetime'].dt.tz_localize(None)
            existing_df['Datetime_TZ'] = existing_df['Datetime_TZ'].dt.tz_localize(None)
            #logger.info(f'step = {step}')
            #logger.info(f'existing_df["Datetime"].dtype): {existing_df["Datetime"].dtype}')
            #logger.info(f'existing_df["Datetime_TZ"].dtype): {existing_df["Datetime_TZ"].dtype}')

            step = 9
            #logger.info(f'step = {step} existing_df.index[0]: {existing_df.index[0]} existing_df.iloc[0].to_dict(): {existing_df.iloc[0].to_dict()}')
            #logger.info(f'get_largest_index_value(data): {get_largest_index_value(data)}')
            #logger.info(f'get_largest_index_value(existing_df): {get_largest_index_value(existing_df)}')
            #logger.info(f'existing_df["Datetime"].dtype): {existing_df["Datetime"].dtype}')
            #logger.info(f'existing_df["Datetime_TZ"].dtype): {existing_df["Datetime_TZ"].dtype}')
            #logger.info(f'existing_df.index.dtype): {existing_df.index.dtype}')
            #logger.info(f'data["Datetime"].dtype): {data["Datetime"].dtype}')
            #logger.info(f'data["Datetime_TZ"].dtype): {data["Datetime_TZ"].dtype}')
            #logger.info(f'data.index.dtype): {data.index.dtype}')

            #logger.info(f'[C] About to check for NaT in in existing_df in Datetime and Datetime_TZ...')
            #check_for_nat(logger, existing_df)
            step = 10
            combined_data = pd.concat([existing_df, data]).sort_index().drop_duplicates()
            combined_data['Datetime'] = pd.to_datetime(combined_data['Datetime'])
            combined_data['Datetime_TZ'] = pd.to_datetime(combined_data['Datetime_TZ'])
            combined_data['Datetime'] = combined_data['Datetime'].dt.tz_localize(None)
            combined_data['Datetime_TZ'] = combined_data['Datetime_TZ'].dt.tz_localize(None)
            data_len = len(combined_data)
            #logger.info(f'data_len: {data_len}')
            #logger.info(f'combined_data.index[data_len-1]: {combined_data.index[data_len - 1]}')
            #logger.info(f'combined_data.iloc[data_len-1].to_dict(): {combined_data.iloc[data_len - 1].to_dict()}')
            #logger.info(f'combined_data["Datetime"].dtype): {combined_data["Datetime"].dtype}')
            #logger.info(f'combined_data["Datetime_TZ"].dtype): {combined_data["Datetime_TZ"].dtype}')
            #logger.info(f'[D] About to check for NaT in in combined_data in Datetime and Datetime_TZ...')
            #check_for_nat(logger, combined_data)
            step = 11
            #logger.info(
            #    f'step = {step} combined_data.index[0]: {combined_data.index[0]} combined_data.iloc[0].to_dict(): {combined_data.iloc[0].to_dict()}')
            #combined_data.iloc[0]['Datetime'] = combined_data.iloc[0]['Datetime'].dt.tz_localize(None)
            #combined_data.iloc[0]['Datetime_TZ'] = combined_data.iloc[0]['Datetime_TZ'].dt.tz_localize(None)
            #step = 11.3
            #logger.info(
            #    f'step = {step} combined_data.index[0]: {combined_data.index[0]} combined_data.iloc[0].to_dict(): {combined_data.iloc[0].to_dict()}')
            #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(combined_data)}')
            combined_data['Datetime'] = pd.to_datetime(combined_data['Datetime'])
            combined_data['Datetime_TZ'] = pd.to_datetime(combined_data['Datetime_TZ'])
            #logger.info(f'combined_data["Datetime"].dtype): {combined_data["Datetime"].dtype}')
            #logger.info(f'combined_data["Datetime_TZ"].dtype): {combined_data["Datetime_TZ"].dtype}')

            combined_data['Datetime'] = combined_data['Datetime'].dt.tz_localize(None)
            combined_data['Datetime_TZ'] = combined_data['Datetime_TZ'].dt.tz_localize(None)
            #logger.info(f'[E] About to check for NaT in in combined_data in Datetime and Datetime_TZ...')
            check_for_nat(logger, combined_data)
            step = 11.5
            #logger.info(f'step: {step} get_price_data(). Combined with existing data.')
        else:
            step = 12
            combined_data = data
            step = 13
            #logger.info(f'step: {step} get_price_data(). No existing data to combine new data with.')
        # Convert the first row to a dictionary
        first_row_dict = combined_data.iloc[0].to_dict()

        # Display the dictionary
        print(first_row_dict)
        step = 14
        #logger.info(f'step: {step} combined_data.index[0]: {combined_data.index[0]}')
        #logger.info(f'combined_data: {first_row_dict}')
        tzinfo = []
        for x in range(len(combined_data)):
            tzinfo.append(combined_data.iloc[x]["Datetime"])
        #logger.info(f'combined_data["Datetime"] values: {tzinfo}')
        tzinfo = []
        for x in range(len(combined_data)):
            tzinfo.append(combined_data.iloc[x]["Datetime_TZ"])
        #logger.info(f'combined_data["Datetime_TZ"] values: {tzinfo}')
        tzinfo = []
        for x in range(len(combined_data)):
            tzinfo.append(combined_data.index[x])
        #logger.info(f'combined_data.index values: {tzinfo}')
        combined_data = combined_data.loc[~combined_data.index.duplicated(keep='last')]
        #logger.info(f'[F] About to check for NaT in in combined_data in Datetime and Datetime_TZ...')
        #check_for_nat(logger, combined_data)
        step = 15
        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(combined_data)}')
        #logger.info(f'step: {step} combined_data.index[0]: {combined_data.index[0]}')
        #logger.info(f'combined_data["Datetime_TZ"].iloc[0]: {combined_data["Datetime_TZ"].iloc[0]}')
        #logger.info(f'combined_data.iloc[0].to_dict(): {combined_data.iloc[0].to_dict()}')
        combined_data.sort_values(by='Datetime_TZ', inplace=True)
        step = 16
        combined_data['Ticker'] = ticker.symbol
        step = 17
        combined_data['PercentChange'] = combined_data['Close'].pct_change() * 100
        step = 18
        combined_data.at[combined_data.index[0], 'PercentChange'] = 0
        step = 19
        combined_data.dropna(subset=['Open'], inplace=True)
        step = 20
        combined_data = combined_data[
            ['Datetime_TZ', 'Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'PercentChange']]
        step = 21
        #combined_data['Datetime_TZ'] = combined_data['Datetime_TZ'].dt.tz_localize("UTC")
        combined_data.index = combined_data.index.tz_localize(None)
        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(combined_data)}')
        #logger.info(f'step: {step} combined_data.index[0]: {combined_data.index[0]}')
        #logger.info(f'combined_data.iloc[0].to_dict(): {combined_data.iloc[0].to_dict()}')
        #logger.info(f'[G] About to check for NaT in in combined_data in Datetime and Datetime_TZ...')
        check_for_nat(logger, combined_data)

    except Exception as e:
        logger.error(f"Error downloading data for {ticker.symbol}: Step: {step}. {e}")
        combined_data = pd.DataFrame(
            columns=['Datetime', 'Datetime_TZ', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return combined_data

def get_missing_dates(ticker, interval, start_day, finish_day, hour_offset, logger):
    # Get the list of dates missing in DailyPrice for the given ticker within the date range
    #print('start_day:',start_day,'timezone.make_naive(start_day):',timezone.make_naive(start_day))
    #print('finish_day:', finish_day, 'timezone.make_naive(finish_day):', timezone.make_naive(finish_day))
    if interval == '1D':
        existing_dates = DailyPrice.objects.filter(
            ticker=ticker, datetime__range=(start_day, finish_day)
        ).values_list('datetime', flat=True)
        logger.info(f'get_missing_dates. start_day: {str(start_day)} finish_day: {str(finish_day)}')

        try:
            # Ensure existing_dates is a list of timezone-naive or aware datetime objects
            existing_dates = [timezone.make_naive(date) for date in existing_dates]
            logger.info(f'existing_dates: {existing_dates[:3]}')
        except Exception as e:
            logger.error(f'Failed to create existing_dates: {e}')
        # Ensure that start_day and finish_day have the time set to 4:00
        start_day_with_time = datetime.combine(start_day, time(hour_offset, 0))
        finish_day_with_time = datetime.combine(finish_day, time(hour_offset, 0))
        logger.info(f'start_day_with_time: {start_day_with_time}')

        all_dates = pd.date_range(start=start_day_with_time, end=finish_day_with_time, freq='D')
        all_dates = all_dates.tz_localize(None)

        # Ensure all_dates is a list of native python datetime objects
        all_dates = [date.to_pydatetime() for date in all_dates]

        logger.info(f'all_dates: {all_dates[:3]}')

        missing_dates = [date for date in all_dates if date not in existing_dates]

        #print('missing_dates:', missing_dates[:3])
        logger.info(f'missing_dates: {missing_dates}')
        #print('max(missing_dates):',max(missing_dates))
    if interval == '15m':
        existing_dates = FifteenMinPrice.objects.filter(ticker=ticker,
                                                        datetime__range=(start_day, finish_day)).values_list(
            'datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='15T')
        all_dates.tz_localize(None)
        missing_dates = [date for date in all_dates if date not in existing_dates]
        # print('all_dates:',all_dates)ķķ
    if interval == '5m':
        existing_dates = FiveMinPrice.objects.filter(ticker=ticker,
                                                     datetime__range=(start_day, finish_day)).values_list(
            'datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='5T')
        all_dates.tz_localize(None)
        missing_dates = [date for date in all_dates if date not in existing_dates]
        # print('all_dates:',all_dates)
        # print('missing_dates:',missing_dates)
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
        # print('column_name:',column_name)
        # print(price_history[column_name])
    return price_history


# List of candlestick functions to replace 'candlestick.xxx'
candlestick_functions = [candlestick.bullish_engulfing, candlestick.bullish_harami, candlestick.hammer,
                         candlestick.inverted_hammer,
                         candlestick.hanging_man, candlestick.shooting_star, candlestick.bearish_engulfing,
                         candlestick.bearish_harami,
                         candlestick.dark_cloud_cover, candlestick.gravestone_doji, candlestick.dragonfly_doji,
                         candlestick.doji_star,
                         candlestick.piercing_pattern, candlestick.morning_star, candlestick.morning_star_doji,
                         # candlestick.evening_star, candlestick.evening_star_doji
                         ]

# List of column names to replace 'xxx' in price_history
column_names = ['bullish_engulfing', 'bullish_harami', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
                'bearish_engulfing', 'bearish_harami',
                'dark_cloud_cover', 'gravestone_doji', 'dragonfly_doji', 'doji_star', 'piercing_pattern',
                'morning_star', 'morning_star_doji',
                # 'evening_star', 'evening_star_doji'
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
    # def find_levels(df, columns=['Close'], window=20, retest_threshold_percent=0.001):
    df.index = pd.to_datetime(df.index, utc=True)
    support = {}
    resistance = {}
    sr_level = {}
    last_high_low_level = None
    retests = {}  # Keep track of retest counts and last retest datetime for each level
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
                #print('Future level close to level1:', future_level)
                close_levels.append(future_level)
        # print('Level1:', level1, 'Level2:',level2, 'abs(level1 - level2):',abs(level1 - level2),'threshold_value:',level1 * threshold_percent, 'Within threshold?',abs(level1 - level2) <= level1 * threshold_percent)
        return close_levels

    # Remove sr_levels that are within threshold of a previous support level
    #print('sr_level:', sr_level)
    sr_keys_sorted = sorted(sr_level, key=sr_level.get)  # sort by datetime
    # print('sr_keys_sorted:',sr_keys_sorted)
    for i in range(len(sr_keys_sorted)):
        level1 = sr_keys_sorted[i]
        if level1 in sr_level:
            # print('Level1:', level1)
            compared_levels = sr_keys_sorted[i + 1:]
            close_levels = within_threshold(level1, compared_levels, retest_threshold_percent)
            if len(close_levels) > 0:
                # print('close_levels:', close_levels)
                all_levels = [level1] + close_levels
                # print('all_levels:', all_levels)
                avg_level = sum(all_levels) / len(all_levels)
                #print('level1 changed from', level1, 'to avg_level', avg_level)
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
                # if abs(level - current_close) <= threshold or abs(level - current_open) <= threshold:
                if abs(level - current_close) <= threshold:
                    # Handle retests
                    retests.setdefault(level, {'count': 0, 'last_retest': None})
                    retests[level]['count'] += 1
                    retests[level]['last_retest'] = current_datetime
                    #print(f"Support/resistance level {level} retested at {current_datetime}.")

    return sr_level, retests, last_high_low_level

from decimal import Decimal

def identify_highs_lows_gann(ticker, df, logger, reversal_days=2, price_move_percent=1.5):
    # New function to compute swing points using WD Gann logic.
    print('Detecting swing points according to Gann logic...')
    logger.info(f'Detecting swing points according to Gann logic...')
    logger.info(f'reversal_days: {reversal_days}')
    df['swing_point_label'] = ''
    df['swing_point_price'] = 0
    df['swing_point_price'] = df['swing_point_price'].astype(float)
    df['swing_point_current_trend'] = 0
    df['candle_count_since_last_swing_point'] = 0
    # print('df.columns:',df.columns)
    print('len(df):', len(df))

    # Delete existing swing points for the ticker as new ones will be determined.
    print('ticker:', ticker)
    try:
        healthy_bullish_count = 0
        healthy_bearish_count = 0
        current_trend_seq_count = 0

        df['healthy_bullish_candle'] = ((df['Close'] > df['Open'] * (1 + price_move_percent)) &
                                        ((df['Close'] - df['Open']) / (df['High'] - df['Low']) > 0.6)).astype(int)
        df['healthy_bearish_candle'] = ((df['Open'] > df['Close'] * (1 + price_move_percent)) &
                                        ((df['Open'] - df['Close']) / (df['High'] - df['Low']) > 0.6)).astype(int)

        # Determine initial direction: If price closes higher, up trend, if lower, down trend
        # Up trend continues until low is lower on next 2 days. (If reversal_days = 2)
        # Down trend continues until high is higher on next 2 days.
        for i in range(0, len(df) - reversal_days ):
            if i == 0:
                # This is start of data. Determine the initial direction.
                index_label = df.index[0]
                first_close = df.iloc[0]['Close']
                second_close = df.iloc[1]['Close']
                if first_close < second_close:
                    # Upward movement over first 2 candles. Assume up trend
                    uptrend_in_progress = True
                    last_swing_point = 'HL'
                else:
                    uptrend_in_progress = False
                    last_swing_point = 'LH'
                last_high_reference = df.iloc[0]['High']
                last_low_reference = df.iloc[0]['Low']
                candle_count_since_last_swing_point = 0
            index_label = df.index[i]
            window_slice = df.iloc[i:i + reversal_days + 1]
            swing_point_occured = False
            candle_count_since_last_swing_point += 1
            if uptrend_in_progress:
                # First check if a new high was not achieved (but only if this is not the first record).
                high_day_0 = df.iloc[i]['High']
                tomorrow_high = df.iloc[i+1]['High']
                if high_day_0 > tomorrow_high:
                    # High today ends tomorrow. Check for reversal in subsequent candles.
                    # Look for reversal to downtrend
                    # Up trend continues until low is lower on next 2 days. (If reversal_days = 2)
                    # Assume the consequetive lows will be lower and test each to check they are indeed.
                    # If they are not, then this is not a swing point.
                    count_consequetive = 0
                    for x in range(0, reversal_days):
                        if window_slice.iloc[0]['Low'] > window_slice.iloc[x+1]['Low'] and window_slice.iloc[0]['High'] > window_slice.iloc[x+1]['High']:
                            count_consequetive += 1
                    #print(index_label,' count_consequetive:',count_consequetive)
                    if count_consequetive == reversal_days:
                        # The subsequent day(s) have had lows lower than on Day 0.
                        # Up trend ended on Day 0.
                        if high_day_0 > last_high_reference:
                            # This high is higher than the previous high so this is a HH.
                            df.at[index_label, 'swing_point_label'] = "HH"
                            if last_swing_point == 'HL':
                                current_trend_seq_count += 1
                            if last_swing_point == 'LL':
                                current_trend_seq_count = 1
                            last_swing_point = 'HH'
                        else:
                            df.at[index_label, 'swing_point_label'] = "LH"
                            if last_swing_point == 'HL':
                                current_trend_seq_count = 1
                            if last_swing_point == 'LL':
                                current_trend_seq_count += 1
                            last_swing_point = 'LH'
                        df.at[index_label, 'swing_point_price'] = high_day_0
                        uptrend_in_progress = False  # Now trend is downward.
                        last_high_reference = high_day_0
                        swing_point_occured = True
            else:
                # Down trend in progress
                low_day_0 = df.iloc[i]['Low']
                tomorrow_low = df.iloc[i+1]['Low']
                if low_day_0 < tomorrow_low:
                    # Today's low is lower than tomorrow's, so look for reversal to uptrend
                    # Down trend continues until high is higher on next 2 days.
                    # Assume the consequetive highs will be higher and test each to check they are indeed.
                    # If they are not, then this is not a swing point.
                    #logger.info(f"i: {i} Downtrend.")
                    count_consequetive = 0
                    #logger.info(f"len(window_slice): {len(window_slice)}")
                    #logger.info(f"window_slice['High']: {window_slice['High']}")
                    for x in range(0, reversal_days):
                        #logger.info(f"window_slice.iloc[x]['High']: {window_slice.iloc[x]['High']} window_slice.iloc[x+1]['High']: {window_slice.iloc[x+1]['High']}")
                        if window_slice.iloc[0]['High'] < window_slice.iloc[x+1]['High'] and window_slice.iloc[0]['Low'] < window_slice.iloc[x+1]['Low']:
                            #logger.info(f"NOT A SWING POINT. x: {x}")
                            count_consequetive += 1
                    #print(index_label, ' count_consequetive:', count_consequetive)
                    if count_consequetive == reversal_days:
                        # The subsequent day(s) have had higher highs than on Day 0.
                        # Down trend ended on Day 0.
                        if low_day_0 > last_low_reference:
                            # This low is higher than the previous high so this is a HL.
                            df.at[index_label, 'swing_point_label'] = "HL"
                            if last_swing_point == 'HH':
                                current_trend_seq_count += 1
                            if last_swing_point == 'LH':
                                current_trend_seq_count = 1
                            last_swing_point = 'HL'
                        else:
                            df.at[index_label, 'swing_point_label'] = "LL"
                            if last_swing_point == 'HH':
                                current_trend_seq_count = 1
                            if last_swing_point == 'LH':
                                current_trend_seq_count += 1
                            last_swing_point = 'LL'
                        uptrend_in_progress = True  # Now trend is upward.
                        df.at[index_label, 'swing_point_price'] = low_day_0
                        last_low_reference = low_day_0
                        swing_point_occured = True
            if swing_point_occured == True:
                df.at[index_label, 'healthy_bullish_candle'] = healthy_bullish_count
                df.at[index_label, 'healthy_bearish_candle'] = healthy_bearish_count
                df.at[index_label, 'candle_count_since_last_swing_point'] = candle_count_since_last_swing_point
                candle_count_since_last_swing_point = 0
                final_swing_point_trend = df.loc[index_label, 'swing_point_current_trend']
                if current_trend_seq_count > 2:
                    # This data point is part of a swing trend.
                    if last_swing_point[0] == "H":
                        df.at[index_label, 'swing_point_current_trend'] = 1
                        # print('Setting uptrend.')
                    elif last_swing_point[0] == "L":
                        df.at[index_label, 'swing_point_current_trend'] = -1
                        # print('Setting downtrend.')
            else:
                # Not a swing point, so reset the healthy candle count to 0 for this data point
                df.at[index_label, 'healthy_bullish_candle'] = 0
                df.at[index_label, 'healthy_bearish_candle'] = 0

        return df, final_swing_point_trend
    except Exception as e:
        print(f"Error in identify_highs_lows_gann() for {ticker.symbol}: {e}")

def identify_highs_lows_gann2(ticker, df, logger, reversal_days=2, price_move_percent=1.5):
    # New function to compute swing points using WD Gann logic.
    print('Detecting swing points according to Gann logic...')
    logger.info(f'Detecting swing points according to Gann logic...')
    logger.info(f'reversal_days: {reversal_days}')
    #logger.info(f'df.columns: {df.columns}')
    try:
        df['swing_point_label'] = ''
        df['swing_point_price'] = 0
        df['magnitude'] = 0
        df['swing_point_price'] = df['swing_point_price'].astype(float)
        df['swing_point_current_trend'] = 0
        df['candle_count_since_last_swing_point'] = 0
        # print('df.columns:',df.columns)
        print('len(df):', len(df))
        #logger.info(f'len(df): {len(df)}')

        # Delete existing swing points for the ticker as new ones will be determined.
        print('ticker:', ticker)

        healthy_bullish_count = 0
        healthy_bearish_count = 0
        current_trend_seq_count = 0

        df['healthy_bullish_candle'] = ((df['Close'] > df['Open'] * (1 + price_move_percent)) &
                                        ((df['Close'] - df['Open']) / (df['High'] - df['Low']) > 0.6)).astype(int)
        df['healthy_bearish_candle'] = ((df['Open'] > df['Close'] * (1 + price_move_percent)) &
                                        ((df['Open'] - df['Close']) / (df['High'] - df['Low']) > 0.6)).astype(int)
        #logger.info(f'Step 1')
        # Determine initial direction: If price closes higher, up trend, if lower, down trend
        # Up trend continues until low is lower on next 2 days. (If reversal_days = 2)
        # Down trend continues until high is higher on next 2 days.
        logger.info(f'len(df) - reversal_days: {len(df) - reversal_days}')
        for i in range(0, len(df) - reversal_days ):
            #logger.info(f'Step 2. i={i}')
            if i == 0:
                # This is start of data. Determine the initial direction.
                index_label = df.index[0]
                first_close = df.iloc[0]['Close']
                second_close = df.iloc[1]['Close']
                if first_close < second_close:
                    # Upward movement over first 2 candles. Assume up trend
                    uptrend_in_progress = True
                    last_swing_point = 'HL'
                else:
                    uptrend_in_progress = False
                    last_swing_point = 'LH'
                last_high_reference = df.iloc[0]['High']
                last_low_reference = df.iloc[0]['Low']
                candle_count_since_last_swing_point = 0
            index_label = df.index[i]
            #logger.info(f'Step 3. index_label={index_label}')
            window_slice = df.iloc[i:i + reversal_days + 1]
            swing_point_occured = False
            candle_count_since_last_swing_point += 1
            average_prior_vol_slice = df.iloc[max(i-2,0):i+1]
            average_volume = average_prior_vol_slice['Volume'].mean()
            #logger.info(f'{index_label}: Open: {df.iloc[i]["Open"]}, High: {df.iloc[i]["High"]}, Low: {df.iloc[i]["Low"]}, Close: {df.iloc[i]["Close"]}')
            if uptrend_in_progress:
                # First check if a new high was not achieved (but only if this is not the first record).
                #logger.info(f'Uptrend in progress... {index_label}')
                high_day_0 = df.iloc[i]['High']
                low_day_0 = df.iloc[i]['Low']
                tomorrow_high = df.iloc[i+1]['High']
                tomorrow_low = df.iloc[i + 1]['Low']
                #logger.info(
                #    f'high_day_0={high_day_0}, tomorrow_high={tomorrow_high}, low_day_0={low_day_0}, tomorrow_low={tomorrow_low}')
                #logger.info(f'Up Step 6.')
                if high_day_0 > tomorrow_high and low_day_0 > tomorrow_low:
                    #logger.info(f'Up Step 7A. Highs and lows lower tomorrow')
                    # High today ends tomorrow and low also lower. Check for confirmations.
                    change_confirmations = 0

                    # Is the close on day_0 in the lower quartile of the day's range?
                    lower_quartile_price = ((high_day_0 - low_day_0) * 0.25) + low_day_0
                    if df.iloc[i]['Close'] <= lower_quartile_price:
                        # Price closed in lower quartile on potential swing point day.
                        #logger.info(f'Up Step 7B. df.iloc[i][Close] <= lower_quartile_price')
                        change_confirmations +=1

                    # Is Day 1 close less than the low of Day 0?
                    if df.iloc[i+1]['Close'] < low_day_0:
                        #logger.info(f'Up Step 7C. df.iloc[i+1][Close] < low_day_0')
                        change_confirmations += 1

                    # If trading volume on day 1 > than on Day 0?
                    if df.iloc[i + 1]['Volume'] > average_volume*1.2:
                        #logger.info(f'Up Step 7D. df.iloc[i + 1][Volume] > average_volume*1.2')
                        change_confirmations += 1

                    # Is there a bearish or reversal candle on Day 1?
                    if df.iloc[i + 1]['bearish'] > 0 or df.iloc[i + 1]['bearish_reversal'] > 0 or df.iloc[i + 1]['reversal'] > 0:
                        #logger.info(f'Up Step 7E. df.iloc[i + 1][bearish] > 0 or df.iloc[i + 1][bearish_reversal] > 0 or df.iloc[i + 1][reversal] > 0')
                        change_confirmations += 1
                    #logger.info(f'change_confirmations = {change_confirmations}')

                    if change_confirmations >= 2:
                        #logger.info(f'Up Step 8. change_confirmations >= 2')
                        # At least 2 signals confirm the swing point.
                        if high_day_0 > last_high_reference:
                            # This high is higher than the previous high so this is a HH.
                            #logger.info(f'Up Step 8A. high_day_0 > last_high_reference. SP = HH')
                            df.at[index_label, 'swing_point_label'] = "HH"
                            if last_swing_point == 'HL':
                                current_trend_seq_count += 1
                            if last_swing_point == 'LL':
                                current_trend_seq_count = 1
                            last_swing_point = 'HH'
                        else:
                            #logger.info(f'Up Step 8B. high_day_0 <= last_high_reference. SP = LH')
                            df.at[index_label, 'swing_point_label'] = "LH"
                            if last_swing_point == 'HL':
                                current_trend_seq_count = 1
                            if last_swing_point == 'LL':
                                current_trend_seq_count += 1
                            last_swing_point = 'LH'
                        df.at[index_label, 'swing_point_price'] = high_day_0
                        uptrend_in_progress = False  # Now trend is downward.
                        last_high_reference = high_day_0
                        df.at[index_label, 'magnitude'] = 1
                        swing_point_occured = True
            else:
                # Down trend in progress
                #logger.info(f'Step 9. Downtrend in progress... {index_label}')
                high_day_0 = df.iloc[i]['High']
                low_day_0 = df.iloc[i]['Low']
                tomorrow_high = df.iloc[i + 1]['High']
                tomorrow_low = df.iloc[i + 1]['Low']
                #logger.info(f'Step 10.')
                #logger.info(
                #    f'high_day_0={high_day_0}, tomorrow_high={tomorrow_high}, low_day_0={low_day_0}, tomorrow_low={tomorrow_low}')
                if high_day_0 < tomorrow_high and low_day_0 < tomorrow_low:
                    # Low today ends tomorrow and high also higher. Check for confirmations.
                    #logger.info(f'Step 11. Highs and lows are both higher tomorrow.')
                    change_confirmations = 0

                    # Is the close on day_0 in the lower quartile of the day's range?
                    upper_quartile_price = high_day_0 - ((high_day_0 - low_day_0) * 0.25)
                    if df.iloc[i]['Close'] >= upper_quartile_price:
                        # Price closed in upper quartile on potential swing point day.
                        #logger.info(f'Step 11A. df.iloc[i][Close] >= upper_quartile_price')
                        change_confirmations += 1

                    # Is Day 1 close greater than the high of Day 0?
                    if df.iloc[i + 1]['Close'] > high_day_0:
                        #logger.info(f'Step 11B. df.iloc[i + 1][Close] > high_day_0')
                        change_confirmations += 1

                    # If trading volume on day 1 > than on Day 0?
                    if df.iloc[i + 1]['Volume'] > average_volume*1.2:
                        #logger.info(f'Step 11C. df.iloc[i + 1][Volume] > average_volume*1.2')
                        change_confirmations += 1

                    # Is there a bearish or reversal candle on Day 1?
                    if df.iloc[i + 1]['bullish'] > 0 or df.iloc[i + 1]['bullish_reversal'] > 0 or df.iloc[i + 1]['reversal'] > 0:
                        #logger.info(f'Step 11D. df.iloc[i + 1][bullish] > 0 or df.iloc[i + 1][bullish_reversal] > 0 or df.iloc[i + 1][reversal] > 0')
                        change_confirmations += 1
                    #logger.info(f'change_confirmations = {change_confirmations}')
                    if change_confirmations >= 2:
                        #logger.info(f'Step 12. change_confirmations >= 2')
                        # At least 2 signals confirm the swing point.
                        # Down trend ended on Day 0.
                        if low_day_0 > last_low_reference:
                            # This low is higher than the previous high so this is a HL.
                            #logger.info(f'Step 12A. low_day_0 > last_low_reference. SP = HL')
                            df.at[index_label, 'swing_point_label'] = "HL"
                            if last_swing_point == 'HH':
                                current_trend_seq_count += 1
                            if last_swing_point == 'LH':
                                current_trend_seq_count = 1
                            last_swing_point = 'HL'
                        else:
                            #logger.info(f'Step 12B. low_day_0 <= last_low_reference. SP = LL')
                            df.at[index_label, 'swing_point_label'] = "LL"
                            if last_swing_point == 'HH':
                                current_trend_seq_count = 1
                            if last_swing_point == 'LH':
                                current_trend_seq_count += 1
                            last_swing_point = 'LL'
                        uptrend_in_progress = True  # Now trend is upward.
                        df.at[index_label, 'swing_point_price'] = low_day_0
                        last_low_reference = low_day_0
                        df.at[index_label, 'magnitude'] = 1
                        swing_point_occured = True
            if swing_point_occured == True:
                #logger.info(f'Step 13.')
                df.at[index_label, 'healthy_bullish_candle'] = healthy_bullish_count
                df.at[index_label, 'healthy_bearish_candle'] = healthy_bearish_count
                df.at[index_label, 'candle_count_since_last_swing_point'] = candle_count_since_last_swing_point
                candle_count_since_last_swing_point = 0
                final_swing_point_trend = df.loc[index_label, 'swing_point_current_trend']
                #logger.info(f'Step 14.')
                if current_trend_seq_count > 2:
                    # This data point is part of a swing trend.
                    #logger.info(f'Step 15.')
                    if last_swing_point[0] == "H":
                        df.at[index_label, 'swing_point_current_trend'] = 1
                        # print('Setting uptrend.')
                    elif last_swing_point[0] == "L":
                        df.at[index_label, 'swing_point_current_trend'] = -1
                        # print('Setting downtrend.')
            else:
                # Not a swing point, so reset the healthy candle count to 0 for this data point
                #logger.info(f'Step 16.')
                df.at[index_label, 'healthy_bullish_candle'] = 0
                df.at[index_label, 'healthy_bearish_candle'] = 0

        return df, final_swing_point_trend
    except Exception as e:
        print(f"Error in identify_highs_lows_gann2() for {ticker.symbol}: {e}")


def add_levels_to_price_history(df, sr_levels, retests):
    # Initialize new columns with default values
    df['level'] = None
    df['level_type'] = 0
    df['level_strength'] = 0
    # print('Adding levels to price_history...')

    # Update 'level', 'level_type', and 'level_strength' based on the retests dictionary and the support levels
    for level, level_datetime in sr_levels.items():
        # print('Found support level:', level, level_datetime)
        df.at[level_datetime, 'level'] = level
        df.at[level_datetime, 'level_type'] = 1
        if level in retests:
            # print('Found retest of support level.')
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

def find_higher_order_swing_points(ticker, price_history, logger):
    # price_history is a pandas dataframe
    magnitude_to_test = 2
    # Retrieve the price candles for swing points at the priod magnitude level.
    swing_points = price_history['magnitude']
    # Loop through each swing point
        # If the swing point label is the first LL, prior swingpoint was a H
        # If the swing point label is the first HH, prior swingpoint was a L


def download_daily_ticker_price(timeframe='Ad hoc', ticker_symbol="All", trigger='Cron'):
    delete_old_prices = False
    try:
        if trigger == 'Cron':
            logger = scheduled_logger
        else:
            logger = default_logger
        display_local_time()
        print('Timeframe:', timeframe)
        print('ticker_symbol:', ticker_symbol)
        logger.info(f'Running download_prices() for ticker_symbol: {str(ticker_symbol)}')

        # ticker_count = Ticker.objects.all().count()
        # Check if the 'TSE stocks' category exists
        tse_stocks_category = TickerCategory.objects.filter(name='TSE stocks').first()
        if not tse_stocks_category:
            # 'TSE stocks' category doesn't exist, handle it as you wish (e.g., raise an exception or return an error response)
            print('TSE stocks category does not exist!!')

        ticker = Ticker.objects.get(symbol=ticker_symbol)
        new_record_count = 0
        if ticker is None:
            print('No Ticker instance found for this symbol')
            logger.error(f'No Ticker instance found for this symbol.')
        else:
            if timeframe == 'Daily':  # and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol):
                print('start_time:')
                start_time = display_local_time()  # record the start time of the loop
                #print('Downloading daily prices...')
                logger.info(f'[PRICE DOWNLOAD] Downloading daily prices...')

                # Query and delete the old prices
                start_day = timezone.now() - timedelta(days=365)
                finish_day = timezone.now()
                if delete_old_prices:
                    old_prices = DailyPrice.objects.filter(datetime__lt=start_day, ticker=ticker)
                    deleted_count, _ = old_prices.delete()
                    print('Deleted', deleted_count, 'older DailyPrice records.')
                    logger.info(f'Deleted {str(deleted_count)} older DailyPrice records.')

                # Check if the stock is marked as a TSE stock.
                is_in_tse_stocks = ticker.categories.filter(pk=tse_stocks_category.pk).exists()
                if is_in_tse_stocks:
                    hour_offset = 15
                else:
                    hour_offset = 4
                logger.info(f'hour_offset = {str(hour_offset)}')
                logger.info(f'Retrieving data from {str(start_day)} to {str(finish_day)}...')

                try:
                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, '1D', start_day, finish_day, logger)
                    if len(price_history) >= 3:
                        # print('Step 11')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(price_history)}')
                        #logger.info(f'About to call add_candle_data()...')
                        price_history = add_candle_data(price_history, candlestick_functions, column_names)
                        # print('Step 12')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(price_history)}')
                        #logger.info(f'About to call add_db_candle_data()...')
                        price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)
                        # print('Step 13')
                        #price_history['Datetime_TZ'] = price_history['Datetime_TZ'].dt.tz_localize('UTC')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(price_history)}')
                        #logger.info(f'About to call count_patterns()...')
                        count_patterns(price_history, pattern_types)
                        # print('Step 14')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(price_history)}')
                        #logger.info(f'About to call find_levels()...')
                        sr_levels, retests, last_high_low_level = find_levels(price_history, window=20)
                        # print('price_history.tail(3) before identify_highs_lows():',price_history.tail(3))
                        # if ticker.symbol == 'NNOX':
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(combined_data): {get_largest_index_value(price_history)}')
                        #logger.info(f'About to call identify_highs_lows_gann2()...')
                        price_history, swing_point_current_trend = identify_highs_lows_gann2(ticker,
                                                                                                                 price_history, logger,
                                                                                                                 reversal_days=2,
                                                                                                                 price_move_percent=1.5)
                        # else:
                        #    price_history, swing_point_current_trend = identify_highs_lows2(price_history, window=3, price_move_percent=1.5)

                        # print('price_history.tail(30) after identify_highs_lows():', price_history.tail(30))
                        # print('Step 15')
                        ticker.last_high_low = last_high_low_level
                        ticker.swing_point_current_trend = swing_point_current_trend
                        # print('Step 16')
                        ticker.save()
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(price_history): {get_largest_index_value(price_history)}')
                        print('ticker.swing_point_current_trend:', ticker.swing_point_current_trend)
                        #logger.info(f'About to call add_levels_to_price_history()...')
                        price_history = add_levels_to_price_history(price_history, sr_levels, retests)
                        # print('Step 17')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'About to call add_ema_and_trend()...')
                        price_history = add_ema_and_trend(price_history)
                        # print('Step 18')
                        #logger.info(
                        #    f'price_history.columns: {price_history.columns}')
                        #logger.info(f'get_largest_index_value(price_history): {get_largest_index_value(price_history)}')
                        #logger.info(
                        #    f'[A] About to check for NaT in Datetime and Datetime_TZ for ticker {str(ticker.symbol)}...')
                        #check_for_nat(logger, price_history)
                        step = 1
                        # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                        for index, row in price_history.iterrows():
                            if math.isnan(row['Close']):
                                step = 2
                                print(row)
                            if not DailyPrice.objects.filter(ticker=ticker, datetime=row['Datetime_TZ']).exists():
                                step = 3
                                new_record_count += 1
                                logger.info(f'About to create new record for datetime {str(row["Datetime_TZ"])}...')
                                daily_price = DailyPrice(
                                    ticker=ticker,
                                    datetime=row['Datetime_TZ'],
                                    datetime_tz=row['Datetime_TZ'],
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
                                    level=row['level'],
                                    level_type=row['level_type'],
                                    level_strength=row['level_strength'],
                                    ema_200=row['EMA_200'],
                                    ema_50=row['EMA_50'],
                                    trend=row['Trend'],
                                    swing_point_label=row['swing_point_label'],
                                    swing_point_current_trend=row['swing_point_current_trend'],
                                    healthy_bullish_count=row['healthy_bullish_candle'],
                                    healthy_bearish_count=row['healthy_bearish_candle'],
                                    candle_count_since_last_swing_point=row['candle_count_since_last_swing_point'],
                                )
                                #logger.info(f'Defined new daily_price instance. datetime_tz: {str(row["Datetime_TZ"])}')
                            else:
                                #logger.info(f'About to update existing record for datetime {str(row["Datetime_TZ"])}...')
                                step = 4
                                daily_price = DailyPrice.objects.get(ticker=ticker, datetime=row['Datetime_TZ'])
                                daily_price.datetime_tz = daily_price.datetime
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
                                daily_price.swing_point_label = row['swing_point_label']
                                daily_price.swing_point_current_trend = row['swing_point_current_trend']
                                daily_price.healthy_bullish_count = row['healthy_bullish_candle']
                                daily_price.healthy_bearish_count = row['healthy_bearish_candle']
                                daily_price.candle_count_since_last_swing_point = row['candle_count_since_last_swing_point']
                            daily_price.save()

                            # Get the ContentType for the DailyPrice model
                            step = 5
                            content_type = ContentType.objects.get_for_model(daily_price)

                            if len(row['swing_point_label']) > 0:
                                # This was noted to be a swing point
                                step = 6

                                # First check if a swing point instance has already been created for this swing point.
                                #logger.info(
                                #    f"About to check for existing SwingPoint instance. ticker:{str(ticker)}, row['Datetime_TZ']: {str(row['Datetime_TZ'])}., "
                                #    f"content_type: {str(content_type)}")
                                #logger.info(f'This record is labelled as a swingpoint.')
                                existing_swing_point_instance = SwingPoint.objects.filter(ticker=ticker,date=row["Datetime_TZ"],magnitude=row["magnitude"],content_type=content_type)
                                step = 7
                                #logger.info(f'existing_swing_point_instance: {str(existing_swing_point_instance)}.')
                                if not existing_swing_point_instance.exists():
                                    step = 8
                                    #logger.info(f'Creating a swingpoint for datetime {str(row["Datetime_TZ"])} because does not exist.')
                                    #logger.info(f'SwingPoint does not exist.')
                                    new_swing_point = SwingPoint.objects.create(
                                        ticker=ticker,
                                        date=row['Datetime_TZ'],
                                        price=row['swing_point_price'],
                                        label=row['swing_point_label'],
                                        candle_count_since_last_swing_point=row['candle_count_since_last_swing_point'],
                                        content_type=content_type,
                                        object_id=daily_price.id,
                                        magnitude=row['magnitude'],
                                    )
                                    step = 9
                                else:
                                    existing_swing_point_instance.magnitude = row['magnitude']
                                    existing_swing_point_instance.save()
                            else:
                                # Should not find any swing points connected to this candle. If exist, should be deleted.
                                step = 10
                                existing_swing_point_instance = SwingPoint.objects.filter(ticker=ticker,
                                                                                          date=row['Datetime_TZ'],
                                                                                          content_type=content_type)
                                step = 11
                                if existing_swing_point_instance:
                                    step = 12
                                    logger.info(f'For {row["Datetime_TZ"]}, existing_swing_point_instance exists: {str(existing_swing_point_instance)}. Deleting...')
                                    existing_swing_point_instance.delete()
                        step = 13
                        #logger.info(
                        #    f'[B] About to check for NaT in Datetime and Datetime_TZ for ticker {str(ticker.symbol)}...')
                        #nat_indexes = check_for_nat(logger, price_history)
                        step = 14

                        # Now retrieve any planned trades for this ticker and amend the price to match the latest close price.
                        # Find the most recent DailyPrice instance for this ticker
                        #logger.info(f'About to find most_recent_daily_price...')
                        most_recent_daily_price = DailyPrice.objects.filter(ticker=ticker).order_by('-datetime').first()
                        step = 15
                        #logger.info(f'trades_to_update...')
                        trades_to_update = Trade.objects.filter(tradingopp__ticker=ticker,action='1',  # Where the action is 'Buy'
                            status='0'  # And the status is 'Planned'
                        )
                        step = 16

                        # Update the price of each trade to the close_price of the most recent DailyPrice
                        for trade in trades_to_update:
                            trade.price = most_recent_daily_price.close_price
                            trade.save()
                        step = 17
                    else:
                        print('Insufficient data.')
                except Exception as e:
                    logger.error(f'index: {index}. step = {step}. Error in download_daily_ticker_price: {e}.')
                print('new_record_count:', new_record_count)
                logger.info(f'Saved {str(new_record_count)} new DailyPrice records for this ticker.')
                end_time = display_local_time()  # record the end time of the loop
    except Exception as e:
        message = f'Error occured in download_daily_ticker_price(): {e}'
        nj_param = Params.objects.get(key='night_job_end_dt')
        end_time = datetime.now()
        nj_param.value = end_time
        nj_param.save()
        nj_param = Params.objects.get(key='night_job_status_message')
        nj_param.value = message
        nj_param.save()
        logger.error(message)

def download_prices(timeframe='Ad hoc', ticker_symbol="All", trigger='Cron'):
    # timeframe = Timeframe for which to download prices.
        # valid values = "Daily", "15 mins", "5 mins"
    # ticker_symbol = which ticker to download
        # valid values = 'AAPL'
        # "All" means all the tickers that are in the database.
    # trigger = what started this call.
        # trigger = "Cron" means a cron job.
        # trigger = "User" means that the user has triggered the price download.

    print('Running download_prices()...')
    try:
        local_time = display_local_time()
        print('Timeframe:', timeframe)
        print('ticker_symbol:', ticker_symbol)
        logger.info(f'Running download_prices() for ticker_symbol: {str(ticker_symbol)}')

        #ticker_count = Ticker.objects.all().count()
        # Check if the 'TSE stocks' category exists
        tse_stocks_category = TickerCategory.objects.filter(name='TSE stocks').first()
        if not tse_stocks_category:
            # 'TSE stocks' category doesn't exist, handle it as you wish (e.g., raise an exception or return an error response)
            print('TSE stocks category does not exist!!')

        #if ticker_symbol == 'All':
        #    logger.info(f'All tickers requested. ticker_count: {str(ticker_count)}')
        #    ticker_group = Ticker.objects.all().order_by('symbol')
        #else:
        #    ticker_group = Ticker.objects.filter(symbol=ticker_symbol)

        ticker = Ticker.objects.get(symbol=ticker_symbol)
        if ticker is None:
            print('No Ticker instance found for this symbol')
            logger.info(f'No Ticker instance found for this symbol.')
        else:
            new_record_count=0
            if timeframe == 'Daily': # and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol):
                print('start_time:')
                start_time = display_local_time()  # record the start time of the loop
                print('Downloading daily prices...')
                start_day = timezone.now() - timedelta(days=365)
                finish_day = timezone.now()
                interval = '1D'

                # Query and delete the old prices
                #old_prices = DailyPrice.objects.filter(datetime__lt=start_day, ticker=ticker)
                #deleted_count, _ = old_prices.delete()
                #print('Deleted', deleted_count, 'older DailyPrice records.')
                #logger.info(f'Deleted {str(deleted_count)} older DailyPrice records.')

                # Check if the stock is marked as a TSE stock.
                is_in_tse_stocks = ticker.categories.filter(pk=tse_stocks_category.pk).exists()
                if is_in_tse_stocks:
                    hour_offset = 15
                else:
                    hour_offset = 4
                logger.info(f'hour_offset = {str(hour_offset)}')

                # Get the max date for DailyPrice
                latest_daily_price = DailyPrice.objects.filter(ticker=ticker).latest('datetime')
                last_update_time = latest_daily_price.datetime

                # Get the list of missing dates
                #missing_dates = get_missing_dates(ticker, interval, start_day, finish_day, hour_offset)

                #logger.info(f'missing_dates = {str(missing_dates)}')
                #if missing_dates:

                # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                #start_day = min(missing_dates)
                start_day = last_update_time

                #finish_day = max(missing_dates)
                finish_day = timezone.now()
                print('Retrieving data from ', start_day, ' to ', finish_day)
                logger.info(f'Retrieving data from {str(start_day)} to {str(finish_day)}...')

                # Request price data for the entire missing date range
                price_history = get_price_data(ticker, interval, start_day, finish_day, logger)
                if len(price_history) >= 3:
                    #print('Step 11')
                    price_history = add_candle_data(price_history, candlestick_functions, column_names)
                    #print('Step 12')
                    price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)
                    #print('Step 13')
                    count_patterns(price_history, pattern_types)
                    #print('Step 14')
                    sr_levels, retests, last_high_low_level = find_levels(price_history, window=20)
                    #print('price_history.tail(3) before identify_highs_lows():',price_history.tail(3))
                    #if ticker.symbol == 'NNOX':
                    price_history, swing_point_current_trend = identify_highs_lows_gann(ticker, price_history, logger, reversal_days=2, price_move_percent=1.5)
                    #else:
                    #    price_history, swing_point_current_trend = identify_highs_lows2(price_history, window=3, price_move_percent=1.5)

                    #print('price_history.tail(30) after identify_highs_lows():', price_history.tail(30))
                    #print('Step 15')
                    ticker.last_high_low = last_high_low_level
                    ticker.swing_point_current_trend = swing_point_current_trend
                    #print('Step 16')
                    ticker.save()
                    print('ticker.swing_point_current_trend:', ticker.swing_point_current_trend)
                    price_history = add_levels_to_price_history(price_history, sr_levels, retests)
                    #print('Step 17')
                    price_history = add_ema_and_trend(price_history)
                    #print('Step 18')
                    # Call the function and pass the 'price_history' DataFrame
                    logger.info(f'[A] About to check for NaT in Datetime and Datetime_TZ for ticker {str(ticker.symbol)}...')
                    nat_indexes = check_for_nat(logger, price_history)
                    # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                    for index, row in price_history.iterrows():
                        if math.isnan(row['Close']):
                            print(row)
                        if not DailyPrice.objects.filter(ticker=ticker, datetime=row['Datetime_TZ']).exists():
                            new_record_count += 1
                            daily_price = DailyPrice(
                                ticker=ticker,
                                datetime=row['Datetime_TZ'],
                                datetime_tz=row['Datetime_TZ'],
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
                                level=row['level'],
                                level_type=row['level_type'],
                                level_strength=row['level_strength'],
                                ema_200=row['EMA_200'],
                                ema_50=row['EMA_50'],
                                trend=row['Trend'],
                                swing_point_label=row['swing_point_label'],
                                swing_point_current_trend=row['swing_point_current_trend'],
                                healthy_bullish_count = row['healthy_bullish_candle'],
                                healthy_bearish_count=row['healthy_bearish_candle'],
                                candle_count_since_last_swing_point=row['candle_count_since_last_swing_point'],
                            )
                            logger.info(f'Defined new daily_price instance. datetime_tz: {str(row["Datetime_TZ"])}')
                        else:
                            daily_price = DailyPrice.objects.get(ticker=ticker, datetime=row['Datetime_TZ'])
                            daily_price.datetime_tz = daily_price.datetime
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
                            daily_price.swing_point_label = row['swing_point_label']
                            daily_price.swing_point_current_trend = row['swing_point_current_trend']
                            daily_price.healthy_bullish_count = row['healthy_bullish_candle']
                            daily_price.healthy_bearish_count = row['healthy_bearish_candle']
                            daily_price.candle_count_since_last_swing_point = row['candle_count_since_last_swing_point']
                        daily_price.save()
                        if len(row['swing_point_label']) > 0:
                            # This was noted to be a swing point
                            # Get the ContentType for the DailyPrice model
                            content_type = ContentType.objects.get_for_model(daily_price)

                            # First check if a swing point instance has already been created for this swing point.
                            #logger.info(f"About to check for existing SwingPoint instance. ticker:{str(ticker)}, row['Datetime_TZ']: {str(row['Datetime_TZ'])}., "
                            #             f"content_type: {str(content_type)}")
                            existing_swing_point_instance = SwingPoint.objects.filter(ticker=ticker, date=row['Datetime_TZ'], magnitude=row['magnitude'], content_type=content_type)
                            #logger.info(f'existing_swing_point_instance: {str(existing_swing_point_instance)}.')
                            if not existing_swing_point_instance.exists():
                                #logger.info(f'SwingPoint does not exist.')
                                new_swing_point = SwingPoint.objects.create(
                                    ticker=ticker,
                                    date=row['Datetime_TZ'],
                                    price=row['swing_point_price'],
                                    label=row['swing_point_label'],
                                    candle_count_since_last_swing_point=row['candle_count_since_last_swing_point'],
                                    content_type=content_type,
                                    object_id=daily_price.id,
                                    magnitude=row['magnitude'],
                                )
                            else:
                                #logger.info(f'SwingPoint does exist.')
                                pass

                else:
                    print('Insufficient data.')
                logger.info(f'[B] About to check for NaT in Datetime and Datetime_TZ for ticker {str(ticker.symbol)}...')
                nat_indexes = check_for_nat(logger, price_history)

                print('new_record_count:',new_record_count)
                logger.info(f'Saved {str(new_record_count)} new DailyPrice records for this ticker.')
                end_time = display_local_time()  # record the end time of the loop
                elapsed_time = end_time - start_time  # calculate elapsed time

                # Ensure at least 20 seconds before the next iteration
                #if elapsed_time.total_seconds() < 20 and ticker_symbol == "All" and ticker_count > 195:
                #    pause_duration = 20 - elapsed_time.total_seconds()
                #    print('Rate throttling for',pause_duration,'secs...')
                #    logger.info(f'Rate throttling for {str(pause_duration)} secs...')
                #    sleep(pause_duration)
            if timeframe == '15 mins' and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol) and ((local_time.hour > 5 and local_time.hour < 15) or trigger=='User'):
                start_day = timezone.now() - timedelta(days=7)
                finish_day = timezone.now()
                interval = '15m'
                print('Checking 15 min data for', ticker.symbol)

                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day, 23, logger)

                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day, ' to ', finish_day)

                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day, logger)
                    print('len(price_history):', len(price_history))
                    if len(price_history) >= 3:
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
                                fifteenmin_price = FifteenMinPrice.objects.get(ticker=ticker, datetime=row['Datetime'])
                                fifteenmin_price.patterns_detected = row['patterns_detected']
                                fifteenmin_price.bullish_detected = row['bullish']
                                fifteenmin_price.bearish_detected = row['bearish']
                                fifteenmin_price.reversal_detected = row['reversal']
                                fifteenmin_price.bullish_reversal_detected = row['bullish_reversal']
                                fifteenmin_price.bearish_reversal_detected = row['bearish_reversal']
                            fifteenmin_price.save()
                    else:
                        print('Insufficient data.')
            if timeframe == '5 mins' and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol) and ((local_time.hour > 5 and local_time.hour < 15) or trigger=='User'):
                start_day = timezone.now() - timedelta(days=5)
                finish_day = timezone.now()
                interval = '5m'
                print('Checking 5 min data for', ticker.symbol)
                # Get the list of missing dates
                missing_dates = get_missing_dates(ticker, interval, start_day, finish_day,23, logger)
                if missing_dates:
                    # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                    start_day = min(missing_dates)
                    finish_day = max(missing_dates)
                    print('Retrieving data from ', start_day, ' to ', finish_day)
                    # Request price data for the entire missing date range
                    price_history = get_price_data(ticker, interval, start_day, finish_day,logger)
                    print('len(price_history):', len(price_history))
                    if len(price_history) >= 3:
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
                    else:
                        print('Insufficient data.')
    except Exception as e:
        print(f"Error in download_prices(): {e}")
