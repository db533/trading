import logging

logging.basicConfig(level=logging.INFO)
import yfinance as yf
from time import sleep
from .models import Ticker, DailyPrice, FifteenMinPrice, FiveMinPrice
import pandas as pd
import pytz
from datetime import datetime, timedelta, timezone, date, time
from candlestick import candlestick
from . import db_candlestick
from django.utils import timezone


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

def get_price_data(ticker, interval, start_time, finish_time):
    # Fetching existing data from the database
    try:
        existing_data = DailyPrice.objects.filter(ticker=ticker).values()
        if len(existing_data) > 0:
            existing_data_retrieved = True
        else:
            existing_data_retrieved = False
        if existing_data_retrieved == True:
            existing_df = pd.DataFrame.from_records(existing_data)
            existing_df['Datetime'] = pd.to_datetime(existing_df['datetime'])
            existing_df['Open'] = existing_df['open_price'].astype(float)
            existing_df['High'] = existing_df['high_price'].astype(float)
            existing_df['Low'] = existing_df['low_price'].astype(float)
            existing_df['Close'] = existing_df['close_price'].astype(float)
            existing_df['Volume'] = existing_df['volume']
            existing_df = existing_df.drop(columns=['datetime', 'ticker_id', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
    except Exception as e:
        print(f"Error fetching existing data for {ticker.symbol}: {e}")
        existing_df = pd.DataFrame(columns=['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])


    try:
        # Ensure start_time and finish_time are timezone-aware
        start_time = start_time.replace(tzinfo=timezone.utc)
        finish_time = finish_time.replace(tzinfo=timezone.utc)

        data = yf.Ticker(ticker.symbol).history(interval=interval, start=start_time, end=finish_time)
        if not data.empty:
            print('Retrieve new price data records...')

            # Create a 'Datetime' column from the index
            data['Datetime'] = data.index
            #data = data.tz_localize(None)

            if existing_data_retrieved == True:
                # Concatenating the new and existing data, while ensuring no duplicate entries
                combined_data = pd.concat([data, existing_df], ignore_index=True)
                #print('combined_data.columns:',combined_data.columns)
                #print(combined_data.tail(5))
            else:
                combined_data = data
            #combined_data = combined_data.set_index('Datetime')
            combined_data.index = pd.to_datetime(combined_data.index)
            combined_data.index = combined_data.index.tz_localize(None)

            # To create an index that is timezone-aware, uncomment next line:
            #combined_data.index = pd.to_datetime(combined_data.index, utc=True)
            # To make the index timezone-naive, use the following line
            #combined_data.index = combined_data.index.tz_localize(None)

            combined_data = combined_data.loc[~combined_data.index.duplicated(keep='last')]

            combined_data = combined_data.sort_values(by='Datetime', ascending=True)

            combined_data['Ticker'] = ticker.symbol  # Add 'Ticker' column with the symbol
            combined_data['PercentChange'] = combined_data['Close'].pct_change() * 100  # Multiply by 100 to get a percentage
            #print('combined_data.index[0]:',combined_data.index[0])
            #print("combined_data.at[combined_data.index[0], 'PercentChange']:", combined_data.at[combined_data.index[0], 'PercentChange'])
            combined_data.at[combined_data.index[0], 'PercentChange'] = 0

            # Duplicating the Datetime index into a new column
            combined_data['Datetime'] = combined_data.index.copy()
            combined_data['Datetime'] = pd.to_datetime(combined_data['Datetime'])
            combined_data['Datetime'] = combined_data['Datetime'].dt.tz_localize('UTC')
            combined_data = combined_data.dropna(subset=['Open'])

            # Reorder the columns
            combined_data = combined_data[['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'PercentChange']]
            #print(combined_data)
            #print(combined_data.tail(5))
    except Exception as e:
        print(f"Error downloading data for {ticker.symbol}: {e}")
        combined_data = pd.DataFrame(columns=['Datetime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])

    return combined_data


def get_missing_dates(ticker, interval, start_day, finish_day):
    # Get the list of dates missing in DailyPrice for the given ticker within the date range
    #print('start_day:',start_day,'timezone.make_naive(start_day):',timezone.make_naive(start_day))
    #print('finish_day:', finish_day, 'timezone.make_naive(finish_day):', timezone.make_naive(finish_day))
    if interval == '1D':
        existing_dates = DailyPrice.objects.filter(
            ticker=ticker, datetime__range=(start_day, finish_day)
        ).values_list('datetime', flat=True)

        # Ensure existing_dates is a list of timezone-naive or aware datetime objects
        existing_dates = [timezone.make_naive(date) for date in existing_dates]
        #print('existing_dates:', existing_dates[:3])

        # Ensure that start_day and finish_day have the time set to 4:00
        start_day_with_time = datetime.combine(start_day, time(4, 0))
        finish_day_with_time = datetime.combine(finish_day, time(4, 0))

        all_dates = pd.date_range(start=start_day_with_time, end=finish_day_with_time, freq='D')
        all_dates = all_dates.tz_localize(None)

        # Ensure all_dates is a list of native python datetime objects
        all_dates = [date.to_pydatetime() for date in all_dates]

        #print('all_dates:', all_dates[:3])

        missing_dates = [date for date in all_dates if date not in existing_dates]

        #print('missing_dates:', missing_dates[:3])
        #print('min(missing_dates):',min(missing_dates))
    if interval == '15m':
        existing_dates = FifteenMinPrice.objects.filter(ticker=ticker,
                                                        datetime__range=(start_day, finish_day)).values_list(
            'datetime', flat=True)
        all_dates = pd.date_range(start=start_day, end=finish_day, freq='15T')
        all_dates.tz_localize(None)
        missing_dates = [date for date in all_dates if date not in existing_dates]
        # print('all_dates:',all_dates)
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
db_candlestick_functions = [db_candlestick.three_white_soldiers,
                            ]

# List of column names to replace 'xxx' in price_history
db_column_names = ['three_white_soldiers',
                   ]

pattern_types = {
    'bullish': ['bullish_engulfing', 'bullish_harami', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
                'three_white_soldiers'],
    'bearish': ['bearish_engulfing', 'bearish_harami', 'dark_cloud_cover', 'gravestone_doji'],
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
    local_time = display_local_time()
    print('Timeframe:', timeframe)
    print('ticker_symbol:', ticker_symbol)

    ticker_count = Ticker.objects.all().count()

    for ticker in Ticker.objects.all().order_by('symbol'):
        if ticker_symbol == 'All':
            print('Ticker:', ticker.symbol)

        new_record_count=0
        if timeframe == 'Daily' and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol):
            print('start_time:')
            start_time = display_local_time()  # record the start time of the loop
            print('Downloading daily prices...')
            start_day = timezone.now() - timedelta(days=365)
            finish_day = timezone.now()
            interval = '1D'

            # Get the list of missing dates
            missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)

            if missing_dates:
                # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                start_day = min(missing_dates)
                finish_day = max(missing_dates)
                print('Retrieving data from ', start_day, ' to ', finish_day)

                # Request price data for the entire missing date range
                price_history = get_price_data(ticker, interval, start_day, finish_day)
                if len(price_history) >= 3:
                    #print('Step 1')
                    price_history = add_candle_data(price_history, candlestick_functions, column_names)
                    #print('Step 2')
                    price_history = add_db_candle_data(price_history, db_candlestick_functions, db_column_names)
                    #print('Step 3')
                    count_patterns(price_history, pattern_types)
                    #print('Step 4')
                    sr_levels, retests, last_high_low_level = find_levels(price_history, window=20)
                    #print('Step 5')
                    ticker.last_high_low = last_high_low_level
                    #print('Step 6')
                    ticker.save()
                    price_history = add_levels_to_price_history(price_history, sr_levels, retests)
                    price_history = add_ema_and_trend(price_history)

                    # Save price_history data to the DailyPrice model only if the 'Datetime' value doesn't exist
                    for index, row in price_history.iterrows():
                        if not DailyPrice.objects.filter(ticker=ticker, datetime=row['Datetime']).exists():
                            new_record_count += 1
                            daily_price = DailyPrice(
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
                else:
                    print('Insufficient data.')
            print('new_record_count:',new_record_count)
            end_time = display_local_time()  # record the end time of the loop
            elapsed_time = end_time - start_time  # calculate elapsed time

            # Ensure at least 20 seconds before the next iteration
            if elapsed_time.total_seconds() < 20 and ticker_symbol == "All" and ticker_count > 195:
                print('Rate throttling for',20 - elapsed_time.total_seconds(),'secs...')
                sleep(20 - elapsed_time.total_seconds())
        if timeframe == '15 mins' and (ticker_symbol == 'All' or ticker_symbol == ticker.symbol) and ((local_time.hour > 5 and local_time.hour < 15) or trigger=='User'):
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
            missing_dates = get_missing_dates(ticker, interval, start_day, finish_day)
            if missing_dates:
                # Set start_day to the smallest date and finish_day to the largest date in missing_dates
                start_day = min(missing_dates)
                finish_day = max(missing_dates)
                print('Retrieving data from ', start_day, ' to ', finish_day)
                # Request price data for the entire missing date range
                price_history = get_price_data(ticker, interval, start_day, finish_day)
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
