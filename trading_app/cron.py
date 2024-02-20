# trading_app/cron.py

from django_cron import CronJobBase, Schedule
from .update_ticker_metrics import update_ticker_metrics
from .price_download import download_prices, category_price_download
from .update_strategies import process_trading_opportunities
from .test_cron_job import test_cron_job
from datetime import datetime, timedelta, timezone, date, time
import pytz
import logging
import time

logger = logging.getLogger('django')

# RUN_AT_TIMES actually execute 3 hours later than defined here.

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

###############################
# Inactive cron jobs:
class DailyPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['00:01']  # Run at 1:00 AM local time
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    # schedule = Schedule(run_every_mins=1)  # Run once a day
    code = 'trading_app.daily_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='Daily')

class TestPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['00:01']  # Run at 23:00 local time
    #schedule = Schedule(run_at_times=RUN_AT_TIMES)
    schedule = Schedule(run_every_mins=3)  # Run once a day
    code = 'trading_app.test_cron_job'

    def do(self):
        category_price_download('Test tickers')
        process_trading_opportunities()

class FifteenMinsPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['{:02d}:{:02d}'.format(hour, minute) for hour in range(5, 14) for minute in [1, 16, 31, 46]]
    #RUN_AT_TIMES = ['07:01', '07:16', '07:31', '07:46', '08:01', '08:16', '08:31', '08:46', '09:01', '09:16', '09:31', '09:46', '10:01', '10:16', '10:31', '10:46', '11:01', '11:16', '11:31', '11:46', '12:01', '12:16', '12:31', '12:46', '13:01', '13:16', '13:31', '13:46', '14:01', '14:16', '14:31', '14:46', '15:01', '15:16', '15:31', '15:46', '16:01', '16:16', '16:31', '16:46', '17:01', '17:16', '17:31', '17:46', '18:01', '18:16', '18:31', '18:46', '19:01', '19:16', '19:31', '19:46', '20:01', '20:16', '20:31', '20:46']
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    #schedule = Schedule(run_every_mins=15)  # Run once a day
    code = 'trading_app.15_min_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='15 mins')

class FiveMinsPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['{:02d}:{:02d}'.format(hour, minute) for hour in range(5, 14) for minute in [6, 11, 21, 26, 36, 41, 51, 56]]
    #RUN_AT_TIMES = ['07:06', '07:11', '07:21', '07:26', '07:36', '07:41', '07:51', '07:56', '08:06', '08:11', '08:21', '08:26', '08:36', '08:41', '08:51', '08:56', '09:06', '09:11', '09:21', '09:26', '09:36', '09:41', '09:51', '09:56', '10:06', '10:11', '10:21', '10:26', '10:36', '10:41', '10:51', '10:56', '11:06', '11:11', '11:21', '11:26', '11:36', '11:41', '11:51', '11:56', '12:06', '12:11', '12:21', '12:26', '12:36', '12:41', '12:51', '12:56', '13:06', '13:11', '13:21', '13:26', '13:36', '13:41', '13:51', '13:56', '14:06', '14:11', '14:21', '14:26', '14:36', '14:41', '14:51', '14:56', '15:06', '15:11', '15:21', '15:26', '15:36', '15:41', '15:51', '15:56', '16:06', '16:11', '16:21', '16:26', '16:36', '16:41', '16:51', '16:56', '17:06', '17:11', '17:21', '17:26', '17:36', '17:41', '17:51', '17:56', '18:06', '18:11', '18:21', '18:26', '18:36', '18:41', '18:51', '18:56', '19:06', '19:11', '19:21', '19:26', '19:36', '19:41', '19:51', '19:56', '20:06', '20:11', '20:21', '20:26', '20:36', '20:41', '20:51', '20:56']
    #schedule = Schedule(run_every_mins=5)  # Run once a day
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    code = 'trading_app.5_min_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='5 mins')

class TestCronJob(CronJobBase):
    schedule = Schedule(run_every_mins=1000000)  # Run once a day
    code = 'trading_app.test_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        #test_cron_job()
        pass

# Create new TradingOpps
class DailyTradingOppCreationCronJob(CronJobBase):
    schedule = Schedule(run_every_mins=1)  # Run once a day
    RUN_AT_TIMES = ['06:00']  # Run at 1:00 AM local time
    #schedule = Schedule(run_at_times=RUN_AT_TIMES)
    code = 'trading_app.trading_opp_creation_cron_job'
    def do(self):
        # Run the update_ticker_metrics function
        start_time = time.time()  # Capture start time
        process_trading_opportunities()
        end_time = time.time()  # Capture end time
        elapsed_time_str = format_elapsed_time(start_time, end_time)

        # Log or print the elapsed time. Here's an example of logging it:
        logger.info(f'DailyTradingOppCreationCronJob completed in {elapsed_time_str}')

###############################
# ACTIVE cron jobs:
# RUN_AT_TIMES actually execute 3 hours later than defined here.

# Download TSE prices. 576 prices.
# Trust start time: 00:03
# Run time 3:12
# Should complete by 03:15
class DailyTSEPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['21:00']
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    #schedule = Schedule(run_every_mins=3)  # Run once a day
    code = 'trading_app.daily_tse_price_download_cron_job'

    def do(self):
        start_time = time.time()  # Capture start time

        display_local_time()
        category_price_download('TSE stocks')
        display_local_time()

        end_time = time.time()  # Capture end time
        elapsed_time = end_time - start_time  # Compute elapsed time
        elapsed_time_str = format_elapsed_time(start_time, end_time)

        # Log or print the elapsed time. Here's an example of logging it:
        logger.info(f'DailyTSEPriceDownloadCronJob completed in {elapsed_time_str}')


# Download US prices. 612 prices.
# Trust start time: 03:30
# Run time 3:25
# Should complete by 06:55
class DailyUSPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['00:28']
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    #schedule = Schedule(run_every_mins=3)  # Run once a day
    code = 'trading_app.daily_us_price_download_cron_job'

    def do(self):
        start_time = time.time()  # Capture start time
        display_local_time()
        category_price_download('US stocks')
        display_local_time()

        end_time = time.time()  # Capture end time
        elapsed_time = end_time - start_time  # Compute elapsed time
        elapsed_time_str = format_elapsed_time(start_time, end_time)

        # Log or print the elapsed time. Here's an example of logging it:
        logger.info(f'DailyUSPriceDownloadCronJob completed in {elapsed_time_str}')


# Update metric scores for tickers
# Trust start time: 07:00
# Run time 0:20
# Should complete by 07:20
class UpdateTickerMetricsCronJob(CronJobBase):
    RUN_AT_TIMES = ['04:00']
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    # schedule = Schedule(run_every_mins=60*24)  # Run once a day
    code = 'trading_app.update_ticker_metrics_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        start_time = time.time()  # Capture start time

        display_local_time()
        update_ticker_metrics()
        process_trading_opportunities()
        display_local_time()

        end_time = time.time()  # Capture end time
        elapsed_time = end_time - start_time  # Compute elapsed time
        elapsed_time_str = format_elapsed_time(start_time, end_time)

        # Log or print the elapsed time. Here's an example of logging it:
        logger.info(f'UpdateTickerMetricsCronJob completed in {elapsed_time_str}')

