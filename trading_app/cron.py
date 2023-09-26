# trading_app/cron.py

from django_cron import CronJobBase, Schedule
from .update_ticker_metrics import update_ticker_metrics
from .price_download import download_prices
from .test_cron_job import test_cron_job

class DailyPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['00:01']  # Run at 1:00 AM local time
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    # schedule = Schedule(run_every_mins=1)  # Run once a day
    code = 'trading_app.daily_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='Daily')

class FifteenMinsPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['{:02d}:{:02d}'.format(hour, minute) for hour in range(9, 23) for minute in [1, 16, 31, 46]]
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    code = 'trading_app.15_min_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='15 mins')

class FiveMinsPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['{:02d}:{:02d}'.format(hour, minute) for hour in range(9, 23) for minute in [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]]
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    code = 'trading_app.5_min_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices(timeframe='5 mins')

class UpdateTickerMetricsCronJob(CronJobBase):
    RUN_AT_TIMES = ['01:00']  # Run at 1:00 AM local time
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    # schedule = Schedule(run_every_mins=60*24)  # Run once a day
    code = 'trading_app.update_ticker_metrics_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        update_ticker_metrics()

class TestCronJob(CronJobBase):
    schedule = Schedule(run_every_mins=1)  # Run once a day
    code = 'trading_app.test_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        test_cron_job()
