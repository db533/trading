# trading_app/cron.py

from django_cron import CronJobBase, Schedule
from .update_ticker_metrics import update_ticker_metrics
from .nightly_price_download import download_prices
from .test_cron_job import test_cron_job

class DailyPriceDownloadCronJob(CronJobBase):
    RUN_AT_TIMES = ['00:01']  # Run at 1:00 AM local time
    schedule = Schedule(run_at_times=RUN_AT_TIMES)
    # schedule = Schedule(run_every_mins=1)  # Run once a day
    code = 'trading_app.daily_price_download_cron_job'

    def do(self):
        # Run the update_ticker_metrics function
        download_prices()

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
