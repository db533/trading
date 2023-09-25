# trading_app/cron.py

from django_cron import CronJobBase, Schedule
from .update_ticker_metrics import update_ticker_metrics
from .test_cron_job import test_cron_job

class UpdateTickerMetricsCronJob(CronJobBase):
    schedule = Schedule(run_every_mins=60*24)  # Run once a day

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
