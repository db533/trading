from django.apps import AppConfig
from . import db_candlestick

class TradingAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'trading_app'
