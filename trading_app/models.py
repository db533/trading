from django.db import models
import re
import ast
from decimal import Decimal
from datetime import datetime, timezone
import pytz
import logging
logger = logging.getLogger('django')
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType


class TickerCategory(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

# Ticker Model
class Ticker(models.Model):
    symbol = models.CharField(max_length=10)
    company_name = models.CharField(max_length=255)
    categories = models.ManyToManyField(TickerCategory, blank=True)
    is_daily = models.BooleanField(default=True)
    is_fifteen_min = models.BooleanField(default=False)
    is_five_min = models.BooleanField(default=False)
    ma_200_trend_strength = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    swing_point_current_trend = models.IntegerField(null=True, default=0)
    last_high_low = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    atr = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    avg_volume_100_days = models.IntegerField(null=True)
    cumulative_two_period_two_day_rsi = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    seven_day_max = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    seven_day_min = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    nearest_level_value = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    nearest_level_type = models.CharField(max_length=20, null = True)
    nearest_level_days_since_retest = models.IntegerField(null = True)
    nearest_level_percent_distance = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    tae_strategy_score = models.DecimalField(max_digits=5, decimal_places=1, null=True)
    bearish_detected = models.DecimalField(max_digits=2, decimal_places=0, null=True)
    bullish_detected = models.DecimalField(max_digits=2, decimal_places=0, null=True)
    reversal_detected = models.DecimalField(max_digits=2, decimal_places=0, null=True)
    bearish_reversal_detected = models.DecimalField(max_digits=2, decimal_places=0, null=True)
    bullish_reversal_detected = models.DecimalField(max_digits=2, decimal_places=0, null=True)
    patterns_detected = models.CharField(max_length=255, null=True, blank=True)
    uptrend_hl_range = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    uptrend_hl_percent = models.DecimalField(max_digits=5, decimal_places=1, null=True)


    def __str__(self):
        return self.symbol

# Base Price Model
class BasePrice(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    datetime = models.DateTimeField()
    datetime_tz = models.DateTimeField(null=True)
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    percent_change = models.FloatField( null=True, blank=True)
    bearish_detected = models.DecimalField(max_digits=2, decimal_places=0)
    bullish_detected = models.DecimalField(max_digits=2, decimal_places=0)
    reversal_detected = models.DecimalField(max_digits=2, decimal_places=0)
    bearish_reversal_detected = models.DecimalField(max_digits=2, decimal_places=0)
    bullish_reversal_detected = models.DecimalField(max_digits=2, decimal_places=0)
    patterns_detected = models.CharField(max_length=255, null=True, blank=True)
    volume = models.IntegerField()
    level = models.DecimalField(max_digits=10, decimal_places=3, null=True)
    level_type = models.IntegerField(default=0) # 1 = Support, 2 = Resistance
    level_strength = models.IntegerField(default=0)
    ema_200 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    ema_50 = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    trend = models.IntegerField(default=0) # 1 = Rising, -1 = Falling, 0 = Stable
    swing_point_label = models.CharField(max_length=3, null=True, blank=True)
    swing_point_current_trend = models.IntegerField(default=0) # 1 = Up-trend, -1 Down-trend, 0 = Pattern not evident
    healthy_bullish_count = models.IntegerField(default=0)  # Count of number of healthy bullish candles since last swing point.
    healthy_bearish_count = models.IntegerField(default=0)  # Count of number of healthy bearish candles since last swing point.
    candle_count_since_last_swing_point = models.IntegerField(default=0)

    #resistance_strength = models.IntegerField()

    class Meta:
        abstract = True

# Daily Price Model
class DailyPrice(BasePrice):
    pass

# 15 Minute Price Model
class FifteenMinPrice(BasePrice):
    pass

# 5 Minute Price Model
class FiveMinPrice(BasePrice):
    pass

# 1 Minute Price Model
class OneMinPrice(BasePrice):
    pass

class TradingStrategy(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    # You can add more fields here to define strategy-specific parameters

    def __str__(self):
        return self.name


class SwingPoint(models.Model):
    ticker = models.ForeignKey('Ticker', on_delete=models.CASCADE, related_name='swing_points')
    date = models.DateTimeField(default=None)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    label = models.CharField(max_length=2)
    candle_count_since_last_swing_point = models.IntegerField(default=None)

    # Fields for linking to a Price model instance
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, limit_choices_to={
        'model__in': ('dailyprice', 'fifteenminprice', 'fiveminprice', 'oneminprice')})
    object_id = models.PositiveIntegerField()
    price_object = GenericForeignKey('content_type', 'object_id')

    def __str__(self):
        return f"{self.ticker.symbol} - {self.date} - {self.label}"


class TradingOpp(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    strategy = models.ForeignKey(TradingStrategy, on_delete=models.CASCADE)
    datetime_identified = models.DateTimeField(auto_now_add=True)
    metrics_snapshot = models.JSONField()  # Snapshot of metrics at the time of identification
    is_active = models.BooleanField(default=True)
    count = models.IntegerField(default=1)
    action_buy = models.BooleanField(default=True)
    swing_points = models.ManyToManyField('SwingPoint', related_name='trading_opps', blank=True)
    datetime_invalidated = models.DateTimeField(null=True, blank = True)

    def __str__(self):
        return f"{self.ticker.symbol} - {self.strategy.name}"

