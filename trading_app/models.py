from django.db import models
import re
import ast
from decimal import Decimal
from datetime import datetime, timezone
import pytz
import logging
logger = logging.getLogger('django')


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
    # You can add more fields here to define strategy-specific parameters

    def __str__(self):
        return self.name


def parse_swing_points(swing_points_str):
    # Pattern to match the tuple structure in the string
    tuple_pattern = re.compile(r"\((datetime\.datetime\([^)]+\), Decimal\('[^)]+'\), '[^']+'\)")

    # Find all tuple strings
    tuple_strings = tuple_pattern.findall(swing_points_str)

    parsed_tuples = []
    for tuple_str in tuple_strings:
        # Parsing the datetime part
        datetime_part = re.search(r"datetime\.datetime\(([^)]+)\)", tuple_str).group(1)
        datetime_args = [int(arg) for arg in datetime_part.split(', ')[:6]]
        # Adjusting for timezone if present
        tzinfo_str = re.search(r"tzinfo=datetime\.timezone\.utc", tuple_str)
        tzinfo = timezone.utc if tzinfo_str else None
        datetime_obj = datetime(*datetime_args, tzinfo=tzinfo)

        # Parsing the Decimal part
        decimal_part = re.search(r"Decimal\('([^)]+)'\)", tuple_str).group(1)
        decimal_obj = Decimal(decimal_part)

        # Parsing the label
        label = re.search(r", '([^']+)'\)", tuple_str).group(1)

        parsed_tuples.append((datetime_obj, decimal_obj, label))

    return parsed_tuples


class TradingOpp(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    strategy = models.ForeignKey(TradingStrategy, on_delete=models.CASCADE)
    datetime_identified = models.DateTimeField(auto_now_add=True)
    metrics_snapshot = models.JSONField()  # Snapshot of metrics at the time of identification
    is_active = models.BooleanField(default=True)
    count = models.IntegerField(default=1)
    action_buy = models.BooleanField(default=True)
    recent_swing_points = models.JSONField(default=list) # date, price, label

    def get_swing_points_as_tuples(self):
        if self.recent_swing_points:
            try:
                swing_points_str = self.recent_swing_points[0]  # Assuming it's wrapped in a list
                return parse_swing_points(swing_points_str)
            except Exception as e:
                print(f"Error parsing swing points: {e}")
        return []

    def __str__(self):
        return f"{self.ticker.symbol} - {self.strategy.name}"
