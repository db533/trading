from django.db import models

# Ticker Model
class Ticker(models.Model):
    symbol = models.CharField(max_length=10)
    company_name = models.CharField(max_length=255)
    is_daily = models.BooleanField(default=False)
    is_fifteen_min = models.BooleanField(default=False)
    is_five_min = models.BooleanField(default=False)
    ma_200_trend_strength = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    last_high_low = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    atr = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    avg_volume_100_days = models.IntegerField(null=True)
    cumulative_two_period_two_day_rsi = models.DecimalField(max_digits=10, decimal_places=2, null=True)

    def __str__(self):
        return self.symbol

# Base Price Model
class BasePrice(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    datetime = models.DateTimeField()
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
