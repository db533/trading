from django.db import models

# Ticker Model
class Ticker(models.Model):
    symbol = models.CharField(max_length=10)
    company_name = models.CharField(max_length=255)
    is_daily = models.BooleanField(default=False)
    is_fifteen_min = models.BooleanField(default=False)
    is_five_min = models.BooleanField(default=False)
    is_one_min = models.BooleanField(default=False)

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
    bullish_engulfing = models.BooleanField(default=False) # Bullish
    bullish_harami = models.BooleanField(default=False) # Bullish
    hammer = models.BooleanField(default=False) # Bullish
    inverted_hammer = models.BooleanField(default=False) # Bullish
    hanging_man = models.BooleanField(default=False) # Bullish
    shooting_star = models.BooleanField(default=False) # Bullish
    bearish_engulfing = models.BooleanField(default=False) # Bearish
    bearish_harami = models.BooleanField(default=False) # Bearish
    dark_cloud_cover = models.BooleanField(default=False) # Bearish
    gravestone_doji = models.BooleanField(default=False)  # Bearish
    dragonfly_doji = models.BooleanField(default=False) # Reversal
    doji_star = models.BooleanField(default=False) # Reversal
    piercing_pattern = models.BooleanField(default=False)  # Reversal
    morning_star = models.BooleanField(default=False)  # Bullish Reversal
    morning_star_doji = models.BooleanField(default=False)  # Bullish Reversal
    #evening_star = models.BooleanField(default=False) # Bearish Reversal
    #evening_star_doji = models.BooleanField(default=False)  # Bearish Reversal
    three_white_soldiers = models.BooleanField(default=False) # Bullish
    volume = models.IntegerField()

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
