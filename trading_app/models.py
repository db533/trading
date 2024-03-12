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
import uuid

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
    magnitude = models.PositiveIntegerField(default=1)

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
    confirmed = models.BooleanField(default=True) # Has price action confirmed this trading opp (True) or is it not yet confirmed. (False)
    stop_loss_price = models.FloatField(null=True)
    profit_taker_price = models.FloatField(null=True)
    reward_risk = models.FloatField(null=True)
    stop_loss_triggered = models.BooleanField(default=False, null=True)
    amount_initially_invested_currency = models.FloatField(null=True)
    amount_still_invested_currency = models.FloatField(null=True)
    income_from_sales_eur = models.FloatField(null=True)
    realised_profit_eur = models.FloatField(null=True)
    commissions_total_eur = models.FloatField(null=True)

    def __str__(self):
        return f"{self.ticker.symbol} - {self.strategy.name}"

    def update_computed_values(self):
        try:
            trades = self.trades.all()  # Get all related trades
            units = 0
            profit_currency = 0
            profit_eur = 0
            amount_initially_invested = 0
            commissions_total_eur = 0
            income_from_sales_eur = 0
            purchase_price = 0
            for trade in trades:
                # Read values from trade:
                deal_price = trade.price
                commission_amount = trade.commission
                unit_amount = trade.units
                status = trade.status
                # Check if the value is blank. If so, set to 0
                if deal_price is None:
                    deal_price = 0
                if commission_amount is None:
                    commission_amount = 0
                if unit_amount is None:
                    unit_amount = 0
                if trade.action == '1' and status == '2':  # Assuming '1' is Buy
                    units += unit_amount
                    #profit_currency -= ((unit_amount * deal_price) - commission_amount)
                    profit_eur -= (((unit_amount * deal_price) + commission_amount)  * trade.rate_to_eur)
                    commissions_total_eur += (trade.commission * trade.rate_to_eur)
                    purchase_price = deal_price
                    amount_initially_invested += (unit_amount * deal_price)
                elif trade.action == '0' and status == "2":  # Assuming '0' is Sell
                    units -= unit_amount
                    #profit_currency += ((unit_amount * deal_price) - commission_amount)
                    income_from_sales_eur += ((unit_amount * deal_price) * trade.rate_to_eur)
                    profit_eur += (((unit_amount * deal_price) - commission_amount) * trade.rate_to_eur)
                    commissions_total_eur += (trade.commission * trade.rate_to_eur)
            self.amount_initially_invested_currency = round(amount_initially_invested,2)
            self.amount_still_invested_currency = round(units * purchase_price, 2)
            #self.realised_profit_currency = round(profit_currency - commissions_total,2)
            self.income_from_sales_eur = round(income_from_sales_eur,2)
            self.realised_profit_eur = round(profit_eur, 2)
            self.commissions_total_eur = round(commissions_total_eur, 2)
            self.save()
        except Exception as e:
            print(f'Error in update_computed_values(): {e}')

    def save(self, *args, **kwargs):
        if self.stop_loss_price is not None and self.profit_taker_price is not None:
            # Get the Buy trades that are linked to this TradingOpp
            trades = self.trades.filter(action=1)  # Get all related Buy trades
            if len(trades) > 0:
                # We have a Buy trade either planned or executed. Use this price for the calculations.
                for trade in trades:
                    transaction_price = trade.price
            else:
                # We do not have a Buy trade associated, so use the latest close price.
                daily_price = self.ticker.dailyprice_set.order_by('-datetime').first()
                if daily_price:  # Check if a DailyPrice instance was found
                    latest_price = float(daily_price.close_price)
                    transaction_price = latest_price
                else:
                    transaction_price = None
            if transaction_price:
                action_buy = self.action_buy
                try:
                    if action_buy == "1":
                        # Action is a buy. Compute reward / risk this way:
                        self.reward_risk = round(
                            (self.profit_taker_price - transaction_price) * 100 / (transaction_price - self.stop_loss_price))
                    else:
                        # Action is a sell. Compute reward / risk this way:
                        self.reward_risk = round(
                            (transaction_price - self.profit_taker_price) * 100 / (self.stop_loss_price - transaction_price))

                except ZeroDivisionError:
                    self.reward_risk = None  # Handle division by zero if purchase_price equals stop_loss_price

        super().save(*args, **kwargs)  # Call the "real" save() method after computing reward_risk


class WaveType(models.Model):
    name = models.CharField(max_length=100)
    rules = models.JSONField(default=list)  # Holds rule definitions for this wave type

    def __str__(self):
        return self.name

class WaveInstance(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    start_sp = models.ForeignKey(SwingPoint, on_delete=models.CASCADE, related_name='starting_waves', verbose_name="Start Swing Point")
    end_sp = models.ForeignKey(SwingPoint, on_delete=models.CASCADE, related_name='ending_waves', verbose_name="End Swing Point")
    wave_type = models.ForeignKey(WaveType, on_delete=models.CASCADE, related_name="waves")

    def __str__(self):
        return f"{self.wave_type.name} from {self.start_sp.date} to {self.end_sp.date}"

    def evaluate_rules(self, n_1_wave=None,n_2_wave=None, n_3_wave=None, n_4_wave=None):
        score = 0
        for rule in self.wave_type.rules:  # Access rules from the wave_type
            if rule['type'] == 'hard_rule':
                if not self.evaluate_hard_rule(rule, n_1_wave):
                    return 0  # Failure to meet a hard rule
            elif rule['type'] == 'guideline':
                adherence = self.evaluate_guideline(rule)
                score += adherence * rule['weighting']
        return score

    def evaluate_hard_rule(self, rule, n_1_wave=None,n_2_wave=None, n_3_wave=None, n_4_wave=None):
        """
        Evaluates a hard rule based on the rule definition and potentially the context
        of a previous wave. If the rule cannot be evaluated due to lack of prior data,
        it is assumed to be valid.

        :param previous_wave: An instance of Wave or None, representing the previous wave in the sequence.
        :return: True if the rule is satisfied or cannot be evaluated due to lack of prior data, False otherwise.
        """
        # Example hard rule evaluation
        if rule['description'] == "Wave 2 cannot retrace more than 100% of Wave 1":
            # If there's no previous wave (e.g., this is the first wave), assume the rule is valid
            if n_1_wave is None:
                return True
            # Otherwise, evaluate the rule based on available data
            # This assumes you have a mechanism to compare the relevant properties of the current and previous waves
            return self.end_sp.price_object.price > n_1_wave.start_sp.price_object.price
        else:
            # Default case for unrecognized rules or those without specific evaluation logic
            return True  # Assume valid if the rule is not recognized or cannot be evaluated

    def evaluate_guideline(self, rule):
        # Implement logic to evaluate guideline adherence
        # Return a value between 0 and 1 representing % adherence
        pass

class WavePattern(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    waves = models.ManyToManyField(WaveInstance, through='WaveSequence')

    def __str__(self):
        return self.name

class WaveSequence(models.Model):
    wave_pattern = models.ForeignKey(WavePattern, on_delete=models.CASCADE)
    wave = models.ForeignKey(WaveInstance, on_delete=models.CASCADE)
    order = models.PositiveIntegerField()

    class Meta:
        ordering = ['order']

class Trade(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    tradingopp = models.ForeignKey(
        TradingOpp,
        on_delete=models.SET_NULL,  # This will set the tradingopp field to NULL instead of deleting the trade
        related_name='trades',
        verbose_name="Trading Opportunity",
        null=True,  # Allows the tradingopp field to be null
        blank=True  # Optional: Allows the field to be blank in forms/admin
    )
    ACTION = (
        ('1', 'Buy'),
        ('0', 'Sell'),
    )
    action = models.CharField(max_length=1,
                              choices=ACTION,
                              default='1',
                              help_text='Is the trade a buy or a sell?',
                              verbose_name='Action')
    planned = models.BooleanField(default=True) # Is the purchase / sale planned or actually executed.
    STATUS = (
        ('0', 'Planned'),
        ('1', 'Trade submitted'),
        ('2', 'Trade executed'),
    )
    status = models.CharField(max_length=1,
                              choices=STATUS,
                              default='0',
                              help_text='Status of the trade',
                              verbose_name='Status')
    price = models.FloatField()
    units = models.FloatField(help_text='Quantity of units bought or sold',)
    rate_to_eur = models.FloatField(verbose_name='Multiplier to convert price to EUR')
    commission = models.FloatField(verbose_name='Commission in foreign currency')
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.tradingopp.ticker.symbol} - {self.action} - {self.date}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)  # Call the "real" save() method.
        if self.tradingopp:  # Check if the trade is linked to a TradingOpp
            self.tradingopp.update_computed_values()

class Params(models.Model):
    TYPES = (
        ('0', 'Float'),
        ('1', "Datetime"),
        ('2', "String"),
    )

    key = models.CharField(max_length=100)
    value = models.CharField(max_length=100)
    type = models.CharField(max_length=2,
                              choices=TYPES,
                              default='0',
                              help_text='Data type for the value',
                              verbose_name='Data type')

    def __str__(self):
        return f"{self.key} - {self.value}"