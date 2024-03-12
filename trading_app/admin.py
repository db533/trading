from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(Ticker)
admin.site.register(TickerCategory)
admin.site.register(TradingStrategy)
admin.site.register(TradingOpp)
admin.site.register(DailyPrice)
admin.site.register(FifteenMinPrice)
admin.site.register(FiveMinPrice)
admin.site.register(WaveType)
admin.site.register(WaveInstance)
admin.site.register(WavePattern)
admin.site.register(WaveValueRule)
admin.site.register(WaveRuleCombineRule)
admin.site.register(WaveRuleCombineRelation)
admin.site.register(RuleEvaluation)
admin.site.register(Params)
