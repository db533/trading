from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(Ticker)
admin.site.register(TickerCategory)
admin.site.register(DailyPrice)
admin.site.register(FifteenMinPrice)
admin.site.register(FiveMinPrice)
