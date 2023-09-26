from django import forms
from .models import Ticker

class TickerForm(forms.ModelForm):
    class Meta:
        model = Ticker
        fields = ['symbol', 'company_name', 'is_daily', 'is_fifteen_min', 'is_five_min']
