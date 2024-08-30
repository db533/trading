from django import forms
from .models import Ticker, TickerCategory

class TickerForm(forms.ModelForm):
    class Meta:
        model = Ticker
        fields = ['symbol', 'company_name', 'categories' , 'is_daily', 'is_fifteen_min', 'is_five_min']


# forms.py
class TickerCategoryForm(forms.ModelForm):
    id = forms.IntegerField(widget=forms.HiddenInput())  # Ensure the ID field is included and hidden
    categories = forms.ModelMultipleChoiceField(
        queryset=TickerCategory.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=False
    )

    class Meta:
        model = Ticker
        fields = ['id', 'categories']

class CategorySelectForm(forms.Form):
    NOT_DEFINED_CHOICE = ('not_defined', 'Not defined')
    categories = forms.ModelMultipleChoiceField(
        queryset=TickerCategory.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=False
    )
    uptrend = forms.BooleanField(required=False)
    downtrend = forms.BooleanField(required=False)
    swing_trend = forms.BooleanField(required=False)
    tae_score = forms.BooleanField(required=False)
    two_period_cum_rsi = forms.BooleanField(required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a 'Not defined' choice
        self.fields['categories'].choices = [
                                                self.NOT_DEFINED_CHOICE
                                            ] + [(cat.id, cat.name) for cat in TickerCategory.objects.all()]

class TickerSymbolForm(forms.Form):
    symbol = forms.CharField(max_length=10, label='Ticker Symbol')

class SymbolForm(forms.Form):
    symbols = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Enter symbols separated by commas'}), label='Symbols')

# forms.py
from django import forms
from .models import Params

class ParamsForm(forms.ModelForm):
    class Meta:
        model = Params
        fields = ['key', 'value', 'type']
