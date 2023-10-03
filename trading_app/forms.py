from django import forms
from .models import Ticker, TickerCategory

class TickerForm(forms.ModelForm):
    class Meta:
        model = Ticker
        fields = ['symbol', 'company_name', 'categories' , 'is_daily', 'is_fifteen_min', 'is_five_min']


class CategorySelectForm(forms.Form):
    NOT_DEFINED_CHOICE = ('not_defined', 'Not defined')

    categories = forms.ModelMultipleChoiceField(
        queryset=TickerCategory.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=False  # Allow no selection
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a 'Not defined' choice
        self.fields['categories'].choices = [
                                                self.NOT_DEFINED_CHOICE
                                            ] + [(cat.id, cat.name) for cat in TickerCategory.objects.all()]
