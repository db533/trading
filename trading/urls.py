from django.contrib import admin
from django.urls import path, include
from trading_app.views import *

from django.views.generic import RedirectView
from rest_framework.authtoken.views import obtain_auth_token
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', RedirectView.as_view(url='trading/', permanent=True)),
    path('admin/', admin.site.urls),
    path('trading/', index, name='index'),
    path('ticker-config/', ticker_config, name='ticker_config'),
    path('edit-ticker/<int:ticker_id>/', edit_ticker, name='edit_ticker'),
    path('login', login_view, name='login_view'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # Add logout view
    path('ticker/<int:ticker_id>/daily-prices/', daily_price_list, name='daily_price_list'),
    path('ticker/<int:ticker_id>/fifteen-min-prices/', fifteen_min_price_list, name='fifteen_min_price_list'),
    path('ticker/<int:ticker_id>/five-min-prices/', five_min_price_list, name='five_min_price_list'),
    path('update_metrics/', update_metrics_view, name='update_metrics'),
    path('ticker/<int:ticker_id>/', ticker_detail, name='ticker_detail'),
]

from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
]
