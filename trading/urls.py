import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from django.contrib import admin
from django.urls import path, include
from trading_app.views import *
from trading_app.tasks import *

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
    path('ticker/<int:ticker_id>/get-prices/<str:timeframe>/', manual_download, name='manual_download'),
    path('update_metrics/<str:ticker_symbol>/', update_metrics_view, name='update_metrics'),
    path('ticker/<int:ticker_id>/', ticker_detail, name='ticker_detail'),
    path('ticker_delete/<int:ticker_id>/', ticker_delete, name='ticker_delete'),
    path('add-ticker/', add_ticker, name='ticker_add'),
    path('manual_category_download/<str:category_name>/', manual_category_download, name='manual_category_download'),
    #path('manual_category_download/<str:category_name>/', background_manual_category_download, name='manual_category_download'),
    path('delete_daily_price/', delete_daily_price, name='delete_daily_price'),
    path('delete_daily_price/<str:symbol>/', delete_daily_price, name='delete_daily_price_with_symbol'),
    path('trading-opps/', trading_opps_view, name='trading-opps'),
    path('trading-opps-recent/', trading_opps_sorted_view, name='trading_opps_sorted'),
    path('swing_point_graph/<int:opp_id>/', generate_swing_point_graph_view, name='swing_point_graph'),
    path('trading-opps-filtered/', trading_opps_filtered, name='trading-opps-filtered'),
    path('tasks/', task_queue_view, name='task_queue'),
    path('ticker-graph/<str:ticker_symbol>/', generate_ticker_graph_view, name='ticker-graph'),
    path('trading-opp/update/<int:opp_id>/', update_tradingopp, name='update_tradingopp'),
    path('trading-opps-with-trades/<str:status>/', trading_opps_with_trades_view, name='trading_opps_with_trades'),
    path('trading-performance-list/', trade_performance_list, name='trade_performance_list'),
    path('trading-opps-with-planned-trades/', trading_opps_with_planned_trades, name='trading_opps_with_planned_trades'),
    path('trading-opps-with-scheduled-trades/', trading_opps_with_scheduled_trades, name='trading_opps_with_scheduled_trades'),
    path('trading-opps-with-executed-trades/', trading_opps_with_executed_trades, name='trading_opps_with_executed_trades'),
    path('trading-opps-with-open-trades/', trading_opps_with_open_trades, name='trading_opps_with_open_trades'),
    path('trading-opps-with-completed-trades/', trading_opps_with_completed_trades, name='trading_opps_with_completed_trades'),
    path('update_all_strategies/', update_all_strategies, name='update_all_strategies'),
    path('update-trades/', update_trades, name='update_trades'),
    path('delete_ticker/', delete_ticker_view, name='delete_ticker'),
    path('params/edit-all/', edit_all_params, name='edit_all_params'),
    path('monthly-performance/', monthly_trading_performance_view, name='monthly_performance'),
]

from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += [
    path('accounts/', include('django.contrib.auth.urls')),
]
