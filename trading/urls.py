from django.contrib import admin
from django.urls import path, include

from django.views.generic import RedirectView

urlpatterns = [
    path('', RedirectView.as_view(url='trading/', permanent=True)),
    path('admin/', admin.site.urls),
    # Include your app's URL configurations here
    # For example, if your app's name is 'trading_app':
    # path('trading/', include('trading_app.urls')),
]

from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
