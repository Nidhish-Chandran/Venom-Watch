"""
Root URL Configuration for Venom Watch
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from accounts import views as accounts_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls')),
    path('accounts/', include('accounts.urls')),
    path('reports/', include('reports.urls')),
    path('ai/', include('ai.urls')),
    path('hotspots/', include('hotspots.urls')),
    path('emergency/', include('emergency.urls')),
    path('ranger/', include('ranger.urls')),
    path('media/verification_docs/<path:path>', accounts_views.protected_media_block_view),
    path('snake-detector/', include('snake_detector.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
