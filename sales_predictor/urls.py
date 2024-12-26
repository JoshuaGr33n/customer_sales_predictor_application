"""sales_predictor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from predictor.views import predict_sales, predict_health, classify_articles

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', predict_sales, name='predict_sales'),
    path('predict_health/', predict_health, name='predict_health'),
    path('classifier/', classify_articles, name='classify_text'),
]
