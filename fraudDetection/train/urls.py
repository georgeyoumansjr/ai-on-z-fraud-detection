from django.contrib import admin
from django.urls import path,include
from django.http import HttpResponse
from django.contrib.auth import views as auth_views #to avoid yk
from django.contrib.auth.decorators import login_required
from train import views as trainViews

def index(request):
    return HttpResponse("This is a response")

urlpatterns = [
    path('',trainViews.addCsvView,name='upload'),
    path('overview/',trainViews.overview,name='overview'),
]