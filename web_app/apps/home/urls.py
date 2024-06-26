# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path, include
from apps.home import views
from apps.static.test import get_python_data
from apps.backend.correlation_matrix import correlation_matrix
from apps.backend.correlation_matrix_interpo import correlation_matrix_interpo
from apps.backend.graph_zoom_in import chart
from apps.backend.Data_management import load_data
from apps.home.views import download_file
from . import dash_app
import warnings
warnings.filterwarnings("ignore")

urlpatterns = [

    # The home page
    path('', views.index, name='home'),

    path('run/', get_python_data, name='run-script'),

    path('chart.html', chart, name='chart'),
    
    path('correlation-no-interpo/', correlation_matrix, name='run-script-correlation'),

    path('correlation-interpo/', correlation_matrix_interpo, name='run-script-correlation-interpol'),

    path('download/', download_file, name = 'download'),

    path('tables.html', load_data, name='chart'),

    path('test.html', include('django_plotly_dash.urls')),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]

#temporary solution
import apps.home.preprocessing as preprocessing
print('Calling preprocessing method'+'.'*50)
preprocessing.preprocessing()
print('Preprocessing done'+'.'*50)