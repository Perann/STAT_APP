# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path, include
from apps.home import views
from apps.static.test import get_python_data
from apps.backend.correlation_matrix import correlation_matrix
from apps.backend.graph_zoom_in import chart

urlpatterns = [

    # The home page
    path('', views.index, name='home'),

    path('run/', get_python_data, name='run-script'),

    path('chart.html', chart, name='chart'),

    path('correlation-no-interpo/', correlation_matrix, name='run-script-correlation'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]
