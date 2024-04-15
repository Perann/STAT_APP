#import libraries
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connections
from django.contrib import messages
from django.urls import reverse
import pandas as pd
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go

def chart(request):

    fig = px.line(x=[1, 2, 3], y=[1, 2, 3])

    fig = fig.to_html()

    context = {'fig': fig}
        
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'home/test.html', context=context)