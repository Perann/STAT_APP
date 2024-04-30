# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import pandas as pd


@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


import mimetypes

def download_file(request):
    df_1 = pd.read_excel("./apps/tmp_data_return.xlsx")
    df_alternative_asset = pd.read_excel("../EnsaeAlternativeTimeSeries.xlsx", sheet_name="Alternative Asset")
    df_classic_asset = pd.read_excel("../EnsaeAlternativeTimeSeries.xlsx", sheet_name= 'Classic Asset')


    with pd.ExcelWriter('./apps/output.xlsx', engine='xlsxwriter') as writer:
        df_alternative_asset.to_excel(writer, sheet_name='Alternative_asset')
        df_classic_asset.to_excel(writer, sheet_name='Classic_asset')
        df_1.to_excel(writer, sheet_name='return_Alternative')


    fl_path = "./apps/output.xlsx"
    filename = "output.xlsx"

    with open(fl_path, 'rb') as fl:
        mime_type, _ = mimetypes.guess_type(fl_path)
        response = HttpResponse(fl.read(), content_type=mime_type)
        response['Content-Disposition'] = f"attachment; filename={filename}"
        return response