from django.http import JsonResponse

def get_python_data(request):
    # Run your Python script here
    data = {"axis" : ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        "values": [5, 6, 5, 6, 5, 100, 4, 6]}  # Example data
    return JsonResponse(data)
