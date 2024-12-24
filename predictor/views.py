from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib
import os

# Load the saved model
# model = joblib.load('C:/Users/LenovoX1/Desktop/Data Science Projects/customer_sales_predictor/best_sales_predictor.pkl')
# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_sales_predictor.pkl')
model = joblib.load(model_path)

def predict_sales(request):
    if request.method == 'POST':
        data = request.POST
        date = pd.to_datetime(data['date'])
        marketing_spend = float(data['marketing_spend'])
        day_of_year = date.dayofyear
        X_new = pd.DataFrame({'day_of_year': [day_of_year], 'marketing_spend': [marketing_spend]})
        prediction = model.predict(X_new)[0]
        return JsonResponse({'prediction': prediction})
    return render(request, 'predictor/predict.html')
