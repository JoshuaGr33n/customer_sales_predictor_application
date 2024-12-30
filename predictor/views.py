from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import joblib
from bs4 import BeautifulSoup
import os
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from django import forms
from predictor.model_module.load_models import *



def predict_sales(request):
    model =  load_sales_prediction_model()
    if request.method == 'POST':
        data = request.POST
        date = pd.to_datetime(data['date'])
        marketing_spend = float(data['marketing_spend'])
        day_of_year = date.dayofyear
        X_new = pd.DataFrame({'day_of_year': [day_of_year], 'marketing_spend': [marketing_spend]})
        prediction = model.predict(X_new)[0]
        return JsonResponse({'prediction': prediction})
    return render(request, 'predictor/predict.html')

def predict_health(request):
    model =  load_health_prediction_model()
    if request.method == 'POST':
        data = request.POST
        age = int(data['age'])
        bmi = float(data['bmi'])
        blood_pressure = float(data['blood_pressure'])
        X_new = pd.DataFrame({'age': [age], 'bmi': [bmi], 'blood_pressure': [blood_pressure]})
        prediction = model.predict(X_new)[0]
        
        # Convert prediction to native Python type
        prediction = int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction)
        
        return JsonResponse({'prediction': prediction})
    return render(request, 'predictor/predict_health.html')


# News Articles Classifier
def fetch_articles(url, article_tag, title_tag, content_tag, title_class=None, content_class=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        for item in soup.find_all(article_tag):
            title_element = item.find(title_tag, class_=title_class) if title_class else item.find(title_tag)
            content_element = item.find(content_tag, class_=content_class) if content_class else item.find(content_tag)
            if title_element and content_element:
                title = title_element.get_text()
                content = content_element.get_text()
                if content:  # Ensure content is not empty
                    articles.append({'title': title, 'content': content})
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles from {url}: {e}")
        return []

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def classify_articles(request):
    model, vectorizer, target_names = news_articles_classifier()
    labeled_articles = []
    # Fetch articles from CNN
    cnn_articles = fetch_articles('https://edition.cnn.com/world', 'div', 'span', 'div', title_class='container__headline-text')
    
    # Preprocess the articles
    preprocessed_articles = [preprocess_text(article['content']) for article in cnn_articles]
    
    # Vectorize the text data
    X = vectorizer.transform(preprocessed_articles)
    
    # Predict the categories
    predictions = model.predict(X)
    
    # Map the predicted labels to category names
    labeled_articles = [{'title': article['title'], 'category': target_names[pred]} for article, pred in zip(cnn_articles, predictions)]
    
    return render(request, 'predictor/classifier.html', {'labeled_articles': labeled_articles})

# housing price
def predict_house_price(request):
    model = load_house_model()
    # Example prediction with dummy data
    prediction = model.predict([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0, 15.3, 396.90, 4.98]])
    return render(request, 'predictor/housing.html', {'prediction': prediction})

# Translate
def translate(request):
    model, vectorizer = load_translate_model()
    if request.method == 'POST':
        english_text = request.POST['english_text']
        english_text_vec = vectorizer.transform([english_text])
        translation = model.predict(english_text_vec)
        return render(request, 'predictor/translate.html', {'translation': translation[0]})
    return render(request, 'predictor/translate.html')


# Employee Turnover
class TurnoverPredictionForm(forms.Form):
    job_satisfaction = forms.FloatField(label="Job Satisfaction (0-1)")
    monthly_income = forms.IntegerField(label="Monthly Income")
   
def predict_turnover(request):
    model, scaler = load_model_turnover()
    if request.method == 'POST':
        form = TurnoverPredictionForm(request.POST)
        if form.is_valid():
            job_satisfaction = float(request.POST.get('job_satisfaction'))
            monthly_income = float(request.POST.get('monthly_income'))
            # Default values for other features
            years_at_company = 5 
            total_working_years = 10  
            work_life_balance = 3  
            income_job_satisfaction = monthly_income * job_satisfaction 

            # Create input data for prediction
            input_data = np.array([[job_satisfaction, monthly_income, 
                                    income_job_satisfaction, 
                                    years_at_company, 
                                    total_working_years, 
                                    work_life_balance
                                    ]])

            # Scale the input data
            input_data_scaled = scaler.transform(input_data) 

            prediction = model.predict(input_data_scaled)

            if prediction[0] == 'Yes':
                result = "Employee is likely to leave."
            else:
                result = "Employee is likely to stay."

            return render(request, 'predictor/employee_turnover.html', {'result': result})
    else:
        form = TurnoverPredictionForm()
    return render(request, 'predictor/employee_turnover.html', {'form': form})
