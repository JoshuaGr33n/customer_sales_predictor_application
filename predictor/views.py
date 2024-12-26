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




# Load the saved model
# model = joblib.load('C:/Users/LenovoX1/Desktop/Data Science Projects/customer_sales_predictor/best_sales_predictor.pkl')

base = 'Path'
# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_sales_predictor.pkl')
model = joblib.load(model_path)

# Load the saved health model
health_model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_health_predictor.pkl')
health_model = joblib.load(health_model_path)

# news article classifier
classifier_model = joblib.load(f'{base}classifier/pkl/news_category_model.pkl')
vectorizer = joblib.load(f'{base}classifier/pkl/news_category_vectorizer.pkl')
target_names = joblib.load(f'{base}classifier/pkl/target_names.pkl')

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

def predict_health(request):
    if request.method == 'POST':
        data = request.POST
        age = int(data['age'])
        bmi = float(data['bmi'])
        blood_pressure = float(data['blood_pressure'])
        X_new = pd.DataFrame({'age': [age], 'bmi': [bmi], 'blood_pressure': [blood_pressure]})
        prediction = health_model.predict(X_new)[0]
        
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
    labeled_articles = []
    # Fetch articles from CNN
    cnn_articles = fetch_articles('https://edition.cnn.com/world', 'div', 'span', 'div', title_class='container__headline-text')
    
    # Preprocess the articles
    preprocessed_articles = [preprocess_text(article['content']) for article in cnn_articles]
    
    # Vectorize the text data
    X = vectorizer.transform(preprocessed_articles)
    
    # Predict the categories
    predictions = classifier_model.predict(X)
    
    # Map the predicted labels to category names
    labeled_articles = [{'title': article['title'], 'category': target_names[pred]} for article, pred in zip(cnn_articles, predictions)]
    
    return render(request, 'predictor/classifier.html', {'labeled_articles': labeled_articles})