import pickle
import joblib
import os


base = ''


def load_sales_prediction_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_sales_predictor.pkl') 
    model = joblib.load(model_path)
    return model


def load_health_prediction_model():
    health_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_health_predictor.pkl')
    model = joblib.load(health_model_path)
    return model


def load_translate_model():
    with open(f'{base}models/translate/translation_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open(f'{base}models/translate/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def news_articles_classifier():
    model = joblib.load(f'{base}classifier/pkl/news_category_model.pkl')
    vectorizer = joblib.load(f'{base}classifier/pkl/news_category_vectorizer.pkl')
    target_names = joblib.load(f'{base}classifier/pkl/target_names.pkl')
    
    return model, vectorizer, target_names


def load_house_model():
    with open(f'{base}models/HousingModel.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def load_model_turnover():
    with open(f'{base}models/employee_turnover/model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open(f'{base}models/employee_turnover/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler
