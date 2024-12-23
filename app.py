import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



with open('fake_news_model.pkl', 'rb') as model_file:
    model_LSTM = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as model_file:
    vectorizer = pickle.load(model_file)


# Streamlit App
st.title("Fake News Classifier")
input_text = st.text_area("Enter news content here:")

if st.button("Classify"):
    text = input_text
    vectorizer = TfidfVectorizer(max_features=5000)
    sample_article_vec = vectorizer.fit_transform([text])
    sample_article_pad = pad_sequences(sample_article_vec.toarray(), maxlen=100)
    prediction = np.argmax(model_LSTM.predict(sample_article_pad))
    print(type(text))
    
    
    print("prediction value:", prediction)
    result = "Fake" if prediction == 1 else "Real"
    st.write(f"Prediction: {result}")