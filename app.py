import numpy as np
import streamlit as st
import tensorflow as tf
import pickle
import nltk 
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

with open('fake_news_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit App
st.title("Fake News Classifier")
input_text = st.text_area("Enter news content here:")

if st.button("Classify"):
    text = input_text
    vectorizer = TfidfVectorizer(max_features=5000)
    sample_article_vec = vectorizer.fit_transform([text])
    sample_article_pad = pad_sequences(sample_article_vec.toarray(), maxlen=100)
    prediction = np.argmax(model_lSTM.predict(sample_article_pad))
    print(type(text))
    
    
    print("prediction value:", prediction)
    result = "Fake" if prediction == 1 else "Real"
    st.write(f"Prediction: {result}")
