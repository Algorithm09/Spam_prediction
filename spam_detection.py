import nltk
from nltk.corpus import stopwords
import pandas as pd
import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer

po = PorterStemmer()
with open("spam_classifier_model.pkl", "rb") as file:
    cv,model = pickle.load(file)

st.title("Spam Email Detection App")
st.subheader("")
st.sidebar.header("User input Parameter")
st.sidebar.info("Copy and paste the content of the email you want to detect ")



st.info("# The model predict if an  email is spam or not")

st.subheader("User input Parameter")




def user_input():
    email = st.sidebar.text_input("Paste Here")
    st.write(email)
    df = re.sub("[^a-zA-Z]", " ", email)
    df = df.lower()
    df = df.split()
    df = [po.stem(words) for words in df if words not in set(stopwords.words("english"))]
    df = " ".join(df)

    return df
data = user_input()
st.subheader("Prediction ")

vec = cv.transform([data]).toarray()

pred = model.predict(vec)
proba = model.predict_proba(vec)

if pred == 1:
    st.write("prediction is ", pred)
    st.write("The probabilit of spam is", proba * 100)
    st.write("this is a spam message")
else:
    st.write('prediction is ', pred)
    st.write("The probabilit of not spam is ", (proba * 100))
    st.write("This email is not a spam")


print(vec)