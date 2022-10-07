import streamlit as st
import json
from pickle import load
import pandas as pd
import sys

sys.path.insert(1, '../notebooks/scripts/development')
from preprocessing import translate_to_en, preprocess_text_series
from scipy.sparse import csr_matrix, hstack
import altair as alt


def load_data():
    # Load tf-idf vectorizer
    with open("../data/modeling/tfidf_1.pkl", "rb") as tfidf_file:
        tfidf_vec = load(tfidf_file)
        tfidf_file.close()

    # Load both scaler
    with open("../data/modeling/scaler_1.pkl", "rb") as tfidf_file:
        scaler = load(tfidf_file)
        tfidf_file.close()

    # Load model
    with open("../data/modeling/model_1_with_mf.pkl", "rb") as model_file:
        model = load(model_file)
        model_file.close()

    # Load token dictionary
    with open("../data/token_dictionary.json", "r") as token_dict_file:
        token_dictionary = json.load(token_dict_file)
        token_dict_file.close()

    return tfidf_vec, scaler, model, token_dictionary


def show_main_page(tfidf_vec, scaler, model, token_dictionary):
    st.markdown("<h1 style='text-align: center; color: green'>Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: green'>🎫</h1>", unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    text = st.text_input(label="Insert the text for the ticket you want to classify")

    is_clicked = st.button(label="Evaluate ticket text")

    if is_clicked:
        X_test = pd.Series([text])
        X_test = X_test.apply(lambda x: translate_to_en(x))
        X_test = preprocess_text_series(X_test, token_dictionary, True)

        text = X_test['text']
        tfidf_text = tfidf_vec.transform(text)

        m_feats = X_test.drop(columns='text')
        m_feats = pd.DataFrame(scaler.transform(m_feats), index=m_feats.index.values)

        tfidf_text = hstack((tfidf_text, csr_matrix(m_feats)))

        predicted = model.predict_proba(tfidf_text)[0]

        x = ["Negative class", "Positive class"]
        bar = alt.Chart(pd.DataFrame({"class": ["Negative", "Positive"], "value": predicted}), width=alt.Step(150))\
            .mark_bar()\
            .encode(
            x=alt.X("class"), y="value",
            color=alt.Color('class', scale=alt.Scale(domain=['Negative', 'Positive'], range=['red', 'green'])))
        bar_text = bar.mark_text(align='center', baseline='middle', dy=-5).encode(text='value')
        st.altair_chart(bar + bar_text)
