import fasttext
import joblib
import numpy as np
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer


st.title("Toxic Comment Classifier")

# Load Model
@st.cache(
    hash_funcs={
        transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast: hash
    },
    allow_output_mutation=True,
)
def load_model():
    preprocessor_logit = joblib.load("models/bow/vectorizer.joblib")
    model_logit = joblib.load("models/bow/logistic.joblib")

    model_fasttext = fasttext.load_model("models/ft.bin")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model_bert = AutoModelForSequenceClassification.from_pretrained(
        "models/transformer_models/checkpoint-27075"
    )

    return preprocessor_logit, model_logit, model_fasttext, tokenizer, model_bert


preprocessor_logit, model_logit, model_fasttext, tokenizer, model_bert = load_model()


def predictor(texts):
    outputs = model_bert(**tokenizer(texts, return_tensors="pt", padding=True))
    tensor_logits = outputs[0]
    probas = F.softmax(tensor_logits, dim=1).detach().numpy()
    return probas


model_option = st.selectbox(
    "Which model do you want to use?",
    ("Logistic Regression", "FastText", "DistilBERT"),
)

text_input = st.text_input(label="Input Text")
result_dict = {0: "non-toxic", 1: "toxic"}

if model_option == "Logistic Regression":
    processed_input = preprocessor_logit.transform([text_input])
    pred = model_logit.predict(processed_input).item()
    pred_prob = model_logit.predict_proba(processed_input).reshape(-1)
elif model_option == "FastText":
    preds = model_fasttext.predict(text_input)
    if preds[0][0] == "__label__1":
        pred = 1
        pred_prob = [1 - preds[1][0], preds[1][0]]
    else:
        pred = 0
        pred_prob = [preds[1][0], 1 - preds[1][0]]
else:
    pred_prob = predictor(text_input)[0]
    pred = np.argmax(pred_prob)
    explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])
    exp = explainer.explain_instance(
        text_input, predictor, num_features=20, num_samples=200
    )


if st.button(label="Make Prediction"):
    df_pred = pd.DataFrame({"name": ["non-toxic", "toxic"], "pred_prob": pred_prob})
    fig = px.bar(df_pred, x="pred_prob", y="name", text="pred_prob", orientation="h")
    fig.update_layout(
        autosize=False,
        width=700,
        height=150,
        margin=dict(l=0, r=0, b=0, t=0),
        yaxis={"title": ""},
        xaxis={"visible": False},
    )
    fig.update_traces(width=0.2, texttemplate="%{text:.3f}", textposition="outside")
    st.write(f"Prediction: {result_dict[pred]}")
    if model_option == "DistilBERT":
        components.html(exp.as_html(), height=500, scrolling=True)
    else:
        st.plotly_chart(fig)
