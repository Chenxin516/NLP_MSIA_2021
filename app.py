import streamlit as st

st.title("Toxic Comment Classifier")

text_input = st.text_input(label="Input Text")

if st.button(label="Make and Explain Prediction"):
    st.write("This function is still being developed.")
