import streamlit as st

def show_navigation():
    with st.container(border=True):
        col1,col2,col3,col4,col5=st.columns(5)
        col1.page_link("Home.py", label="Home", icon="🏠")
        col2.page_link("pages/0_upload_pdf.py", label="Upload PDF", icon="1️⃣")
        col3.page_link("pages/1_chat_with_AI.py", label="Chat", icon="2️⃣")
        col4.page_link("pages/2_retreival_augmented_chat.py", label="RAG", icon="🌎")
        col5.page_link("pages/3_unstructured.py", label="Unstructured")