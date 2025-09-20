import streamlit as st
import os
from rag_backend import create_vector_store, create_rag_chain, user_input

st.set_page_config(page_title="Journey to the Centre of the Earth Chatbot", page_icon=":books:")
st.header("Journey to the Centre of the Earth Chatbot :books:")
st.write("Ask me anything about the book!")


@st.cache_resource
def get_resources():
    book_path = "book.txt"
    if not os.path.exists(book_path):
        st.error(f"Book file not found. Please place 'journey_to_the_center.txt' in the 'books' folder.")
        st.stop()

    vector_store = create_vector_store(book_path)
    rag_chain = create_rag_chain(vector_store)
    return rag_chain


try:
    rag_chain = get_resources()
except Exception as e:
    st.error(f"Failed to initialize the chatbot. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about the book..."):

    st.session_state.messages.append({"role": "user", "content": prompt})


    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = user_input(prompt, rag_chain)
        st.markdown(response)


    st.session_state.messages.append({"role": "assistant", "content": response})