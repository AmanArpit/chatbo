import yaml
import os
import streamlit as st
import tempfile
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter

# Load API key from YAML file
with open('gemini_api_credentials.yml', 'r') as file:
    api_creds = yaml.safe_load(file)
gemini_api_key = api_creds.get("Gemini_key")

# Set the environment variable for the API key
os.environ['GOOGLE_API_KEY'] = gemini_api_key

# Initialize the Gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    streaming=True,
    api_key=gemini_api_key  # Using the loaded API key
)

# Streamlit UI customization
st.set_page_config(page_title="MULTIPDF QA Chatbot", page_icon="🤖")
st.title("Welcome to MULTIPLE PDF QA RAG Chatbot 🤖")

# File uploader and document processing
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

    retriever = vectordb.as_retriever()
    return retriever

# Initialize retriever
uploaded_files = st.sidebar.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# QA Chain Setup (Fixing pipeline chain here)
qa_template = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know,
don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

# Streamlit message history
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question?")

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_container = st.write("")
        pm_handler = PostMessageHandler(sources_container)
        config = {"callbacks": [stream_handler, pm_handler]}
        response = qa_rag_chain.invoke({"question": user_prompt}, config)
        st.write(response.content)
