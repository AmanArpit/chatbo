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

from langchain_google_genai import ChatGoogleGenerativeAI

# Assuming you have a Google API key
google_api_key = "AIzaSyCPC7p0EaOzo72MKbVubYt_dsBGAqv5QCc"

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.1,
    streaming=True
)

# Customize Streamlit UI
st.set_page_config(page_title="MULTIPDF QA Chatbot", page_icon="🤖")
st.title("Welcome to MULTIPLE PDF QA RAG Chatbot 🤖")

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
    
    # Use Google Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)
    
    retriever = vectordb.as_retriever()
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown("__Sources:__ "+"\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]), width=1000)

uploaded_files = st.sidebar.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# QA Template
qa_template = """
    Use only the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know,
    don't try to make up an answer. Keep the answer as concise as possible.
    {context}
    Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Combining retriever and model in a pipeline
qa_chain = LLMChain(llm=gemini, prompt=qa_prompt)

# Chat history
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
