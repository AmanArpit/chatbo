 import streamlit as st
import langchain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import LLMChain, PromptTemplate
from langchain.llms import ChatOpenAI
import pandas as pd
import os
import tempfile

# Customize Streamlit UI
st.set_page_config(page_title="MULTIPDF QA Chatbot", page_icon="🤖")
st.title("Welcome to MULTIPLE PDF QA RAG Chatbot 🤖")

# Function to configure retriever
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

    # Use a simple vector database for demonstration
    vectordb = Chroma.from_documents(doc_chunks)
    retriever = vectordb.as_retriever()
    return retriever

# Upload PDF files
uploaded_files = st.sidebar.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Configure retriever
retriever = configure_retriever(uploaded_files)

# Set up LLM
llm = ChatOpenAI(temperature=0.1, streaming=True)

# Define prompt template for QA
qa_template = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, 
don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
"""

# Create a chain for QA
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_template,
)

qa_chain = LLMChain(
    llm=llm,
    prompt=qa_prompt,
    input_keys=["question"],
    output_key="answer",
)

# Function to get sources
def get_sources(documents):
    sources = []
    for d in documents:
        metadata = {
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "content": d.page_content[:200],
        }
        sources.append(metadata)
    return sources

# User input for query
user_prompt = st.text_input("Enter your question")

if user_prompt:
    # Get context using retriever
    context_docs = retriever.query(user_prompt)
    context = format_docs(context_docs)

    # Run QA chain
    answer = qa_chain({"context": context, "question": user_prompt})

    # Display answer and sources
    st.write(f"Answer: {answer['answer']}")

    # Get and display sources
    sources = get_sources(context_docs)
    if sources:
        st.markdown("__Sources:__")
        st.dataframe(data=pd.DataFrame(sources[:3]), width=1000)

