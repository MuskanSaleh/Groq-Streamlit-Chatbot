import os
import streamlit as st 
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

#load the groq and google api key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key,model_name = "llama3-8b-8192")


print(llm)

# Create a structured prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    Provide the most accurate response possible.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("us_data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter  =RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_docs  = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)
        
prompt1 = st.text_input("What you want to ask from the documents?")

if st.button("creating vector store"):
    vector_embedding()
    st.write("vector store db is ready")
    
import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    
    
    #with a streamlit expander
    with  st.expander("Document Similarity Search"):
        #find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------")