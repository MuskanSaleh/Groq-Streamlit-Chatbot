import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not openai_key or not groq_key:
    st.error("API keys not found! Please set OPENAI_API_KEY and GROQ_API_KEY.")
else:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["GROQ_API_KEY"] = groq_key

# Initialize embeddings and vector storage
if "vector" not in st.session_state:
    # Create embeddings
    st.session_state.embeddings = OpenAIEmbeddings()

    # Load webpage data
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Sindh")
    st.session_state.docs = st.session_state.loader.load()

    # Split text into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Store embeddings in FAISS vector database
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Initialize Streamlit UI
st.title("ChatGroq Demo 🤖")

# Load Language Model
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# Create a structured prompt
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    Provide the most accurate response possible.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create document retrieval chain
retriever = st.session_state.vector.as_retriever() if "vector" in st.session_state else None
if retriever:
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
else:
    st.error("Vector store is not initialized. Please refresh and try again.")

# User Input
prompt = st.text_input("💬 Ask me a question:")
    
if prompt and retriever:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    elapsed_time = time.process_time() - start

    st.write(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    st.write("### 🤖 Answer:")
    st.write(response['answer'])
    
    # Expand to show similar documents
    with st.expander("🔎 Relevant Documents Found:"):
        for i, doc in enumerate(response["context"]):
            st.write(f"📄 **Document {i+1}:**")
            st.write(doc.page_content)
            st.write("---")
