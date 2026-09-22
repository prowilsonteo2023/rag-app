import os
import tempfile
import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    UnstructuredImageLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE SETUP ---
st.set_page_config(page_title="Multi-Format OCR RAG App", layout="wide")

st.title("📂 Multi-Format Document RAG Dashboard")
st.markdown("Upload files (PDF, DOCX, XLSX, HTML, Images), run OCR, and query your data using OpenAI or Gemini.")

# Persistent storage folder for the Vector DB on your PC
PERSIST_DIR = "./chroma_db"

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Configuration")
provider = st.sidebar.selectbox("Choose LLM Provider", ["OpenAI", "Gemini"])

api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")

# Set up appropriate embeddings and models based on selection
if provider == "OpenAI":
    os.environ["OPENAI_API_KEY"] = api_key
    llm_model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    embeddings = OpenAIEmbeddings()
else:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm_model = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- AUTO-LOAD PERSISTENT VECTOR STORE ---
if "vector_store" not in st.session_state:
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            st.session_state.vector_store = Chroma(
                persist_directory=PERSIST_DIR, 
                embedding_function=embeddings
            )
        except Exception:
            st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

# --- DOCUMENT UPLOAD DASHBOARD ---
st.sidebar.header("2. Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload files", 
    type=["pdf", "docx", "xlsx", "html", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents") and api_key and uploaded_files:
    with st.spinner("Processing files, running OCR where needed, and updating Vector DB..."):
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if uploaded_file.type in ["image/png", "image/jpeg"]:
                    loader = UnstructuredImageLoader(tmp_path)
                else:
                    loader = UnstructuredFileLoader(tmp_path)
                
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
            finally:
                os.remove(tmp_path)
        
        if docs:
            split_docs = text_splitter.split_documents(docs)
            
            st.session_state.vector_store = Chroma.from_documents(
                documents=split_docs, 
                embedding=embeddings, 
                persist_directory=PERSIST_DIR
            )
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} files and saved to disk!")

# --- SEARCH & CHAT INTERFACE ---
st.header("💬 Search Knowledge Base")

user_query = st.text_input("Ask a question about your uploaded documents:")

if user_query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first using the sidebar.")
    elif not api_key:
        st.warning(f"Please enter your {provider} API key in the sidebar.")
    else:
        with st.spinner("Searching Vector DB and generating answer..."):
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
            
            if provider == "OpenAI":
                llm = ChatOpenAI(model=llm_model, temperature=0.2)
            else:
                llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.2)
                
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know.\n\n"
                "{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            
            st.subheader("Answer:")
            st.write(response["answer"])
            
            with st.expander("View Source Context Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**")
                    st.text(doc.page_content)