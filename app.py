import streamlit as st
import os
from tempfile import TemporaryDirectory

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="PDF RAG with Groq", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot (Groq + Llama 3.3)")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your key from https://console.groq.com"
    )
    
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200)
    
    if st.button("Clear Vector Database"):
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
            del st.session_state.retriever
            st.success("Vector database cleared!")
            st.rerun()

# ====================== MAIN APP ======================
if not groq_api_key:
    st.warning("👆 Please enter your Groq API Key in the sidebar")
    st.stop()

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.retriever = None

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    help="You can upload multiple PDFs"
)

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs and building vector database..."):
            try:
                all_docs = []
                
                # Save uploaded files temporarily and load
                for uploaded_file in uploaded_files:
                    with TemporaryDirectory() as temp_dir:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        loader = PyPDFLoader(temp_path)
                        all_docs.extend(loader.load())
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                texts = text_splitter.split_documents(all_docs)
                
                st.success(f"Created {len(texts)} chunks from {len(uploaded_files)} PDF(s)")
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=texts, 
                    embedding=embeddings
                )
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )
                
                st.success("✅ Vector Database Ready!")
                
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

# ====================== CHAT INTERFACE ======================
if st.session_state.vectorstore is None:
    st.info("Please upload and process PDFs first")
else:
    st.success("✅ RAG System is Ready! Ask questions about your documents.")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model=model_name,
                        temperature=temperature,
                        max_tokens=1024
                    )
                    
                    prompt_template = ChatPromptTemplate.from_template(
                        """You are a helpful assistant. Answer the question using only the provided context.
If the answer cannot be found in the context, say "I don't have enough information from the documents."

Context:
{context}

Question: {input}
Answer:"""
                    )
                    
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
                    
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Show sources (collapsible)
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response.get("context", []), 1):
                            st.markdown(f"**Source {i}**")
                            st.caption(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.caption("Built with LangChain + Groq + Streamlit")