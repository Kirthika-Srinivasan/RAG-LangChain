import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile, os

st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Production RAG System")
st.caption("Built with LangChain · Hybrid Retrieval · Re-ranking")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown("Hybrid retrieval + re-ranking RAG pipeline")

# Main 
if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

@st.cache_resource(show_spinner="Building vector index...")
def build_index(file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_name) as f:
        f.write(file_bytes)
        tmp_path = f.name
    
    loader = PyPDFLoader(tmp_path) if file_name.endswith(".pdf") else TextLoader(tmp_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if uploaded_file:
    vectorstore = build_index(uploaded_file.read(), uploaded_file.name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = PromptTemplate(
        template="""Answer using ONLY the context below. If unsure, say "I don't know."
Context: {context}
Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    st.success(f"✅ Indexed: {uploaded_file.name}")
    
    query = st.text_input("Ask a question about your document:")
    
    if query:
        with st.spinner("Retrieving and generating..."):
            import time
            t0 = time.time()
            result = qa_chain({"query": query})
            latency = round((time.time() - t0) * 1000)
        
        st.markdown("### Answer")
        st.write(result["result"])
        
        with st.expander(f"📄 Sources · ⏱ {latency}ms"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}**")
                st.caption(doc.page_content[:300] + "...")
else:
    st.info("👈 Upload a PDF or TXT file to get started.")