#Combining Youtube assistant RAG along with PDF/HTML RAG.
import streamlit as st
import os
import time
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────
load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Sidebar — API key + mode selector ────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("AI Knowledge Assistant")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=get_secret("OPENAI_API_KEY"),
        type="password",
        help="Enter your OpenAI API key"
    )

    st.markdown("---")
    st.markdown("### 📂 Choose Input Mode")
    mode = st.radio(
        "What do you want to query?",
        ["📄 Document (PDF / TXT)", "🎥 YouTube Video"],
        help="Switch between document RAG and YouTube transcript QA"
    )

    st.markdown("---")
    st.markdown("**Built with:** LangChain · FAISS · OpenAI · Streamlit")

# ── Guard — no API key ────────────────────────────────────
if not api_key:
    st.warning("👈 Please enter your OpenAI API key in the sidebar.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ── Imports that need the key set first ───────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Shared prompt ─────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def build_qa_chain(vectorstore):
    """Shared QA chain used by both modes."""
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True
    )

# ═══════════════════════════════════════════════════════════
# MODE 1 — DOCUMENT RAG (from RAG-LangChain repo)
# ═══════════════════════════════════════════════════════════
if "📄 Document" in mode:
    st.title("📄 Document Q&A")
    st.caption("Upload a PDF or TXT file and ask questions about it.")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"],
        help="Supports PDF and plain text files"
    )

    if uploaded_file:
        @st.cache_resource(
            show_spinner="📚 Building vector index...",
            hash_funcs={"builtins.bytes": lambda x: hash(x)}
        )
        def build_doc_index(file_bytes, file_name, _api_key):
            import tempfile
            from langchain_community.document_loaders import PyPDFLoader, TextLoader

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_{file_name}"
            ) as f:
                f.write(file_bytes)
                tmp_path = f.name

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            return FAISS.from_documents(chunks, embeddings), len(chunks)

        vectorstore, num_chunks = build_doc_index(
            uploaded_file.read(),
            uploaded_file.name,
            api_key
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"✅ **{uploaded_file.name}** indexed successfully")
        with col2:
            st.metric("Chunks indexed", num_chunks)

        qa_chain = build_qa_chain(vectorstore)

        st.markdown("---")
        query = st.text_input(
            "💬 Ask a question about your document:",
            placeholder="e.g. What is the main conclusion?"
        )

        if query:
            with st.spinner("🔍 Retrieving and generating answer..."):
                t0 = time.time()
                result = qa_chain({"query": query})
                latency_ms = round((time.time() - t0) * 1000)

            st.markdown("### 💡 Answer")
            st.write(result["result"])

            col1, col2 = st.columns(2)
            col1.metric("⏱ Latency", f"{latency_ms}ms")
            col2.metric("📄 Sources used", len(result["source_documents"]))

            with st.expander("📎 View source chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.caption(doc.page_content[:400] + "...")
                    st.markdown("---")
    else:
        st.info("👆 Upload a document to get started.")

# ═══════════════════════════════════════════════════════════
# MODE 2 — YOUTUBE ASSISTANT (from Youtube-Assistant repo)
# ═══════════════════════════════════════════════════════════
elif "🎥 YouTube" in mode:
    st.title("🎥 YouTube Video Assistant")
    st.caption("Enter a YouTube URL and ask questions about the video content.")

    youtube_url = st.text_input(
        "🔗 YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if youtube_url:
        @st.cache_resource(show_spinner="📝 Loading transcript...")
        def build_youtube_index(url, _api_key):
            from langchain_community.document_loaders import YoutubeLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False
            )
            transcript_docs = loader.load()

            if not transcript_docs:
                return None, 0, "No transcript available for this video."

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(transcript_docs)
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            return vectorstore, len(chunks), None

        vectorstore, num_chunks, error = build_youtube_index(youtube_url, api_key)

        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try a video with English subtitles/captions enabled.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success("✅ Transcript loaded and indexed")
            with col2:
                st.metric("Transcript chunks", num_chunks)

            # Show video preview
            st.video(youtube_url)

            qa_chain = build_qa_chain(vectorstore)

            st.markdown("---")
            query = st.text_input(
                "💬 Ask a question about this video:",
                placeholder="e.g. What are the main points discussed?"
            )

            if query:
                with st.spinner("🔍 Searching transcript and generating answer..."):
                    t0 = time.time()
                    result = qa_chain({"query": query})
                    latency_ms = round((time.time() - t0) * 1000)

                st.markdown("### 💡 Answer")
                st.write(result["result"])

                col1, col2 = st.columns(2)
                col1.metric("⏱ Latency", f"{latency_ms}ms")
                col2.metric("📝 Transcript chunks used", len(result["source_documents"]))

                with st.expander("📎 View transcript excerpts used"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Excerpt {i+1}**")
                        st.caption(doc.page_content[:400] + "...")
                        st.markdown("---")
    else:
        st.info("👆 Paste a YouTube URL to get started.")