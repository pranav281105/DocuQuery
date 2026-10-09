"""
app.py — DocuQuery Streamlit application.

Wires together the RAG pipeline: upload → extract → chunk → embed → retrieve → generate.
"""

import streamlit as st

from rag.chunker import chunk_pages
from rag.embedder import embed_chunks, load_model, retrieve
from rag.extractor import extract_pages
from rag.generator import generate_answer

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="DocuQuery", page_icon="📄", layout="centered")
st.title("📄 DocuQuery")
st.caption("Upload a PDF and ask questions about its content.")

# ── Cache the embedding model so it loads only once ──────────────────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def get_model():
    return load_model()


model = get_model()

# ── Sidebar: PDF upload ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Upload a text-based PDF\n"
        "2. Ask a question below\n"
        "3. DocuQuery retrieves relevant passages and generates a grounded answer"
    )

# ── Process uploaded PDF ──────────────────────────────────────────────────────

# Use session state keyed on the file name so uploading a new PDF replaces the index.
if uploaded_file is not None:
    file_key = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_key") != file_key:
        # New file — reset all stored state.
        st.session_state.file_key = file_key
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.doc_name = uploaded_file.name

        with st.spinner("Extracting text and building index…"):
            try:
                pdf_bytes = uploaded_file.read()
                pages = extract_pages(pdf_bytes)
                chunks = chunk_pages(pages)
                embeddings = embed_chunks(model, chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings

                st.success(
                    f"✅ Indexed **{uploaded_file.name}** — "
                    f"{len(pages)} page(s), {len(chunks)} chunk(s)"
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"❌ {exc}")

# ── Question input and answer display ────────────────────────────────────────

chunks = st.session_state.get("chunks")
embeddings = st.session_state.get("embeddings")

if chunks is not None:
    st.divider()
    query = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. What are the main conclusions?",
    )

    if query.strip():
        with st.spinner("Retrieving relevant passages…"):
            top_chunks = retrieve(model, query, chunks, embeddings, top_k=5)

        with st.spinner("Generating answer…"):
            try:
                answer = generate_answer(query, top_chunks)
            except EnvironmentError as exc:
                st.error(f"❌ {exc}")
                st.stop()
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
                st.stop()

        st.subheader("Answer")
        st.markdown(answer)

        st.divider()
        st.subheader("Source Passages")
        for i, chunk in enumerate(top_chunks, start=1):
            with st.expander(f"Passage {i} — Page {chunk['page']}"):
                st.markdown(chunk["text"])

elif uploaded_file is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
