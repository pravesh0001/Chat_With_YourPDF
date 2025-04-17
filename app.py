
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from PyPDF2 import PdfReader
except ImportError:
    install("PyPDF2")
    from PyPDF2 import PdfReader

try:
    import streamlit as st
except ImportError:
    install("streamlit")
    import streamlit as st

try:
    import nltk
except ImportError:
    install("nltk")
    import nltk

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    install("sentence-transformers")
    from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    install("faiss-cpu")
    import faiss

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np


nltk.download("punkt")
from nltk.tokenize import sent_tokenize


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def embed_chunks(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings)


def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


st.set_page_config(page_title="Chat With Your PDF", layout="centered")
st.title("📄 Chat With Your PDF")
st.markdown("Upload a PDF and ask questions based on its content.")


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("⏳ Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)

        if text:
            chunks = chunk_text(text)
            embedding_model = load_embedding_model()
            chunk_embeddings = embed_chunks(chunks, embedding_model)
            index = create_faiss_index(chunk_embeddings)

            st.success("✅ PDF processed successfully!")

           
            query = st.text_input("💬 Ask a question based on the PDF:")
            if query:
                query_embedding = embedding_model.encode([query])
                D, I = index.search(np.array(query_embedding), k=3)
                top_chunks = [chunks[i] for i in I[0]]

                st.write("🔍 Top relevant chunks:")
                for i, ch in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}:** {ch}")
        else:
            st.error("❌ Could not extract text from the PDF. Try a different file.")
