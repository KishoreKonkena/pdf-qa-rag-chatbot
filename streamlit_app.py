import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import tempfile

from pdf_loader import load_pdf_text
from chunker import chunk_text
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from rag_pipeline import RAGPipeline
from config import CHUNK_SIZE, CHUNK_OVERLAP

st.set_page_config(page_title="PDF QA Bot", layout="centered")
st.title("📄 PDF Question Answering Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF..."):
        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        embedder = EmbeddingModel()
        embeddings = embedder.encode(chunks)

        vector_store = FAISSVectorStore(len(embeddings[0]))
        vector_store.add(embeddings, chunks)

        rag = RAGPipeline(vector_store, embedder)

    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Generating answer..."):
            answer = rag.answer(question)
        st.markdown("### ✅ Answer")
        st.write(answer)
