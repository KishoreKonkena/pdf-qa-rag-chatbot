from pdf_loader import load_pdf_text
from chunker import chunk_text
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from rag_pipeline import RAGPipeline
from config import CHUNK_SIZE, CHUNK_OVERLAP

PDF_PATH = "Copy of Konkena_Kishore_resume.pdf"

def main():
    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Generating embeddings...")
    embedder = EmbeddingModel()
    embeddings = embedder.encode(chunks)

    print("Building FAISS index...")
    vector_store = FAISSVectorStore(embedding_dim=len(embeddings[0]))
    vector_store.add(embeddings, chunks)

    rag = RAGPipeline(vector_store, embedder)

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        answer = rag.answer(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
