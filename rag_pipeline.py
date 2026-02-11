import os
from openai import OpenAI
from dotenv import load_dotenv
from config import OPENAI_CHAT_MODEL, TOP_K

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGPipeline:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def answer(self, question: str) -> str:
        # Embed query
        query_embedding = self.embedding_model.encode([question])[0]

        # Retrieve relevant chunks
        context_chunks = self.vector_store.search(query_embedding, TOP_K)
        context = "\n".join(context_chunks)

        # RAG prompt
        prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()
