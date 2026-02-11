# config.py

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OR if you use OpenAI embeddings
#OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

OPENAI_CHAT_MODEL = "gpt-4o-mini"   # or gpt-3.5-turbo
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
