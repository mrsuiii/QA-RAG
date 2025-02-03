import os
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH","data")
OLLAMA_URL = os.getenv("OLLAMA_URL","http://ollama:11411")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY","embeddings_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
API_KEY = os.getenv("API_KEY","secret")
