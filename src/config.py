import os
from dotenv import load_dotenv

load_dotenv()

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
API_KEY = os.getenv("API_KEY")
OLLAMA_URL  = os.getenv("OLLAMA_URL")

print(API_KEY)

