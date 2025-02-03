from config import EMBEDDING_MODEL

from langchain_huggingface import HuggingFaceEmbeddings
def get_embedding_model():
    """ Get the embedding model based on the environment variable"""
    embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    return embedding_model
