from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import tiktoken
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
def robust_count_tokens(text: str, model: str = "llama3.2:1b"):
    """
    Count tokens in a given text using tiktoken.
    Tries to use the model-specific encoding and falls back to a default encoding if necessary.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The name of the model to determine the encoding.
        
    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Try to get encoding for the specified model.
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to a default encoding if the model-specific encoding is unavailable.
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class Retrivial:
    """
    Class that handle the retrivial, augmentation, and generation process for the question answering task
    """
    def __init__(self,query):
        self.PROMPT_TEMPLATE = """
You are a technical documentation assistant.
Your task is to answer the following question using only the information provided in the context.
Ensure that your answer is concise, accurate, and directly supported by the context.
Whenever you reference supporting information, include only its citation in the format [Source ID: n] (do not include any excerpts or snippet text from the context).
If the context does not contain enough information to answer the question, state that explicitly.

Context:
{context}

Question: {question}

Answer (with citations): 
"""
        self.query = "what is the objective of the game?" if query is None else query 
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        client = chromadb.HttpClient(host="chroma", port=8000, settings=Settings(allow_reset=True))

        self.db = Chroma(client=client, collection_name="my_collection", embedding_function = self.embedding )
    def find_relevant_docs(self):
        """ Find the relevant documents based on the query"""
        relevant_docs = self.db.similarity_search_with_score(self.query,k= 5)
        return relevant_docs
    def format_docs_with_id(self,docs) :
        """
        Format the documents with the source id
        Args: docs : list[Document]
        Returns: prompt_formated : str
        """
        formatted = [
            f"Source ID: {i+1}\nArticle Title: {doc[0].metadata['source']}\nArticle Snippet: {doc[0].page_content}"
            for i, doc in enumerate(docs)
        ]
        return "\n\n" + "\n\n".join(formatted)
    def retrieve(self):
        """
        Retrieve the relevant documents based on the query and format the prompt
        Returns: prompt_formated : str
        """
        relevant_docs = self.find_relevant_docs()
        # for docs,score in relevant_docs:
        #     print(docs.metadata)
        #     print(score)
        #     print(docs.page_content)
        formatted_context = self.format_docs_with_id(relevant_docs)
        prompt_template = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt_formated = prompt_template.format(context = formatted_context,question  = self.query)
       
        return prompt_formated,formatted_context
    
    def generate_response(self,prompt_formated : str):
        """
        generate response based on the prompt
        Args: prompt_formated : str
        Returns: response_text : str

        """
        model = OllamaLLM(
            base_url="http://ollama:11434/",
    model="llama3.2:1b",
    temperature=0.1,
)
        response_text =model.invoke(prompt_formated)
        return response_text
    
    def run(self):
        """
        Run the retrivial and generation process. Start from retrivial relevant data, augmentating the prompt, and generate the response
        Returns: response_text : str, token_len : int
        """
        prompt_formated,formated_context = self.retrieve()
        response_text = self.generate_response(prompt_formated)
        print(response_text)
        token_len = robust_count_tokens(response_text)
        return response_text,formated_context,token_len
    
    
if __name__ == "__main__":    
    retriever = Retrivial()
    retriever.run()
    
    



