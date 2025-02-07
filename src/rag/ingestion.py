from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain.schema.document import Document
import os
import tempfile
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
class DocumentIngestor:

    """
    Class to handle the document ingestion process. It loads, splits, clear, and embeds documents into the vector store
    """
    def __init__(self):
        # self.embedding = OllamaEmbeddings(model = "llama3.2:1b")
        # self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # self.embedding = OllamaEmbeddings(model = "nomic-embed-text")
        
        self.embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # self.vs = Chroma(persist_directory= PERSIST_DIRECTORY,embedding_function = self.embedding)

        client = chromadb.HttpClient(host="chroma", port=8000, settings=Settings(allow_reset=True))

        self.vs = Chroma(client=client, collection_name="my_collection", embedding_function = self.embedding )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 80, length_function = len,is_separator_regex=False )
        
    def load_documents(self, doc_path: str, doc_type: str):
        """
        Load documents from the specified DOCUMENT_PATH based on the document type.

        Args:
            doc_type (str): The type of document to load. Supported values are "pdf" and "md".

        Returns:
            A document loaded from the directory.
        """
        
        if doc_type == "md":
            loader = UnstructuredMarkdownLoader(doc_path)
            document = loader.load()
        else:
            loader = PyPDFLoader(file_path=doc_path)
            document = loader.load()
            
        return document

    def split_documents(self,documents : list[Document]):
        """Split the documents into chunks
        Args:
            documents (list[Document]): The list of documents to split.
        Returns: Chunks of the documents.
        """
        return self.text_splitter.split_documents(documents)
    
    def embed_to_vs(self,chunks : list[Document]):
        """
        Function to embed the new chunks to the vector store. The chunks that added to vector store ensured new.
        """
        chunks_with_ids = self.calculate_chunk_ids(chunks)
        existing_items = self.vs.get(include = [])
        existing_ids = set(existing_items['ids'])
        print(f"Number of existing documents in DB : {len(existing_ids)}")
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata['id'] not in existing_ids:
                new_chunks.append(chunk)
        if len(new_chunks) == 0:
            print("No new documents to add")
        else:
            new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
            self.vs.add_documents(new_chunks, ids = new_chunk_ids)
            # self.vs.persist()
            print(f"Added {len(new_chunks)} new documents to DB")

    def calculate_chunk_ids(self, chunks : list[Document]):
        """Calculate and add id metadata to each chunk based on the source and page number."""
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks
    def run_from_api(self, file_content: bytes, filename: str, extension: str):
    # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, filename)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_content)
            
            # Process the file.
            documents = self.load_documents(temp_file_path, extension)
            # Normalize the source in each document to the original filename
            for doc in documents:
                doc.metadata["source"] = filename
            chunks = self.split_documents(documents)
            self.embed_to_vs(chunks)

            
    def clear_database(self) -> int:
        """
        Clears all vectors/documents from the vector store.
        
        Returns:
            int: The number of vectors that were deleted.
            
        Raises:
            Exception: If there is an error during the deletion process.
        """
        try:
            # Retrieve all current documents from the vector store.
            data = self.vs.get(include=[])
            ids = data.get("ids", [])
            print("before clear:", len(ids))
            if ids:
                self.vs.delete(ids)
                num_deleted = len(ids)
                print(f"Deleted {num_deleted} vectors from the database.")
                return num_deleted
            else:
                print("No vectors found in the database to delete.")
                return 0
        except Exception as e:
            print(f"Error clearing the database: {e}")
            raise e
        
    # def delete_directory(self):
    #     if os.path.exists(PERSIST_DIRECTORY):
    #         shutil.rmtree(PERSIST_DIRECTORY)
    #         print(f"Deleted directory: {PERSIST_DIRECTORY}")

    def check(self):
        """Check the number of documents in the vector store."""
        print(len(self.vs.get()['documents']))

if __name__ == "__main__":
    di = DocumentIngestor()
    di.check()
    # try:
        # num_deleted = di.clear_database()
    #     print(f"Clear database successful: {num_deleted} vectors deleted.")
    # except Exception as err:
    #     print(f"Error: {err}")
    