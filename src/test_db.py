from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into pages
loader = PyPDFLoader("test.pdf")
pages = loader.load_and_split()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size = 700, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


client = chromadb.HttpClient(host='localhost', port=8000,settings=Settings(allow_reset=True))
# client.reset()  # resets the database
collection = client.get_or_create_collection("my_collection1")

for doc in docs:
    collection.add(
        ids=[str(uuid.uuid1())],metadatas=doc.metadata, documents=doc.page_content
    )

db = Chroma(
    client=client,
    collection_name="my_collection1",
    embedding_function=embedding_function,
)
print('jdj')
print(len(db.get(include=[])['ids']))
query = "how much player get when the game started?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# try:
#     data = db.get(include=[])
#     print("Dokumen yang ada:", len(data['ids']))
# except Exception as e:
#     print("Terjadi kesalahan:", e)