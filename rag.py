# rag.py
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION_NAME = "tramites"

def crear_vectorstore_qdrant(ruta_pdf: str):
    loader = PyPDFLoader(ruta_pdf)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    client = QdrantClient(host="localhost", port=6333)
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    print("Qdrant cargado con los documentos.")
    return vectorstore
