import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION_NAME = "tramites"

def crear_vectorstore_qdrant(ruta_pdf: str):
    if not os.path.exists(ruta_pdf):
        raise FileNotFoundError(f" El archivo no existe: {ruta_pdf}")

    print("Cargando documento...")
    loader = PyPDFLoader(ruta_pdf)
    docs = loader.load()

    print("Generando embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Conectando con Qdrant...")
    client = QdrantClient(host="localhost", port=6333)

    # Crear la colección
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Crear el vectorstore
    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    # Añadir documentos
    vectorstore.add_documents(docs)

    print("Base vectorial cargada correctamente.")
