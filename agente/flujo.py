
from langgraph.graph import StateGraph, END
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from rag import COLLECTION_NAME


# Define el estado de la conversación
class Estado(dict):
    pass

# Nodos
def preguntar_tramite(state: Estado):
    print("¿Qué trámite necesitas realizar?")
    tramite = input("> ").strip()
    return {"tramite": tramite}

def consultar_info(state: Estado):
    qa = cargar_rag()
    respuesta = qa.run(state["tramite"])
    print(f"\nInformación útil:\n{respuesta}")
    return {}

def fin(state: Estado):
    print("\nProceso terminado.")
    return {}

# Grafo
def crear_flujo():
    builder = StateGraph(Estado)
    builder.add_node("preguntar_tramite", preguntar_tramite)
    builder.add_node("consultar_info", consultar_info)
    builder.add_node("fin", fin)

    builder.set_entry_point("preguntar_tramite")
    builder.add_edge("preguntar_tramite", "consultar_info")
    builder.add_edge("consultar_info", "fin")
    builder.set_finish_point("fin")

    return builder.compile()

def cargar_rag():
    embeddings = OpenAIEmbeddings()
    client = QdrantClient(host="localhost", port=6333)
    
    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    
    modelo = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=modelo, retriever=vectorstore.as_retriever())
    return qa
