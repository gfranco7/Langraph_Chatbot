from langgraph.graph import StateGraph, END
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from typing import TypedDict, Optional

from modelos.gemini import GeminiChat

COLLECTION_NAME = "tramites"

# Estado compartido con TypedDict (recomendado para LangGraph)
class Estado(TypedDict):
    tramite: Optional[str]
    paso_actual: Optional[str]
    informacion_recopilada: Optional[dict]
    conversacion_terminada: Optional[bool]


def crear_prompt_espanol():
    template = """
    Eres un asistente especializado en trámites gubernamentales en Colombia. 
    Tu trabajo es ayudar a los ciudadanos a completar sus trámites paso a paso.
    Ten presente que los usuarios se comunicaran contigo a través de telefonos
    celulares, entonces la información que brindes tiene que ser breve pero concisa.

    Usa el siguiente contexto para responder en español:
    {context}
    
    Pregunta: {question}
    
    Instrucciones:
    - Responde SIEMPRE en español
    - Sé claro y específico
    - Si el contexto no contiene información suficiente, menciona que necesitas más detalles
    - Guía al usuario paso a paso
    - Menciona documentos requeridos, costos, y lugares donde realizar el trámite
    
    Respuesta:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Configura RAG (embeddings + vectorstore + LLM)
def cargar_rag():
    print("Conectando con Qdrant...")
    client = QdrantClient(host="localhost", port=6333)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    modelo = GeminiChat()  # Usamos tu wrapper corregido
    qa = RetrievalQA.from_chain_type(
        llm=modelo, 
        retriever=vectorstore.as_retriever(),
        # Sin return_source_documents=True para usar .run()
        chain_type_kwargs={
            "prompt": crear_prompt_espanol()
        }
    )
    return qa

# Paso 1: Pregunta inicial
def preguntar_tramite(state: Estado):
    print("\n" + "="*60)
    print("ASISTENTE DE TRÁMITES GUBERNAMENTALES")
    print("="*60)
    print("¿Qué trámite necesitas realizar?")
    print("Ejemplo: 'Quiero sacar mi cédula de ciudadanía'")
    tramite = input("\n> ").strip()
    
    return {
        "tramite": tramite,
        "paso_actual": "consulta_inicial",
        "informacion_recopilada": {},
        "conversacion_terminada": False
    }

# Paso 2: Consulta inicial al RAG
def consultar_info_inicial(state: Estado):
    print(f"\nBuscando información sobre: {state['tramite']}")
    
    qa = cargar_rag()
    respuesta = qa.run(state["tramite"])
    
    print(f"\nInformación encontrada:")
    print("-" * 50)
    print(respuesta)
    print("-" * 50)
    
    return {
        **state,
        "paso_actual": "recopilar_datos"
    }

# Paso 3: Recopilar datos específicos del usuario
def recopilar_datos_usuario(state: Estado):
    print(f"\nAhora voy a recopilar algunos datos para ayudarte mejor con tu trámite.")
    
    datos = state.get("informacion_recopilada", {})
    
    # Preguntas comunes para la mayoría de trámites
    if "nombre" not in datos:
        nombre = input("\n¿Cuál es tu nombre completo? > ").strip()
        datos["nombre"] = nombre
    
    if "documento" not in datos:
        documento = input("¿Cuál es tu número de documento de identidad? > ").strip()
        datos["documento"] = documento
    
    if "ciudad" not in datos:
        ciudad = input("¿En qué ciudad te encuentras? > ").strip()
        datos["ciudad"] = ciudad
    
    print(f"\nPerfecto {datos['nombre']}, tengo tus datos básicos.")
    
    return {
        **state,
        "informacion_recopilada": datos,
        "paso_actual": "consulta_personalizada"
    }

# Paso 4: Consulta personalizada basada en los datos
def consulta_personalizada(state: Estado):
    datos = state["informacion_recopilada"]
    
    # Crear consulta personalizada
    consulta_personalizada = f"""
    {state['tramite']} en {datos['ciudad']}. 
    Necesito información específica sobre documentos requeridos, costos, 
    horarios de atención y ubicación exacta de las oficinas.
    """
    
    print(f"\nBuscando información específica para tu caso en {datos['ciudad']}...")
    
    qa = cargar_rag()
    respuesta = qa.run(consulta_personalizada)
    
    print(f"\nInformación personalizada:")
    print("-" * 50)
    print(respuesta)
    print("-" * 50)
    
    return {
        **state,
        "paso_actual": "seguimiento"
    }

# Paso 5: Seguimiento y preguntas adicionales
def seguimiento_tramite(state: Estado):
    while True:
        print(f"\n¿Tienes alguna pregunta adicional sobre tu trámite?")
        print("Puedes preguntar sobre:")
        print("- Documentos específicos que necesitas")
        print("- Costos exactos")
        print("- Horarios de atención")
        print("- Ubicación de oficinas")
        print("- Requisitos especiales")
        print("\nEscribe 'finalizar' para terminar.")
        
        pregunta = input("\n> ").strip()
        
        if pregunta.lower() in ['finalizar', 'terminar', 'salir', 'fin']:
            return {
                **state,
                "conversacion_terminada": True,
                "paso_actual": "finalizar"
            }
        
        if pregunta:
            print(f"\nConsultando: {pregunta}")
            
            # Crear consulta contextual
            consulta_contextual = f"""
            Sobre el trámite: {state['tramite']}
            Ciudad: {state['informacion_recopilada']['ciudad']}
            Pregunta específica: {pregunta}
            """
            
            qa = cargar_rag()
            respuesta = qa.run(consulta_contextual)
            
            print(f"\nRespuesta:")
            print("-" * 40)
            print(respuesta)
            print("-" * 40)

# Función para determinar el siguiente paso
def router(state: Estado):
    paso = state.get("paso_actual", "consulta_inicial")
    
    if paso == "consulta_inicial":
        return "consultar_info_inicial"
    elif paso == "recopilar_datos":
        return "recopilar_datos_usuario"
    elif paso == "consulta_personalizada":
        return "consulta_personalizada"
    elif paso == "seguimiento":
        return "seguimiento_tramite"
    else:
        return "fin"


# Paso final
def fin(state: Estado):
    datos = state.get("informacion_recopilada", {})
    nombre = datos.get("nombre", "")
    
    print(f"\n¡Perfecto {nombre}!")
    print("="*60)
    print("RESUMEN DE TU TRÁMITE")
    print("="*60)
    print(f"Trámite: {state['tramite']}")
    print(f"Ciudad: {datos.get('ciudad', 'N/A')}")
    print(f"Documento: {datos.get('documento', 'N/A')}")
    print("\nSi necesitas más ayuda, puedes ejecutar el programa nuevamente.")
    print("¡Que tengas un excelente día! 🌟")
    return state


# Construcción del flujo con LangGraph
def crear_flujo():
    builder = StateGraph(Estado)

    # Añadir todos los nodos
    builder.add_node("preguntar_tramite", preguntar_tramite)
    builder.add_node("consultar_info_inicial", consultar_info_inicial)
    builder.add_node("recopilar_datos_usuario", recopilar_datos_usuario)
    builder.add_node("consulta_personalizada", consulta_personalizada)
    builder.add_node("seguimiento_tramite", seguimiento_tramite)
    builder.add_node("fin", fin)

    # Configurar el flujo
    builder.set_entry_point("preguntar_tramite")
    
    # Usar el router para determinar el siguiente paso
    builder.add_conditional_edges(
        "preguntar_tramite",
        router,
        {
            "consultar_info_inicial": "consultar_info_inicial",
            "fin": "fin"
        }
    )
    
    builder.add_conditional_edges(
        "consultar_info_inicial",
        router,
        {
            "recopilar_datos_usuario": "recopilar_datos_usuario",
            "fin": "fin"
        }
    )
    
    builder.add_conditional_edges(
        "recopilar_datos_usuario",
        router,
        {
            "consulta_personalizada": "consulta_personalizada",
            "fin": "fin"
        }
    )
    
    builder.add_conditional_edges(
        "consulta_personalizada",
        router,
        {
            "seguimiento_tramite": "seguimiento_tramite",
            "fin": "fin"
        }
    )
    
    builder.add_conditional_edges(
        "seguimiento_tramite",
        router,
        {
            "seguimiento_tramite": "seguimiento_tramite",  # Permite loop
            "fin": "fin"
        }
    )
    
    builder.set_finish_point("fin")

    return builder.compile()