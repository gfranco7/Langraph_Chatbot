from agente.flujo import crear_flujo
from rag import crear_vectorstore_qdrant
import os

if __name__ == "__main__":
    ruta_pdf = "docs/pdf_test.pdf"
    
    if os.path.exists(ruta_pdf):
        print("Cargando documentos a la base vectorial...")
        crear_vectorstore_qdrant(ruta_pdf)
    else:
        print("No se encontró el archivo PDF. Asegúrate de que existe:", ruta_pdf)
    
    print("\n🤖 Iniciando el asistente de trámites...")
    flujo = crear_flujo()
    
    estado_inicial = {
        "tramite": None,
        "paso_actual": None,
        "informacion_recopilada": {},
        "conversacion_terminada": False
    }
    
    resultado = flujo.invoke(estado_inicial)
    print(f"\n🎯 Sesión finalizada exitosamente.")