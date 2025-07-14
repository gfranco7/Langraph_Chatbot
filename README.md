# Langraph_Chatbot

# Asistente de Trámites Gubernamentales con LangGraph + RAG

Este proyecto implementa un agente conversacional que guía al usuario paso a paso en la realización de **trámites gubernamentales en Colombia**, utilizando tecnologías de **LangGraph**, **RAG (Retrieval-Augmented Generation)**, y **Qdrant** para recuperación de información precisa desde archivos PDF legales o administrativos.

---

## 🧠 Tecnologías Clave

- **LangGraph** – Para definir el flujo del agente conversacional como un grafo de estados.
- **LangChain** – Orquestador de cadenas de RAG y manejo de prompts.
- **Qdrant** – Motor de búsqueda vectorial para almacenar y consultar embeddings.
- **HuggingFace Embeddings** – Para vectorizar los contenidos de los documentos PDF.
- **PDF Parsing** – Extracción y segmentación de texto desde archivos PDF cargados.

---

## 📁 Estructura del Proyecto

- `agente/` – Contiene la lógica del agente conversacional y definición del flujo LangGraph.
- `data/` – Carpeta para almacenamiento temporal o procesamiento de datos.
- `docs/` – Contiene los archivos PDF con información sobre los trámites.
- `modelos/` – Modelos de datos o clases estructuradas utilizadas por el sistema.
- `main.py` – Punto de entrada del sistema para ejecutar el asistente.
- `rag.py` – Funciones para cargar, dividir y vectorizar PDFs usando Qdrant.
- `docker-compose.yml` – Orquestación de servicios (como Qdrant) en contenedores Docker.
- `requirements.txt` – Lista de dependencias necesarias para el proyecto.
- `.gitignore` – Archivos y carpetas ignorados por Git.
- `README.md` – Documentación general del proyecto.
- `__pycache__/` – Carpeta generada automáticamente por Python (código compilado).

---

## 🔄 Flujo del Agente

1. **Inicio**: El usuario escribe una duda sobre un trámite (ej: "¿Cómo saco el RUT?").
2. **Validación**: El agente identifica si la pregunta puede resolverse con los documentos cargados.
3. **Recuperación**: Se hace una búsqueda semántica en el PDF vectorizado con Qdrant.
4. **Respuesta**: El agente responde con información precisa y contextualizada.
5. **Guía**: Si aplica, se orienta al usuario paso a paso con instrucciones claras.

---

## 🧪 Ejemplo de Uso

```python
from agente.flujo import crear_flujo
from rag import crear_vectorstore_qdrant

# Crear el vectorstore a partir del PDF cargado
crear_vectorstore_qdrant("docs/pdf_test.pdf")

# Ejecutar el asistente
flujo = crear_flujo()
flujo.invoke({})
```

---

## ⚙️ Requisitos

- Python 3.10+
- langgraph
- langchain
- qdrant-client
- huggingface-hub
- pypdf
- tiktoken

Instala todas las dependencias con:

```bash
pip install -r requirements.txt
```

---

## 📌 Aplicaciones Potenciales

- Orientación ciudadana en línea
- Portales institucionales de atención
- Asistentes de alcaldías o entidades públicas

---

## ✍️ Autor

**Gean Franco Jacome Laguna**

---

## 📜 Licencia

Proyecto educativo para prácticas de agentes LLM y recuperación aumentada de información. Verifica los derechos sobre los documentos usados como fuente.
