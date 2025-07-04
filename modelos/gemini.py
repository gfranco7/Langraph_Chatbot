import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

load_dotenv()

class GeminiChat(LLM): #Se hereda la clase base LLM 
    model_name: str = "gemini-1.5-flash"


    #Método constructor
    def __init__(self, model_name="gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)  #Llama al constructor y le pasa mas argumentos
        self.model_name = model_name #Guarda
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))# Configura
        self._model = genai.GenerativeModel(model_name) # Crea

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    @property
    def _identifying_params(self) -> dict:
        """Parámetros identificativos del modelo"""
        return {"model_name": self.model_name}
    