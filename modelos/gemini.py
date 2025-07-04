import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

load_dotenv()

class GeminiChat(LLM):
    model_name: str = "gemini-1.5-flash"
    
    def __init__(self, model_name="gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)  
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(model_name)

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        print(f"\n💬 [Prompt enviado a Gemini]:\n{prompt}\n")
        response = self._model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    @property
    def _identifying_params(self) -> dict:
        """Parámetros identificativos del modelo"""
        return {"model_name": self.model_name}