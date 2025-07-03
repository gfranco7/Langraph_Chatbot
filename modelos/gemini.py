from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


class GeminiChat(BaseChatModel):
    def __init__(self, model_name="gemini-1.5-flash", temperature=0.2):
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        self.temperature = temperature

    def _generate(self, messages, stop=None, run_manager=None) -> ChatResult:
        history = []
        for m in messages:
            if isinstance(m, HumanMessage):
                history.append({"role": "user", "parts": [m.content]})
            elif isinstance(m, AIMessage):
                history.append({"role": "model", "parts": [m.content]})

        prompt = messages[-1].content
        response = self.chat.send_message(prompt)
        text = response.text

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"
