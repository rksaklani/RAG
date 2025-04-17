from langchain_core.language_models import LLM
from typing import Optional, List
from pydantic import BaseModel, Field
import requests

class OllamaLLM(LLM, BaseModel):
    api_url: str = Field(...)
    model_name: str = Field(default="llama3.1:8b")
    system_prompt: str = Field(
        default="You are a persional assistant. Keep answers simple and concise."
    )

    @property
    def _llm_type(self) -> str:
        return "ollama-custom"

    def _construct_prompt(self, user_prompt: str) -> str:
        return f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>\n"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        final_prompt = self._construct_prompt(prompt)
        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers={"Content-Type": "application/json"}, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Request failed: {e}"
