from typing import Union, List, Dict
from .base import BaseChat, BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Ollama client factory"""
    def __init__(self, host: str = None):
        self.host = host if host else "http://localhost:11434"

    def create_client(self):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`"
            )
        return ollama.Client(host=self.host)


class ChatOllama(BaseChat):
    """Ollama chat model implementation"""
    def __init__(self, host: str = None, model: str = None):
        self.host = host
        self.model = model
        super().__init__()

    def _initialize_client(self):
        client_factory = OllamaClient(self.host)
        return client_factory.create_client()

    def _create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        options = kwargs.pop('options', {})
        response = self._client.chat(
            messages=messages,
            model=self.model,
            options=options
        )
        output = response["message"]["content"]
        return output

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        messages = [{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': prompt}]
        output = self._create_chat_completion(messages, options=self.config.get_config())
        return self._format_output(output)
