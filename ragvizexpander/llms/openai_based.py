from typing import Union, List, Dict
from .base import BaseChat, BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI client factory"""
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url

    def create_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. "
                "Please install it with `pip install openai`"
            )
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


class ChatOpenAI(BaseChat):
    """OpenAI chat model implementation"""
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        super().__init__()

    def _initialize_client(self):
        client_factory = OpenAIClient(self.api_key, self.base_url)
        return client_factory.create_client()

    def _create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            **kwargs
        )
        return response.choices[0].message.content

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        messages = [{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': prompt}]
        output = self._create_chat_completion(messages, **self.config.get_config())
        return self._format_output(output)
