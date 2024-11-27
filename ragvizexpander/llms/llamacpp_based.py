from typing import Union, List, Dict
from .base import BaseChat


class LlamaCppClient:
    """LlamaCpp client factory"""
    def __init__(self, model_path=None):
        self.model_path = model_path
    
    def create_client(self):
        try:
            import llama_cpp
        except ImportError:
            raise ValueError(
                "The llama-cpp-python package is not installed. "
                "Please install it using `pip install llama-cpp-python`"
            )
        return llama_cpp.Llama(model_path=self.model_path)


class ChatLlamaCpp(BaseChat):
    """LlamaCpp chat model implementation"""
    def __init__(self, model_path=None):
        self.model_path = model_path
        super().__init__()
    
    def _initialize_client(self):
        client_factory = LlamaCppClient(self.model_path)
        return client_factory.create_client()
    
    def _create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self._client.create_chat_completion(
            messages=messages,
            **kwargs
        )
        return response['choices'][0]['text']

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:

        messages=[{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': prompt}]
        output = self._create_chat_completion(messages, **self.config.get_config())
        return self._format_output(output)
