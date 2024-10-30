from ast import Param
from typing import (
    Optional,
    Union,
    List,
)
import json
from json_repair import repair_json


class ChatOllama:
    """Ollama chat model"""
    def __init__(self,
                 model_name=None,
                 host="http://localhost:11434/api/generate"
                 ):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. Please install it with `pip install ollama`"
            )

        if host is not None:
            self.host = host

        self.model_name = model_name

        self._client = ollama.Client(host=self.host)

    def __call__(self, sys_msg: str, prompt: str, response_format: str) -> Union[str, List[str]]:
        response = self._client.chat(
            model=self.model_name,
            messages=[{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}],
        )

        output = response["message"]["content"]

        if response_format == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())

        return output
