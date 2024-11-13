from typing import (
    Union,
    List,
)
import json
from json_repair import repair_json


class ChatOllama:
    """Ollama chat model"""
    def __init__(self, host=None, model=None):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`"
            )
        if not host:
            self.host = "http://localhost:11434"
        else:
            self.host = host
        self._client = ollama.Client(host=self.host)
        self.model = model
        self.config = {}

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        response = self._client.chat(
            messages=[{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}],
            model=self.model,
            options=self.config
        )

        output = response["message"]["content"]

        if "response_format" in self.config and self.config["response_format"] == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())

        return output
