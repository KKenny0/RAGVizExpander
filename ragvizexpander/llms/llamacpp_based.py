import json
from typing import Union, List
from json_repair import repair_json


class ChatLlamaCpp:
    """LlamaCpp chat model"""
    def __init__(self, model_path=None):
        try:
            import llama_cpp
        except ImportError:
            raise ValueError(
                "The llama-cpp-python package is not installed. "
                "Please install it using `pip install llama-cpp-python`"
            )
        if model_path:
            self._client = llama_cpp.Llama(model_path=model_path)
        self.config = {}

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        response = self._client.create_chat_completion(
            messages=[{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}],
            **self.config
        )
        output = response['choices'][0]['text']

        if "response_format" in self.config and self.config['response_format'] == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())

        return output
