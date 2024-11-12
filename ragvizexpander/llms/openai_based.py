from typing import (
    Union,
    List,
)
import json
from json_repair import repair_json


class ChatOpenAI:
    """OpenAI chat model"""
    def __init__(self, config=None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. Please install it with `pip install openai`"
            )

        if config:
            self.base_url = config.get("base_url", "")
            self.api_key = config.get("api_key", "")
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url
            )
        self.config = config

    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        response = self._client.chat.completions.create(
            messages=[{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}],
            **self.config
        )

        output = response.choices[0].message.content

        if "response_format" in self.config and self.config["response_format"] == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())

        return output
