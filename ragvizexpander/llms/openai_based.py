from typing import (
    Union,
    List,
)
import json
from json_repair import repair_json


class ChatOpenAI:
    """OpenAI chat model"""
    def __init__(self,
                 base_url=None, api_key=None, model_name=None
                 ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ValueError(
                "The openai python package is not installed. Please install it with `pip install openai`"
            )

        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        self._client = OpenAI(
            api_key=self.api_key, base_url=self.base_url
        )

    def __call__(self, sys_msg: str, prompt: str, response_format: str) -> Union[str, List[str]]:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}],
            temperature=0,
            response_format={'type': response_format},
            seed=0
        )

        output = response.choices[0].message.content

        if response_format == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())

        return output
