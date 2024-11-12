"""
query_expansion.py

This module provides functionalities for expanding queries using GPT-4 powered chat completions.
It includes generating sub-questions and hypothetical answers for a given query.
"""

import os
import json
from typing import List, Union

from openai import OpenAI

from .constants import MULTIPLE_QNS_SYS_MSG, HYDE_SYS_MSG


def generate_sub_qn(query: str, client) -> List[str]:
    """
    Generates sub-questions for a given query using the GPT-4 model.

    Args:
        query (str): The original query for which sub-questions need to be generated.
        client

    Returns:
        List[str]: A list of generated sub-questions.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
    try:
        client.config.update({"response_format": "json_object"})
        sub_qns = _chat_completion(MULTIPLE_QNS_SYS_MSG,
                                   query,
                                   client)
    except Exception as e:
        raise RuntimeError(f"Error in generating sub-questions: {e}") from e
    return sub_qns


def generate_hypothetical_ans(query: str, client) -> str:
    """
    Generates a hypothetical answer for a given query using the GPT-4 model.

    Args:
        query (str): The original query for which a hypothetical answer is needed.
        client

    Returns:
        str: The generated hypothetical answer.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
    try:
        hyp_ans = _chat_completion(HYDE_SYS_MSG, query, client)
    except Exception as e:
        raise RuntimeError(f"Error in generating hypothetical answer: {e}") from e
    return hyp_ans


def _chat_completion(sys_msg: str, prompt: str, client) -> Union[str, List[str]]:
    """
    A helper function to perform chat completions using the OpenAI API.

    Args:
        sys_msg (str): The system message for setting up the context.
        prompt (str): The user prompt for generating the completion.

    Returns:
        Union[str, List[str]]: The response from the chat completion, either as text or a list.

    Raises:
        OpenAIError: If an error occurs in the OpenAI API call.
    """
    output = client(sys_msg, prompt)

    return output
