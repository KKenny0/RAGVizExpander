from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
import json
from json_repair import repair_json


class LLMConfig:
    """Singleton configuration manager for LLM models"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.config = {}
        return cls._instance
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    @abstractmethod
    def create_client(self):
        """Create and return the specific LLM client"""
        pass


class BaseChat(ABC):
    """Template for chat implementations"""
    def __init__(self):
        self.config = LLMConfig()
        self._client = self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the specific LLM client"""
        pass
    
    @abstractmethod
    def _create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Create chat completion with the specific implementation"""
        pass
    
    def _format_output(self, output: str) -> Union[str, List[str]]:
        """Format the output based on configuration"""
        if "response_format" in self.config.get_config() and \
           self.config.get_config()['response_format'] == "json_object":
            json_output = repair_json(output)
            output = json.loads(json_output)
            output = list(output.values())
        return output
    
    def __call__(self, sys_msg: str, prompt: str) -> Union[str, List[str]]:
        """Template method defining the algorithm structure"""
        messages = [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': prompt}
        ]
        raw_output = self._create_chat_completion(messages, **self.config.get_config())
        return self._format_output(raw_output)
