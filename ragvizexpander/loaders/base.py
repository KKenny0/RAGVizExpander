import io

from abc import ABC, abstractmethod
from typing import List, Type, Dict, Union
from pathlib import Path


class LoaderStrategy(ABC):
    """Abstract base class for different loading strategies"""
    
    @abstractmethod
    def load(self, file_path: str) -> List[str]:
        """Load document using specific strategy"""
        pass


class DocumentLoader(ABC):
    """Abstract base class for all document loaders.
    Template method pattern for common loading behavior.
    """
    
    # Class variable to store valid strategy types for each loader
    _valid_strategies: Dict[str, List[Type[LoaderStrategy]]] = {}
    
    @classmethod
    def register_strategies(cls, strategies: List[Type[LoaderStrategy]]):
        """Register valid strategies for this loader type"""
        cls._valid_strategies[cls.__name__] = strategies
    
    def __init__(self, strategy: LoaderStrategy = None):
        self._strategy = self._validate_strategy(strategy) if strategy else None
    
    def _validate_strategy(self, strategy: LoaderStrategy) -> LoaderStrategy:
        """Validate if the strategy is allowed for this loader"""
        valid_strategies = self._valid_strategies.get(self.__class__.__name__, [])
        if not valid_strategies:
            raise ValueError(f"No strategies registered for {self.__class__.__name__}")
        
        if not isinstance(strategy, tuple(valid_strategies)):
            valid_names = [s.__name__ for s in valid_strategies]
            raise ValueError(
                f"Invalid strategy for {self.__class__.__name__}. "
                f"Must be one of: {', '.join(valid_names)}"
            )
        return strategy
    
    @abstractmethod
    def load_data(self, file_path: Union[str, io.BytesIO]) -> List[str]:
        """Load document data and return list of text content"""
        pass

    def validate_file(self, file_path: Union[str, io.BytesIO]) -> bool:
        """Validate if file exists and has correct extension"""
        if isinstance(file_path, io.BytesIO):
            return True  # BytesIO is always valid

        path = Path(file_path)
        return path.exists() and path.suffix.lower() in self.supported_extensions()
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        pass
    
    def set_strategy(self, strategy: LoaderStrategy):
        """Change loading strategy at runtime"""
        self._strategy = self._validate_strategy(strategy)


class LoaderFactory:
    """Factory class for creating document loaders"""
    
    _loaders = {}
    
    @classmethod
    def register_loader(cls, extensions: List[str], loader_class: Type[DocumentLoader]):
        """Register a loader class for given file extensions"""
        for ext in extensions:
            cls._loaders[ext.lower()] = loader_class
    
    @classmethod
    def get_loader(cls, file_path: str) -> DocumentLoader:
        """Get appropriate loader for given file path"""
        ext = Path(file_path).suffix.lower()
        loader_class = cls._loaders.get(ext)
        if loader_class is None:
            raise ValueError(f"No loader registered for extension: {ext}")
        return loader_class()
