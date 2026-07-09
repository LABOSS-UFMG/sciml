#####################################################################################
from abc import ABC, abstractmethod
from typing import Any

#####################################################################################
class CallbackBase(ABC):
    """
    Abstract base class for training callbacks.

    A callback lets you plug custom behavior into specific points of the
    training loop(e.g. saving checkpoints, logging progress stopping early)
    without modifying the trainer itself.
    """

    @abstractmethod
    def on_train_start(self, *args, **kargs) -> Any:
        """Called once, before the first training iteration"""
        pass
    
    @abstractmethod
    def on_iteration_start(self, *args, **kargs) -> Any:
        """Called at the beginning of each training iteration"""
        pass
    
    @abstractmethod
    def on_iteration_end(self, *args, **kargs) -> Any:
        """Called at the end of each training iteration"""
        pass
    
    @abstractmethod
    def on_train_end(self, *args, **kargs) -> Any:
        """Called once, after the last training iteration"""
        pass
    
    @abstractmethod
    def on_exception(self, *args, **kargs) -> Any:
        """Called when an exception is raised inside the training loop"""
        pass

#####################################################################################
