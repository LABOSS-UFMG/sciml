# -------------------------------------------------------------------------------- #
from abc import ABC, abstractmethod

# -------------------------------------------------------------------------------- #
class CallbackBase(ABC):
    """
    Abstract base class for training callbacks.

    A callback lets you plug custom behavior into specific points of the
    training loop without modifying the trainer itself.
    """

    @abstractmethod
    def on_train_start(self) -> None:
        """Called once, before the first training iteration"""
        pass
    
    @abstractmethod
    def on_iteration_start(self) -> None:
        """Called at the beginning of each training iteration"""
        pass
    
    @abstractmethod
    def on_iteration_end(self) -> None:
        """Called at the end of each training iteration"""
        pass
    
    @abstractmethod
    def on_train_end(self) -> None:
        """Called once, after the last training iteration"""
        pass
    
    @abstractmethod
    def on_exception(self) -> None:
        """Called when an exception is raised inside the training loop"""
        pass

# -------------------------------------------------------------------------------- #
