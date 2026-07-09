#####################################################################################
from abc import ABC, abstractmethod
from typing import Any

#####################################################################################
class TrainerBase(ABC):
    """
    Abstract base class for training neural network models.

    A trainer is the orchestrator of the training loop: it ties together a
    network, one or more samplers (to generate collocation/boundary/data
    points), one or more losses (to be minimized), and optionally metrics
    and callbacks (to monitor and react to training progress).

    Note
    ----
    Most users will not need to subclass ``TrainerBase`` directly. Instead,
    use one of the concrete trainers already provided by the library
    (e.g. ``StandardTrainer``), and focus on implementing the problem
    specific parts of your model (namely, a ``LossBase`` subclass).
    """

    @abstractmethod
    def fit(self, *args, **kargs) -> Any:
        """
        Execute the training process.

        Concrete trainers define their own specific signature (e.g. network,
        samplers, losses, optimizer, number of epochs, callbacks). Consult
        the docstring of the concrete trainer being used for details.

        Parameters
        ----------
        *args, **kwargs : Any
            Trainer-specific arguments.

        Returns
        -------
        Any
            The return value is trainer specific (e.g. a training history
            object with recorded losses and metrics over time).
        """
        pass

#####################################################################################