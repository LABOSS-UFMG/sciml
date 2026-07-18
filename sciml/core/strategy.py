#####################################################################################
from abc import ABC, abstractmethod
from typing import Iterable

from sciml.core.model import ModelBase
from sciml.core.objective import Objective
from sciml.core.evaluation import Evaluation

#####################################################################################
class StrategyBase(ABC):
    """
    Optimization strategy used during training.

    A ``Strategy`` defines how one or more objectives are evaluated and how
    their losses are combined to update the model parameters.
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name : str
            Short identifier used when logging or plotting this strategy.
        """
        # ---------------------------------------------------------------------------
        self.name = name
        # ---------------------------------------------------------------------------
        return
    
    @abstractmethod
    def step(
            self,
            model: ModelBase,
            objectives: Iterable[Objective],
        ) -> Evaluation:
        """
        Perform a single optimization step.

        Parameters
        ----------
        model : ModelBase
            Model to be optimized.
        objectives : Iterable[Objective]
            Collection of objectives evaluated during the optimization
            step.

        Returns
        -------
        dict[str, Context]
            Dictionary containing the evaluation context associated with
            each objective.
        """
        pass

#####################################################################################
