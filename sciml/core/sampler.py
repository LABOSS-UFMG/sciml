#####################################################################################
from abc import ABC, abstractmethod

from sciml.core.context import Context

#####################################################################################
class SamplerBase(ABC):
    """
    Abstract base class for sampling training and testing data.

    A sampler is responsible for generating the points fed to the network
    and to the losses at each training iteration (e.g. collocation points
    inside the domain, points on the boundary, or points with observed
    data). Each call to ``next`` produces one such batch of points.

    Note
    ----
    Most users will not need to subclass ``SamplerBase`` directly. The
    library already provides samplers for common cases (e.g. uniform
    sampling on a rectangular domain). Subclass this only when a new
    sampling strategy or a non-trivial geometry is required.
    """
    
    @abstractmethod
    def next(self) -> Context:
        """
        Generate the next batch of sampled points.

        Returns
        -------
        context : Context
            Evaluation context containing all variables required by the neural network.
        """
        pass

#####################################################################################