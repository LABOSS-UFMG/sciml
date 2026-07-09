#####################################################################################
from abc import ABC, abstractmethod

from sciml.core.contracts import Batch

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
    # Subclasses should override:
    name: str       # Short identifier used when logging or referring to this sampler.

    @abstractmethod
    def next(self) -> Batch:
        """
        Generate the next batch of sampled points.

        Returns
        -------
        Batch
            A batch containing the sampled points (and, if applicable,
            associated target data) to be consumed by a network and a
            ``LossBase`` (see ``sciml.core.contracts.Batch``).
        """
        pass

#####################################################################################