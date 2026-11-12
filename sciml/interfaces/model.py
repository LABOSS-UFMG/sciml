# -------------------------------------------------------------------------------- #
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager

from sciml.contracts.context import Context

# -------------------------------------------------------------------------------- #
class ModelBase(ABC):
    """
    Abstract base class for trainable models.

    A ``ModelBase`` is responsible for evaluating a neural network and
    storing the computed variables in a :class:`Context`. Unlike the
    standard PyTorch ``forward`` method, the evaluation does not return
    any value directly. Instead, all outputs required by subsequent
    components (e.g. loss functions) are written to the provided context.

    This design allows multiple losses to share intermediate results and
    enables lazy computation of derived quantities, such as partial
    derivatives.
    """

    @contextmanager
    def evaluation_mode(self) -> Generator[None, None, None]:
        """
        Provide a context for evaluating the model. The default preserves
        the current module and autograd modes.

        Examples
        --------
        A subclass with Dropout or BatchNorm can override this method::

            @contextmanager
            def evaluation_mode(self) -> Generator[None, None, None]:
                modes = {m: m.training for m in self.network.modules()}
                try:
                    self.network.eval()
                    yield
                finally:
                    for module, training in modes.items():
                        module.training = training

        Evaluate an objective inside the context::

            with model.evaluation_mode():
                evaluation = objective.evaluate(model)
        """
        yield

    @abstractmethod
    def compute(self, context: Context) -> None:
        """
        Evaluate the model.

        The implementation should read the required input variables from
        the context, compute the model outputs and store the resulting
        tensors back into the same context.

        Parameters
        ----------
        context : Context
            Evaluation context containing the input variables and used to
            store the model outputs.
        """
        pass

# -------------------------------------------------------------------------------- #
