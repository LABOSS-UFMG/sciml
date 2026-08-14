# -------------------------------------------------------------------------------- #
import torch

from sciml.contracts.context import Context
from sciml.interfaces import ModelBase, LossBase

# -------------------------------------------------------------------------------- #
class LinearModel(ModelBase):
    """Wraps a single linear layer: y = net(x). Used as a minimal ModelBase."""

    def __init__(self) -> None:
        self.network = torch.nn.Linear(1, 1, bias=False)

    def compute(self, context: Context) -> None:
        context["y"] = self.network(context["x"])

class SquaredLoss(LossBase):
    """Pushes the model's output towards zero; enough to check optimization steps."""

    def evaluate(self, context: Context) -> torch.Tensor:
        return (context["y"] ** 2).mean()

class ConstantSampler():
    """Sampler returning the same batch of inputs every call, for reproducible tests."""

    def __init__(self, batch_size: int = 8, value: float = 2.0) -> None:
        self.batch_size = batch_size
        self.value = value

    def next(self) -> Context:
        context = Context()
        context["x"] = torch.full((self.batch_size, 1), self.value)
        return context

# -------------------------------------------------------------------------------- #
