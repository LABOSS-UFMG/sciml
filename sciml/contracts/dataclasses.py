# -------------------------------------------------------------------------------- #
import torch

from dataclasses import dataclass, field
from typing import Dict, Sequence

# -------------------------------------------------------------------------------- #
@dataclass(slots=True)
class Evaluation():
    """Store the information of the current objective"""
    # Objective name
    name: str

    # Objective function value
    objective: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))

    # Losses and its corresponding weights
    losses: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass(slots=True)
class Step():
    """Store the information of the current optimization step"""
    # Strategy name
    name: str
    
    # Evaluations
    evaluations: Sequence[Evaluation]

# -------------------------------------------------------------------------------- #
