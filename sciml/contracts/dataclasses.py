# -------------------------------------------------------------------------------- #
import torch

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional

# -------------------------------------------------------------------------------- #
@dataclass(slots=True)
class Evaluation():
    """Store the information of the current objective"""
    # Objective name
    name: str

    # Objective function value
    objective: Optional[torch.Tensor] = None

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
