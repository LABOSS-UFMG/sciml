#####################################################################################
import torch

from typing import Any, Dict, NamedTuple

#####################################################################################
class Evaluation(NamedTuple):
    """Store the information of the current iteration"""
    values: Dict[str, torch.Tensor]   # Loss values computed by each loss function
    weights: Dict[str, float]         # Weight associated with each loss
    metadata: Dict[str, Any]          # Auxiliary metadata

#####################################################################################
