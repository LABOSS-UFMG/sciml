#####################################################################################
from typing import NamedTuple, Dict, Any, Optional

#####################################################################################
class Batch(NamedTuple):
    """
    Standard data container exchanged between a ``SamplerBase`` and ``LossBase``.

    Using a single, explicit structure for this exchange (instead of loose
    positional arguments) makes the contract between samplers and losses
    explicit and type-checkable.

    Attributes
    ----------
    inputs : Dict[str, Any]
        Sampled points, keyed by variable name (e.g. ``{"xt": tensor}`` for
        a batch of (x, t) collocation points). The keys used here must match
        what the corresponding ``LossBase.evaluate`` expects to find.
    targets : Dict[str, Any], optional
        Reference values associated with ``inputs``, when available (e.g.
        observed data for an observational loss, or exact boundary values).
        Not all losses require targets — collocation losses derived purely
        from the PDE residual, for instance, typically leave this as ``None``.
    """
    inputs: Dict[str, Any]
    targets: Optional[Dict[str, Any]] = None

#####################################################################################