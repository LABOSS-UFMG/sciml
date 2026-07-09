#####################################################################################
import torch

#####################################################################################
def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Compute the derivative of ``y`` with respect to ``x`` via autograd.

    This returns, for each column of ``x``, the partial derivative of
    ``y`` with respect to that variable (i.e. the full gradient of ``y``
    w.r.t. ``x``, not a single component).

    Note
    ----
    For ``order > 1``, this differentiates the *same* tensor ``y``
    repeatedly with respect to ``x`` — it computes pure higher-order
    derivatives (e.g. ``d²y/dx²``), not mixed partials. To compute mixed
    derivatives (e.g. ``d²u/dx dt``), call ``derivative`` once to obtain
    the first-order gradient, then call it again on the relevant
    component. See the class-level example above.

    Parameters
    ----------
    y : torch.Tensor
        Output of the network (e.g. ``u(x)``), shape ``[N, 1]``.
    x : torch.Tensor
        Input tensor with respect to which the derivative is taken.
        Must have ``requires_grad=True``. Shape ``[N, D]``.
    order : int, default=1
        Order of the derivative (1 = first derivative, 2 = second, ...).

    Returns
    -------
    torch.Tensor
        Shape ``[N, D]``: derivative of ``y`` with respect to each of
        the ``D`` variables in ``x``.
    """
    for _ in range(order):
        y = torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
        )[0]
    return y

#####################################################################################