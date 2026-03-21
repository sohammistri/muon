import torch.nn as nn
import torch
import torch.optim as optim


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)  # momentum = beta * momentum + (1-beta) * grad
    update = momentum*beta + grad*(1-beta) if nesterov else momentum

    if update.ndim == 4:  # for the case of conv filters
        update = update.view(update.shape[0], -1)  # keep first dim, flatten the rest

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95, steps=5,
                 weight_decay=0, nesterov=True, maximize=False):

        defaults = dict(lr=lr, momentum=momentum, steps=steps,
                        weight_decay=weight_decay, nesterov=nesterov)
        self.maximize = maximize
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            steps = group['steps']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue

                if self.maximize:
                    grad = -p.grad
                else:
                    grad = p.grad

                # Get state for this parameter
                state = self.state[p]
                # Initialize momentum buffer if it doesn't exist
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                # Apply weight decay directly to parameters (AdamW style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Apply zeropower_via_newtonschulz5 if parameter is a matrix
                if p.ndim >= 2:
                    org_shape = p.shape
                    orthogonal_buff = muon_update(grad, state["momentum_buffer"],
                                                beta=momentum, ns_steps=steps, nesterov=nesterov)
                    if p.ndim == 4:
                        orthogonal_buff = orthogonal_buff.view(org_shape)

                    # Apply update to parameters
                    p.add_(orthogonal_buff, alpha=-lr)
                else:
                    # For non-matrix parameters, use standard momentum update
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                    # furhter modify buf if nesterov
                    if nesterov:
                        buf = grad + momentum * buf

                    # Apply update to parameters
                    p.add_(buf, alpha=-lr)

        return loss