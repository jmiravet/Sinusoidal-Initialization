import math
from typing import Optional as _Optional

import torch
import torch.nn as nn
from torch import Tensor

def sinusoidal_(
    tensor: Tensor,
    generator: _Optional[torch.Generator] = None,
) -> Tensor:
    r"""Fill the input `Tensor` with sinusoidal patterns for structured initialization.

    The method is described in `Sinusoidal Initialization, Time for a New Start` - 
    Fernandez-Hernandez, A. et al. (2025).

    This initialization assigns each output channel (or neuron) a sinusoidal
    weight pattern with varying frequency and phase, scaled to preserve
    variance similar to Kaiming/Xavier methods.

    The sinusoidal initialization encourages smooth and diverse feature
    selectivity at initialization, potentially aiding convergence and
    interpretability in early training stages.

    Args:
        tensor: an n-dimensional `torch.Tensor`
            - For Linear layers: shape ``(out_features, in_features)``
            - For Conv layers: shape ``(out_channels, in_channels, kernel_height, kernel_width)``
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sinusoidal_(w)
        >>> conv_w = torch.empty(16, 3, 3, 3)
        >>> nn.init.sinusoidal_(conv_w)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            sinusoidal_, (tensor,), tensor=tensor, generator=generator
        )

    with torch.no_grad():
        # Flatten input channels for unified handling
        shape = tensor.shape
        if tensor.ndimension() == 2:
            n_out, n_in = shape
            n_flat = n_in
        elif tensor.ndimension() >= 3:
            n_out, n_in = shape[0], shape[1] * math.prod(shape[2:])
            n_flat = n_in
        else:
            raise ValueError("Tensor must have at least 2 dimensions")

        # Create sinusoidal pattern
        phase = (2 * math.pi / n_out)
        freq = (2 * math.pi / n_flat)
        position = torch.arange(n_flat, dtype=torch.float32) * freq + phase

        # Give each neuron a different wave
        neuron = torch.arange(n_out, dtype=torch.float32) + 1
        weights = torch.sin(position.unsqueeze(0) * neuron.unsqueeze(1))

        # Normalize amplitude
        var = torch.var(weights, unbiased=True)
        amplitude = math.sqrt(2.0 / (var.item() * (n_in + n_out)))
        weights = weights * amplitude

        # Reshape for conv layers if needed
        if tensor.ndimension() >= 3:
            weights = weights.view(shape[0], shape[1], *shape[2:])

        tensor.copy_(weights)
        return tensor

def sinusoidal_init(m: nn.Module):
    gen = torch.Generator().manual_seed(42)
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if getattr(m, "weight", None) is not None:
            sinusoidal_(m.weight, generator=gen)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def orthogonal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def xavier_normal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def xavier_uniform(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def kaiming_normal(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def kaiming_uniform(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
