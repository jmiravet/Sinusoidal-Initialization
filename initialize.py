import torch
import torch.nn as nn
import numpy as np
import math

def fernandez_sinusoidal(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        for i in range(n_out):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_out+n_in)))
        weight = weight * a

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal3(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) * (2 * np.pi / n_in) + (2 * np.pi / n_out)

        for i in range(n_out):
            weight[i,:] = np.sin(position * (i+1))
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/ (var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange(c_out) * (2 * np.pi / c_out)

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[i]
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal2(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        for i in range(n_out):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_out+n_in)))
        weight = weight * a

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(c_out) 
        for i in range(n_in):
            l = ((i+1) * 2*np.pi / n_in)
            phase = np.random.rand() * 2 * np.pi
            i_position = position * l + phase 
            weight[:,i] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal4(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in)
        phases = np.arange((n_out // 2)) * (2 * np.pi / (n_out // 2))

        for i in range(n_out):
            l = ((i//2)+1) * 2*np.pi / (n_in)
            i_position = position * l + phases[(i//2)] + ((i%2) * np.pi / 2)
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/(var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange((c_out // 2)) * (2 * np.pi / (c_out // 2))

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[(i//2)] + ((i%2) * np.pi / 2)
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal5(module, mode="all"):
    def initialize_linear(weight, mode):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in)

        for i in range(0,n_out):
            i_position = ((position+i) % n_in) - (n_in-1) / 2
            weight[i,:] = np.sin(np.pi * (i+1)  *(i_position/n_in))
        
        var = np.var(weight, ddof=1)
        if mode == "fan_in":
            amplitude = np.sqrt(2/ (var * (n_in)))
        elif mode == "fan_out":
            amplitude = np.sqrt(2/ (var * (n_out)))
        else:
            amplitude = np.sqrt(2/ (var * (n_out+n_in)))

        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight, mode):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 

        for i in range(c_out):
            i_position = ((position+i) % n_in) - (n_in-1) / 2
            weight[i,:] = np.sin(np.pi * (i+1)  *(i_position/n_in))
        
        var = np.var(weight, ddof=1)
        if mode == "fan_in":
            amplitude = np.sqrt(2/ (var * (n_in)))
        elif mode == "fan_out":
            amplitude = np.sqrt(2/ (var * (c_out)))
        else:
            amplitude = np.sqrt(2/ (var * (c_out+n_in)))
        weight = weight * amplitude
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight, mode)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight, mode)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

def fernandez_sinusoidal_random(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) * (2 * np.pi / n_in) + (2 * np.pi / n_out)

        for i in range(n_out):
            weight[i,:] = np.sin(position * (i+1))
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/ (var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        idx = torch.randperm(weight.shape[0])
        weight = weight[idx].view(weight.size())
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange(c_out) * (2 * np.pi / c_out)

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[i]
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        idx = torch.randperm(weight.shape[0])
        weight = weight[idx].view(weight.size())
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

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

def fernandez_sinusoidal3(module):
    def initialize_linear(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        n_out, n_in = weight.shape
        weight = np.empty((n_out, n_in), dtype=np.float32)
        position = np.arange(n_in) * (2 * np.pi / n_in) + (2 * np.pi / n_out)

        for i in range(n_out):
            weight[i,:] = np.sin(position * (i+1))
        
        var = np.var(weight, ddof=1)
        amplitude = np.sqrt(2/ (var * (n_out+n_in)))
        weight = weight * amplitude

        weight = torch.from_numpy(weight)
        return weight

    def initialize_conv(weight):
        # A: amplitud, l: longitud de onda, d: desplazamiento
        c_out, c_in, h, w = weight.shape
        n_in = c_in*h*w
        weight = np.empty((c_out, n_in), dtype=np.float32)
        position = np.arange(n_in) 
        phases = np.arange(c_out) * (2 * np.pi / c_out)

        for i in range(c_out):
            l = ((i+1) * 2*np.pi / c_out)
            i_position = position * l + phases[i]
            weight[i,:] = np.sin(i_position)
        
        var = np.var(weight, ddof=1)
        a = np.sqrt(2/(var * (n_in+c_out)))
        weight = weight * a
        weight = weight.reshape((c_out, c_in, h, w))

        weight = torch.from_numpy(weight)
        return weight

    if isinstance(module, nn.Linear):
        module.weight.data = initialize_linear(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        module.weight.data = initialize_conv(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def default_initialization(module):
    pass

def init_orthogonal(m: nn.Module):
    """
    Inicializa pesos ortogonales en Linear y Conv, sesgos a cero.
    Ignora módulos con pesos 1D (ej. BatchNorm).
    """
    if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
        if m.weight.ndimension() >= 2:
            nn.init.orthogonal_(m.weight)
        else:
            # ej. BatchNorm: inicializa a 1 para no romper la normalización
            nn.init.ones_(m.weight)
    if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor):
        nn.init.zeros_(m.bias)

# ---------- Utilidades ----------
def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def _hadamard(n: int, device=None, dtype=None):
    """Genera H_n con entradas ±1; requiere n potencia de 2."""
    if not _is_power_of_two(n):
        raise ValueError("Hadamard size must be a power of two")
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H

def _zero_matrix_like(m: int, n: int, device, dtype):
    return torch.zeros((m, n), device=device, dtype=dtype)

# ---------- ZerO para matrices (Algoritmo 1) ----------
def zero_init_on_matrix(W: torch.Tensor, match_repo_normalization: bool = False):
    """
    Inicializa una matriz W (out x in) según ZerO.

    - Si out == in: identidad
    - Si out <  in: identidad parcial (copiando primeras 'out' dims)
    - Si out >  in: I* H_m I* con escala:
        * Paper:       c = 2^{-(m-1)/2}
        * Repo (flag): c = 2^{-m/2}
      donde m = ceil(log2(out)), H_m es de tamaño 2^m.
    """
    out_f, in_f = W.shape
    dev, dt = W.device, W.dtype
    with torch.no_grad():
        W.zero_()
        if out_f <= in_f:
            idx = torch.arange(out_f, device=dev)
            W[idx, idx] = 1.0
        else:
            m = math.ceil(math.log2(out_f))
            M = 1 << m
            H = _hadamard(M, device=dev, dtype=dt)
            if match_repo_normalization:
                c = 2 ** (-m / 2.0)        # como en su repo
            else:
                c = 2 ** (-(m - 1) / 2.0)  # como en el paper
            block = c * H[:out_f, :in_f]
            W.copy_(block)

# ---------- ZerO para capas lineales ----------
def _zero_init_linear(weight: torch.Tensor, match_repo_normalization: bool = False):
    zero_init_on_matrix(weight, match_repo_normalization)

# ---------- ZerO para convoluciones (Algoritmo 2) ----------
def _zero_init_conv2d(weight: torch.Tensor, match_repo_normalization: bool = False):
    """
    weight: (cout, cin, kh, kw)
    Inserta el mapa canales->canales en el centro espacial.
    """
    cout, cin, kh, kw = weight.shape
    if kh != kw:
        raise ValueError("ZerO conv init asume kernels cuadrados")
    center = kh // 2  # se recomienda kernel impar
    with torch.no_grad():
        weight.zero_()
        # Construimos la matriz canales->canales y la volcamos en el centro
        Wcc = torch.empty((cout, cin), device=weight.device, dtype=weight.dtype)
        zero_init_on_matrix(Wcc, match_repo_normalization)
        weight[:, :, center, center].copy_(Wcc)

# ---------- Hook para model.apply ----------
def zero_init(module: nn.Module, match_repo_normalization: bool = False):
    """
    Uso: model.apply(lambda m: zero_init(m)) o model.apply(partial(zero_init, match_repo_normalization=True))
    """
    if isinstance(module, nn.Linear):
        _zero_init_linear(module.weight, match_repo_normalization)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        _zero_init_conv2d(module.weight, match_repo_normalization)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def blumenfeld_init(eps: float = 0.0):
    """
    Devuelve una función de inicialización estilo ConstNet (Blumenfeld et al., ICML'20).
    Se puede pasar a model.apply.

    eps: desviación estándar del ruido gaussiano minúsculo para romper simetría.
    """
    def _init(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.zeros_(module.weight)
                if eps > 0:
                    with torch.no_grad():
                        module.weight.add_(torch.randn_like(module.weight) * eps)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
                if eps > 0:
                    with torch.no_grad():
                        module.bias.add_(torch.randn_like(module.bias) * eps)
    return _init

def arcsine_random_init(module: nn.Module = None):
    """
    Inicialización con distribución arcoseno en [-a, a], con Var = 2/(fan_in + fan_out).
    Uso flexible:
      model.apply(arcsine_random_init)      # directo
      model.apply(arcsine_random_init())    # también funciona
    """

    def _calc_fans(w: torch.Tensor):
        # Adaptado a Linear y Conv*d, incluidos grupos
        if w.ndim < 2:
            # No inicializamos pesos 1D aquí (p. ej., BatchNorm)
            return None, None
        receptive_field = 1
        if w.ndim > 2:
            receptive_field = w[0][0].numel()  # producto de kernel dims
        fan_in  = w.size(1) * receptive_field
        fan_out = w.size(0) * receptive_field
        return fan_in, fan_out

    @torch.no_grad()
    def _init(m: nn.Module):
        if not hasattr(m, "weight") or not isinstance(m.weight, torch.Tensor):
            return
        fan_in, fan_out = _calc_fans(m.weight)
        if fan_in is None:
            return  # pesos 1D u otros casos que no tocaremos

        # Var objetivo tipo "He/Xavier-sum": 2/(fan_in + fan_out)
        var = 2.0 / float(fan_in + fan_out)
        a = math.sqrt(2.0 * var)  # porque Var[arcsine[-a,a]] = a^2/2  => a = sqrt(2*Var)

        # Muestreo por transformación inversa: X = a*cos(pi*U), U~U(0,1)
        U = torch.rand_like(m.weight)
        W = a * torch.cos(math.pi * U)
        m.weight.copy_(W)

        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

    # Soporta ambos usos
    if module is None:
        return _init
    else:
        return _init(module)
