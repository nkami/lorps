import torch
import torch.nn as nn
import copy
import numpy as np

from utils import sep_polys_sum, sep_polys_sum_multiply, sep_polys_sum_add
from schrodinger_equation import DOMAIN_ABS_EDGE, EDGE_VAL

# parts of code taken from: https://github.com/AdityaLab/pinnsformer


class SPS(nn.Module):
    def __init__(self, s, d, k, equation):
        super(SPS, self).__init__()
        coeffs = torch.zeros(s, d, k)
        coeffs[:, :, 0].uniform_(-1, 1)
        self.params = nn.Parameter(coeffs)
        self.equation = equation

    def forward(self, x):
        dim = self.params.shape[1]

        if self.equation == 'nonlocal_diffusion':
            boundary_conditions = torch.zeros((1, dim, 3)).to(self.params.device)
            boundary_conditions[0, :, 0] = -1
            boundary_conditions[0, :, 2] = 1
            cur_params = sep_polys_sum_multiply(self.params, boundary_conditions)
        elif self.equation == 'schrodinger':
            boundary_conditions = torch.zeros((1, dim, 3)).to(self.params.device)
            boundary_conditions[0, :, 0] = -1 * DOMAIN_ABS_EDGE ** 2
            boundary_conditions[0, :, 2] = DOMAIN_ABS_EDGE ** 2

            boundary_conditions2 = torch.zeros((1, dim, 3)).to(self.params.device)
            boundary_conditions2[0, 0, 0] = EDGE_VAL

            cur_params = sep_polys_sum_multiply(self.params, boundary_conditions)
            cur_params = sep_polys_sum_add(cur_params, boundary_conditions2)
            x = x / DOMAIN_ABS_EDGE
        elif self.equation == 'heat':
            # boundary and initial conditions: prod * t * u
            boundary_poly_prod = torch.zeros((1, dim, 3)).to(self.params.device)
            boundary_poly_prod[0, :-1, 1] = 3
            boundary_poly_prod[0, :-1, 2] = -3
            boundary_poly_prod[0, -1, 1] = 1

            cur_params = sep_polys_sum_multiply(self.params, boundary_poly_prod)
        else:
            cur_params = self.params

        return sep_polys_sum(x, cur_params)


class SinAct(nn.Module):
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class FLS(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(FLS, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class PINN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(PINN, self).__init__()

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class QRes_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QRes_block, self).__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1 * x2 + x1)


class QRes(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(QRes, self).__init__()
        self.N = num_layer-1
        self.inlayer = QRes_block(in_dim, hidden_dim)
        self.layers = get_clones(QRes_block(hidden_dim, hidden_dim), num_layer-1)
        self.outlayer = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        src = self.inlayer(x)
        for i in range(self.N):
            src = self.layers[i](src)
        src = self.outlayer(src)
        return src


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class PINNsformer(nn.Module):
    def __init__(self, in_dim, d_out, d_model, d_hidden, N, heads):
        super(PINNsformer, self).__init__()

        self.linear_emb = nn.Linear(in_dim, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        ])

    def forward(self, x):
        src = self.linear_emb(x)

        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        return output


def get_model_setup(method, equation, dim, s=None, k=None):
    """Build a model; optional s/k overrides apply only to SPS."""
    if method == 'sps':
        supported = {'nonlocal_diffusion', 'convolution', 'heat', 'schrodinger'}
        if 'fredholm' not in equation and equation not in supported:
            raise ValueError(f'Unknown SPS equation: {equation}')
        default_s, default_k = (20, 8)
        s = default_s if s is None else s
        k = default_k if k is None else k
        if s < 1 or k < 1:
            raise ValueError('s and k must be positive')
        return SPS(s, dim, k, equation), 1e-3, -1, None
    if s is not None or k is not None:
        raise ValueError('s and k overrides are only supported for SPS')

    if method == 'pinn':
        if 'fredholm' in equation:
            return PINN(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 15_000, None
        elif equation == 'convolution':
            return PINN(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-4, 15_000, None
        elif equation == 'nonlocal_diffusion':
            return PINN(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-4, 10_000, None
        elif equation == 'heat':
            return PINN(in_dim=dim, hidden_dim=64, out_dim=1, num_layer=4), 5e-4, 1_000, None
        else:
            return None, None, None

    if method == 'fls':
        if 'fredholm' in equation:
            return FLS(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 15_000, None
        elif equation == 'convolution':
            return FLS(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 15_000, None
        elif equation == 'nonlocal_diffusion':
            return FLS(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 10_000, None
        elif equation == 'heat':
            return FLS(in_dim=dim, hidden_dim=64, out_dim=1, num_layer=4), 5e-3, 1_000, None
        else:
            return None, None, None

    if method == 'qres':
        if 'fredholm' in equation:
            return QRes(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 15_000, None
        elif equation == 'convolution':
            return QRes(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 15_000, None
        elif equation == 'nonlocal_diffusion':
            return QRes(in_dim=dim, hidden_dim=256, out_dim=1, num_layer=4), 5e-3, 9_000, None
        elif equation == 'heat':
            return QRes(in_dim=dim, hidden_dim=32, out_dim=1, num_layer=4), 5e-3, 1_000, None
        else:
            return None, None, None

    if method == 'pinnsformer':
        if 'fredholm' in equation:
            return PINNsformer(in_dim=dim, d_out=1, d_model=16, d_hidden=128, N=1, heads=2), 5e-3, 20_000, 10
        elif equation == 'convolution':
            return PINNsformer(in_dim=dim, d_out=1, d_model=16, d_hidden=64, N=1, heads=2), 5e-3, 12_000, 10
        elif equation == 'nonlocal_diffusion':
            return PINNsformer(in_dim=dim, d_out=1, d_model=16, d_hidden=64, N=1, heads=2), 5e-3, 3_000, 10
        elif equation == 'heat':
            return PINNsformer(in_dim=dim, d_out=1, d_model=8, d_hidden=16, N=1, heads=2), 5e-3, 100, 10
        else:
            return None, None, None
