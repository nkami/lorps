from pathlib import Path
import torch
import math
from scipy.special import erfi
from torch.quasirandom import SobolEngine

from utils import sep_polys_sum_integral, sep_polys_sum_add, sep_polys_sum_multiply, sep_polys_sum_prepend_dims, \
    sample_nd_points, sep_polys_sum_remove_trailing_zeros, lorps_L2_sq

SIGMA = 2

def _set_up_pretrained_terms(num_dims):
    device_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_pretrained_sinh_gauss = torch.load(Path(__file__).resolve().parent / "projections/sinh_1d.pt", map_location=torch.device(device_location))
    sinh_1d_exp_params = loaded_pretrained_sinh_gauss['state_dict']["params"]
    padded_1d_params = []
    for cur_dim in range(num_dims):
        cur_pad = torch.zeros((1, num_dims - 1, sinh_1d_exp_params.shape[2]), device=sinh_1d_exp_params.device)
        cur_pad[0, :, 0] = 1
        cur_pad_left = cur_pad[:, :cur_dim, :]
        cur_pad_right = cur_pad[:, cur_dim:, :]
        cur_padded_1d_params = torch.concat([cur_pad_left, sinh_1d_exp_params, cur_pad_right], dim=1)
        padded_1d_params.append(cur_padded_1d_params)
    sum_of_sinhs = torch.cat(padded_1d_params, dim=0)

    erfi_const = torch.ones((1, num_dims, 1), device=sum_of_sinhs.device)
    erfi_const[0, 0, 0] = (1 / (1 * num_dims)) * (SIGMA * (math.pi ** 0.5) * erfi(0.5 / SIGMA)) ** (num_dims - 1)
    sum_of_sinhs = sep_polys_sum_multiply(sum_of_sinhs, erfi_const)

    loaded_pretrained_exp = torch.load(Path(__file__).resolve().parent / "projections/exp_1d.pt", map_location=torch.device(device_location))
    exp_1d_params = loaded_pretrained_exp['state_dict']["params"]
    exp_prod = exp_1d_params.repeat(1, num_dims, 1)
    neg_one_const = torch.ones((1, num_dims, 1), device=sum_of_sinhs.device)
    neg_one_const[0, 0, 0] = -1
    neg_exp_prod = sep_polys_sum_multiply(exp_prod, neg_one_const)

    neg_f = sep_polys_sum_add(sum_of_sinhs, neg_exp_prod)

    loaded_pretrained_gauss = torch.load(Path(__file__).resolve().parent / "projections/gauss_2d.pt", map_location=torch.device(device_location))
    gauss_2d_params = loaded_pretrained_gauss['state_dict']["params"]

    kernel = []
    for cur_dim in range(num_dims):
        cur_pad = torch.zeros((gauss_2d_params.shape[0], 2 * num_dims, gauss_2d_params.shape[2]), device=sum_of_sinhs.device)
        cur_pad[:, :, 0] = 1
        cur_pad[:, cur_dim, :] = gauss_2d_params[:, 0, :]
        cur_pad[:, num_dims + cur_dim, :] = gauss_2d_params[:, 1, :]
        kernel.append(cur_pad)
    kernel = torch.cat(kernel, dim=0)
    kernel_const = torch.ones((1, 2 * num_dims, 1), device=sum_of_sinhs.device)
    kernel_const[0, 0, 0] = 1 / (1 * num_dims)
    kernel = sep_polys_sum_multiply(kernel, kernel_const)

    return kernel, neg_f


def generate_sps_func(dim=10, device='cpu'):
    pretrained_kernel, pretrained_neg_f = _set_up_pretrained_terms(dim)
    kernel = pretrained_kernel.to(device)
    neg_f = pretrained_neg_f.to(device)

    def deblurring_loss_func(model):
        params = model.params

        prepend_params = sep_polys_sum_prepend_dims(params, num_dims=dim)
        integral_params = sep_polys_sum_multiply(prepend_params, kernel)

        neg_one_const = torch.ones((1, dim, 1), device=params.device)
        neg_one_const[0, 0, 0] = -1

        res_start = torch.ones(dim, device=params.device) * -0.5
        res_end = torch.ones(dim, device=params.device) * 0.5
        integral_res_params = sep_polys_sum_integral(integral_params, res_start, res_end, dim_range=(dim, 2 * dim))
        integral_res_params = sep_polys_sum_remove_trailing_zeros(integral_res_params)

        neg_integral_res_params = sep_polys_sum_multiply(integral_res_params, neg_one_const)

        eq_params = sep_polys_sum_add(neg_integral_res_params, neg_f)
        eq_params = sep_polys_sum_add(eq_params, params)

        start = torch.ones(dim, device=params.device) * -0.5
        end = torch.ones(dim, device=params.device) * 0.5
        integral_val = lorps_L2_sq(eq_params, start, end)
        loss_terms = [integral_val]

        loss_terms = torch.stack(loss_terms, dim=0)
        loss = torch.sum(loss_terms)

        return torch.log(loss)

    optimized_func = torch.compile(deblurring_loss_func)
    return optimized_func


def generate_pinn_func(dim=10, num_points=None, method='monte_carlo', sampling='uniform'):
    def kernel_function(x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        d = x.shape[-1]

        indices = torch.arange(d)
        kernel = torch.sum(torch.exp(-1 * (x[:, :, indices] - y[:, :, indices]) ** 2 / SIGMA ** 2), dim=-1)
        return kernel

    def calc_residual(model, x_points, y_points):
        epsilon = 0.0001
        mask = torch.abs(x_points)[:, 0] > epsilon
        filtered_points = x_points[mask]

        u_x = model(filtered_points)
        u_y = model(y_points)
        u_y = torch.transpose(u_y, dim0=0, dim1=1)
        kernel = kernel_function(filtered_points, y_points)
        volume = 1 ** dim
        inner_integral = (1 / dim) * volume * torch.mean(kernel * u_y, dim=1)

        norm_exp = torch.exp(torch.sum(filtered_points ** 2 / (SIGMA ** 2), dim=1))
        erfi_const = (SIGMA * (math.pi ** 0.5) * erfi(0.5 / SIGMA)) ** (filtered_points.shape[1] - 1)
        sinh_term = torch.sum((SIGMA ** 2) * torch.exp(-1 * filtered_points ** 2 / (SIGMA ** 2)) * (torch.sinh(filtered_points / SIGMA ** 2) / filtered_points), dim=1)
        f = norm_exp - (1 / dim) * erfi_const * sinh_term
        residual = torch.squeeze(u_x) - inner_integral - f
        return residual

    def convolution_func(model):
        device = next(model.parameters()).device
        if method == 'monte_carlo' and sampling == 'uniform':
            x_points = (torch.rand((int(0.5 * num_points), dim)) - 0.5).to(device)
            y_points = (torch.rand((int(0.5 * num_points), dim)) - 0.5).to(device)
            residual1 = calc_residual(model, x_points, y_points)
            ie_loss = torch.mean(residual1 ** 2)
            return ie_loss
        elif method == 'monte_carlo' and sampling == 'sobol':
            soboleng = SobolEngine(dimension=dim)
            soboleng2 = SobolEngine(dimension=dim)
            x_points = soboleng.draw(int(0.5 * num_points)).to(device) - 0.5
            y_points = soboleng2.draw(int(0.5 * num_points)).to(device) - 0.5
            residual1 = calc_residual(model, x_points, y_points)
            ie_loss = torch.mean(residual1 ** 2)
            return ie_loss
        elif method == 'double_sampling' and sampling == 'uniform':
            x_points = (torch.rand((int(0.5 * num_points), dim)) - 0.5).to(device)
            y1_points = (torch.rand((int(0.25 * num_points), dim)) - 0.5).to(device)
            y2_points = (torch.rand((int(0.25 * num_points), dim)) - 0.5).to(device)
            residual1 = calc_residual(model, x_points, y1_points)
            residual2 = calc_residual(model, x_points, y2_points)
            ie_loss = torch.mean(residual1 * residual2)
            return ie_loss
        else:
            return None

    return convolution_func


def get_convolution_loss_func(method, dim=10, num_points=None, device='cpu'):
    if method == 'sps':
        return generate_sps_func(dim=dim, device=device)
    else:
        approximation_method, sampling = method.split('__')[1],  method.split('__')[2]
        print(approximation_method, sampling)
        return generate_pinn_func(dim=dim, num_points=num_points, method=approximation_method, sampling=sampling)


def convolution_eval_func(model, dim=10, num_chunks=None):
    num_eval_points = 100_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=-0.5, range_max=0.5).to(device)
    y = torch.exp(torch.sum(X ** 2 / (SIGMA ** 2), dim=1))
    if num_chunks is None:
        out = model(X)
    else:
        chunk_size = num_eval_points // num_chunks
        all_outs = []
        for cur_chunk in range(num_chunks):
            cur_out = model(X[cur_chunk * chunk_size: min((cur_chunk + 1) * chunk_size, num_eval_points), :])
            all_outs.append(cur_out)
        out = torch.cat(all_outs, dim=0)
    out = torch.squeeze(out)

    l2_relative_error = torch.norm(out - y) / torch.norm(y)
    rmse = torch.sqrt(torch.mean((out - y) ** 2))

    return l2_relative_error, rmse