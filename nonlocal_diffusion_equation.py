from pathlib import Path
import torch
import math
from torch.quasirandom import SobolEngine

from utils import sep_polys_sum_integral, sep_polys_sum_add, sep_polys_sum_multiply, sep_polys_sum_prepend_dims, \
    sep_polys_sum_laplace, sep_polys_sum_remove_trailing_zeros, compute_torch_laplacian, sample_nd_points, lorps_L2_sq


def _set_up_pretrained_terms(num_dims):
    device_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_pretrained_cos = torch.load(Path(__file__).resolve().parent / "projections/cos_1d.pt", map_location=torch.device(device_location))
    cos_1d_params = loaded_pretrained_cos['state_dict']["params"]
    print(cos_1d_params.shape)
    cos_prod = cos_1d_params.repeat(num_dims, num_dims, 1)
    if cos_prod.shape[2] <= 2:
        zeros = torch.zeros(cos_prod.shape[0], cos_prod.shape[0], 2).to(cos_prod.device)
        cos_prod = torch.cat([cos_prod, zeros], dim=2)

    square_sum = torch.zeros((num_dims, num_dims, cos_prod.shape[2])).to(cos_prod.device)
    square_sum[:, :, 0] = 1
    for i in range(num_dims):
        square_sum[i, i, 0] = 0
        square_sum[i, i, 2] = 1

    res = torch.cat([square_sum, cos_prod], dim=1)
    return res, cos_1d_params.repeat(1, num_dims, 1)


def generate_sps_func(dim=10, device='cpu'):
    integrand, _ = _set_up_pretrained_terms(dim)
    square_sum = torch.zeros((dim, dim, 3))
    square_sum[:, :, 0] = 1
    for i in range(dim):
        square_sum[i, i, 0] = 0
        square_sum[i, i, 2] = 1

    boundary_conditions = torch.zeros((1, dim, 3))
    boundary_conditions[0, :, 0] = -1
    boundary_conditions[0, :, 2] = 1

    def nonlocal_diffusion_loss_func(model):
        params = model.params
        cur_boundary_conditions = boundary_conditions.to(params.device)
        params = sep_polys_sum_multiply(params, cur_boundary_conditions)

        prepend_params = sep_polys_sum_prepend_dims(params, num_dims=dim)
        kernel = integrand.to(params.device)
        integral_params = sep_polys_sum_multiply(prepend_params, kernel)

        res_start = torch.ones(dim, device=params.device) * -1
        res_end = torch.ones(dim, device=params.device) * 1
        integral_res_params = sep_polys_sum_integral(integral_params, res_start, res_end, dim_range=(dim, 2 * dim))
        integral_res_params = sep_polys_sum_remove_trailing_zeros(integral_res_params)

        laplace = sep_polys_sum_laplace(params)

        neg_one_const = torch.ones((1, dim, 1), device=params.device)
        neg_one_const[0, 0, 0] = -1

        pi_const = torch.ones((1, dim, 1), device=params.device)
        pi_const[0, 0, 0] = ((math.pi ** 2) / 4) * dim
        pi_params = sep_polys_sum_multiply(params, pi_const)

        cur_square_sum = square_sum.to(params.device)
        neg_cur_square_sum = sep_polys_sum_multiply(cur_square_sum, neg_one_const)

        eq_params = sep_polys_sum_add(laplace, integral_res_params)
        eq_params = sep_polys_sum_add(eq_params, pi_params)
        eq_params = sep_polys_sum_add(eq_params, neg_cur_square_sum)

        start = torch.ones(dim, device=params.device) * -1
        end = torch.ones(dim, device=params.device) * 1
        integral_val = lorps_L2_sq(eq_params, start, end)
        loss_terms = [integral_val]

        loss_terms = torch.stack(loss_terms, dim=0)
        loss = torch.sum(loss_terms)

        return torch.log(loss)

    optimized_func = torch.compile(nonlocal_diffusion_loss_func)
    return optimized_func


def generate_pinn_func(dim=10, num_points=None, method='monte_carlo', sampling='uniform'):
    def kernel_function(x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        d = x.shape[-1]

        indices = torch.arange(d)
        kernel = torch.sum(x[:, :, indices] ** 2 * torch.prod(torch.cos((math.pi / 2) * y), dim=-1, keepdim=True), dim=-1)
        return kernel

    def calc_residual(model, x_points, y_points):
        x_points.requires_grad = True
        u_x = model(x_points)
        laplacian_u_x = compute_torch_laplacian(model, x_points)
        u_y = model(y_points)
        u_y = torch.transpose(u_y, dim0=0, dim1=1)
        kernel = kernel_function(x_points, y_points)
        volume = 2 ** dim

        inner_integral = volume * torch.mean(kernel * u_y, dim=1)
        residual = laplacian_u_x + inner_integral + (math.pi ** 2 / 4) * dim * torch.squeeze(u_x) - torch.sum(x_points ** 2, dim=1)
        return residual

    def nonlocal_diffusion_func(model):
        device = next(model.parameters()).device
        if method == 'monte_carlo' and sampling == 'uniform':
            x_points = (2 * torch.rand((int(0.25 * num_points), dim)) - 1).to(device)
            y_points = (2 * torch.rand((int(0.25 * num_points), dim)) - 1).to(device)
            residual = calc_residual(model, x_points, y_points)
            boundary_points = []
            for i in range(dim):
                points_neg = torch.rand((int((0.25 / dim) * num_points), dim)) * 2 - 1
                points_neg[:, i] = -1

                points_pos = torch.rand((int((0.25 / dim) * num_points), dim)) * 2 - 1
                points_pos[:, i] = 1

                boundary_points.append(points_neg)
                boundary_points.append(points_pos)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual ** 2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        elif method == 'monte_carlo' and sampling == 'sobol':
            soboleng = SobolEngine(dimension=dim)
            soboleng2 = SobolEngine(dimension=dim)
            x_points = 2 * soboleng.draw(int(0.5 * num_points)).to(device) - 1
            y_points = 2 * soboleng2.draw(int(0.5 * num_points)).to(device) - 1
            residual = calc_residual(model, x_points, y_points)
            boundary_points = []
            for i in range(dim):
                points_neg = soboleng.draw(int((0.25 / dim) * num_points)) * 2 - 1
                points_neg[:, i] = -1

                points_pos = soboleng.draw(int((0.25 / dim) * num_points)) * 2 - 1
                points_pos[:, i] = 1

                boundary_points.append(points_neg)
                boundary_points.append(points_pos)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual ** 2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        elif method == 'double_sampling' and sampling == 'uniform':
            x_points = (2 * torch.rand((int(0.5 * num_points), dim)) - 1).to(device)
            y1_points = (2 * torch.rand((int(0.25 * num_points), dim)) - 1).to(device)
            y2_points = (2 * torch.rand((int(0.25 * num_points), dim)) - 1).to(device)
            residual1 = calc_residual(model, x_points, y1_points)
            residual2 = calc_residual(model, x_points, y2_points)
            boundary_points = []
            for i in range(dim):
                points_neg = torch.rand((int((0.25 / dim) * num_points), dim)) * 2 - 1
                points_neg[:, i] = -1

                points_pos = torch.rand((int((0.25 / dim) * num_points), dim)) * 2 - 1
                points_pos[:, i] = 1

                boundary_points.append(points_neg)
                boundary_points.append(points_pos)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual1 * residual2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        else:
            return None

    return nonlocal_diffusion_func


def get_nonlocal_diffusion_loss_func(method, dim=10, num_points=None, device='cpu'):
    if method == 'sps':
        return generate_sps_func(dim=dim, device=device)
    else:
        approximation_method, sampling = method.split('__')[1],  method.split('__')[2]
        print(approximation_method, sampling)
        return generate_pinn_func(dim=dim, num_points=num_points, method=approximation_method, sampling=sampling)


def nonlocal_diffusion_eval_func(model, dim=10, num_chunks=None):
    num_eval_points = 100_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=-1, range_max=1).to(device)
    y = torch.prod(torch.cos((math.pi / 2) * X), dim=1)
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

