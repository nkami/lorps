import torch
from torch.quasirandom import SobolEngine

from utils import sep_polys_sum_integral, sep_polys_sum_add, sep_polys_sum_multiply, sep_polys_sum, \
    create_circulant_matrix, sample_nd_points, sep_polys_sum_prepend_dims, \
    sep_polys_sum_remove_trailing_zeros, lorps_L2_sq


def generate_sps_func(dim=10, device='cpu', alpha=5):
    diagonal_values = torch.ones(dim)
    diagonal_matrix = torch.diag(diagonal_values)
    base_vec = torch.zeros(dim)
    base_vec[0] = -1
    base_vec[1] = 1
    circulant_mat = create_circulant_matrix(base_vec).T
    lin_row = torch.cat([circulant_mat[1:, :], circulant_mat[0:1, :]], dim=0)
    y_kernel = torch.stack([torch.logical_not(diagonal_matrix + lin_row), lin_row,
                            torch.zeros_like(diagonal_matrix), diagonal_matrix], dim=2)

    combined_linear_part = torch.ones((dim, 2 * dim, 2))
    combined_linear_part[:, :, 1] = 0
    for i in range(dim):
        combined_linear_part[i, dim + i, 0] = 0
        combined_linear_part[i, dim + i, 1] = -1 * alpha

    eye = torch.eye(dim) * 1
    x_kernel = torch.stack([torch.logical_not(eye), eye * 1,
                            torch.zeros_like(eye), torch.zeros_like(eye)], dim=2)
    combined_kernel = torch.cat([x_kernel, y_kernel], dim=1)
    combined_kernel = sep_polys_sum_add(combined_kernel, combined_linear_part)

    def fredholm_loss_func(model):
        params = model.params
        prepend_params = sep_polys_sum_prepend_dims(params, num_dims=dim)
        kernel = combined_kernel.to(params.device)
        integral_params = sep_polys_sum_multiply(prepend_params, kernel)

        res_start = torch.ones(dim, device=params.device) * -0.5
        res_end = torch.ones(dim, device=params.device) * 0.5
        integral_res_params = sep_polys_sum_integral(integral_params, res_start, res_end, dim_range=(dim, 2 * dim))
        integral_res_params = sep_polys_sum_remove_trailing_zeros(integral_res_params)

        neg_one_const = torch.ones((1, dim, 1), device=params.device)
        neg_one_const[0, 0, 0] = -1

        eq_params = sep_polys_sum_add(params, integral_res_params)
        eq_params = sep_polys_sum_add(eq_params, neg_one_const)

        start = torch.ones(dim, device=params.device) * -0.5
        end = torch.ones(dim, device=params.device) * 0.5
        integral_val = lorps_L2_sq(eq_params, start, end)
        loss_terms = [integral_val]

        loss_terms = torch.stack(loss_terms, dim=0)
        loss = torch.sum(loss_terms)

        return torch.log(loss)

    optimized_func = torch.compile(fredholm_loss_func)
    return optimized_func


def generate_pinn_func(dim=10, num_points=None, method='monte_carlo', sampling='uniform', alpha=5):
    def kernel_function(x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        d = x.shape[-1]

        indices = torch.arange(d)
        kernel = torch.sum(x[:, :, indices] * y[:, :, indices] ** 3 *
                           y[:, :, (indices + 1) % d] * y[:, :, (indices + 2) % d] + alpha * y[:, :, indices], dim=-1)
        return kernel

    def calc_residual(model, x_points, y_points):
        u_x = model(x_points)
        u_y = model(y_points)
        u_y = torch.transpose(u_y, dim0=0, dim1=1)
        kernel = kernel_function(x_points, y_points)
        volume = 1 ** dim
        inner_integral = volume * torch.mean(kernel * u_y, dim=1)
        residual = torch.squeeze(u_x) - inner_integral - 1
        return residual

    def fredholm_loss_func(model):
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

    return fredholm_loss_func


def get_fredholm_loss_func(method, dim=10, num_points=None, device='cpu', alpha=5):
    print(f'using alpha {alpha}')

    if method == 'sps':
        return generate_sps_func(dim=dim, device=device, alpha=alpha)
    else:
        approximation_method, sampling = method.split('__')[1],  method.split('__')[2]
        print(approximation_method, sampling)
        return generate_pinn_func(dim=dim, num_points=num_points, method=approximation_method, sampling=sampling, alpha=alpha)


def fredholm_eval_func(model, dim=10, num_chunks=None):
    num_eval_points = 100_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=-0.5, range_max=0.5).to(device)
    y = torch.ones(X.shape[0]).to(device)
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
