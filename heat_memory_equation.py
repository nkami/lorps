from pathlib import Path
import torch
import torch.nn.functional as F
import math
from scipy.special import erfi
from torch.quasirandom import SobolEngine

from utils import sep_polys_sum_integral, sep_polys_sum_add, sep_polys_sum_multiply, sep_polys_sum_prepend_dims, \
    sample_nd_points, sep_polys_sum_remove_trailing_zeros, sep_polys_sum_laplace, sep_polys_sum_volt_int, \
    sep_polys_sum_differentiate, compute_torch_laplacian, lorps_L2_sq

Q = 1


def _set_up_pretrained_terms(num_dims, q=None):
    q = Q if q is None else q
    device_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaded_pretrained_sin = torch.load(
        Path(__file__).resolve().parent / "projections/sin_1d.pt",
        map_location=torch.device(device_location)
    )
    sin_1d_params = loaded_pretrained_sin['state_dict']["params"]

    if q != 1:
        loaded_pretrained_sin_q = torch.load(
            Path(__file__).resolve().parent / f"projections/sin_1d_Q_{q}.pt",
            map_location=torch.device(device_location)
        )
        sin_q_params = loaded_pretrained_sin_q['state_dict']["params"]

        # Pad whichever projection has the lower polynomial degree.
        max_k = max(sin_1d_params.shape[-1], sin_q_params.shape[-1])
        sin_1d_params = F.pad(
            sin_1d_params,
            (0, max_k - sin_1d_params.shape[-1])
        )
        sin_q_params = F.pad(
            sin_q_params,
            (0, max_k - sin_q_params.shape[-1])
        )

    sin_prod = sin_1d_params.repeat(1, num_dims, 1)

    if q != 1:
        sin_prod[:, 0, :] = sin_q_params[:, 0, :]

    print(f'SIN PRODUCT SHAPE: {sin_prod.shape}')
    return sin_prod


def generate_sps_func(dim=10, device='cpu', q=None):
    q = Q if q is None else q

    # boundary and initial conditions: prod * t * u
    boundary_poly_prod = torch.zeros((1, dim, 3))
    boundary_poly_prod[0, :-1, 1] = 3
    boundary_poly_prod[0, :-1, 2] = -3
    boundary_poly_prod[0, -1, 1] = 1

    sin_prod = _set_up_pretrained_terms(num_dims=dim, q=q)
    if sin_prod.shape[2] <= 3:
        import torch.nn.functional as F
        pad = max(0, 4 - sin_prod.shape[-1])
        sin_prod = F.pad(sin_prod, (0, pad), mode="constant", value=0.0)

    # f term (last dim is t):
    sin_prod[:, -1, :] = 0
    sin_prod[:, -1, 1] = 2
    sin_prod[:, -1, 3] = ((math.pi ** 2) * (q ** 2 + dim - 2)) / 3


    def heat_memory_loss_func(model):
        params = model.params

        cur_boundary_poly_prod = boundary_poly_prod.to(params.device)

        # initial and boundary conditions
        params = sep_polys_sum_multiply(params, cur_boundary_poly_prod)

        prepend_params = sep_polys_sum_prepend_dims(params, num_dims=1, start_dim=dim-1)
        laplace_prepend_params = sep_polys_sum_laplace(prepend_params, exclude_last_dims=2)
        integral_res_params = sep_polys_sum_volt_int(laplace_prepend_params, integrated_dim=dim, limit_dim=dim-1)
        integral_res_params = sep_polys_sum_remove_trailing_zeros(integral_res_params)

        cur_f = sin_prod.to(params.device)

        neg_one_const = torch.ones((1, dim, 1), device=params.device)
        neg_one_const[0, 0, 0] = -1

        t_deriv_params = sep_polys_sum_differentiate(params, dim=dim-1)
        neg_t_deriv_params = sep_polys_sum_multiply(t_deriv_params, neg_one_const)

        eq_params = sep_polys_sum_add(integral_res_params, cur_f)
        eq_params = sep_polys_sum_add(eq_params, neg_t_deriv_params)

        start = torch.ones(dim, device=params.device) * 0
        end = torch.ones(dim, device=params.device) * 1
        integral_val = lorps_L2_sq(eq_params, start, end)
        loss_terms = [integral_val]

        loss_terms = torch.stack(loss_terms, dim=0)
        loss = torch.sum(loss_terms)

        return torch.log(loss)

    optimized_func = torch.compile(heat_memory_loss_func)
    return optimized_func


def generate_pinn_func(dim=10, num_points=None, method='monte_carlo', sampling='uniform', q=None):
    q = Q if q is None else q
    def calc_residual(model, x_points, t_points, y_time_points):
        N, d = x_points.shape
        M = y_time_points.shape[0]

        x_points.requires_grad_(True)
        y_time_points.requires_grad_(True)
        t_points.requires_grad_(True)

        z_xt = torch.cat([x_points, t_points], dim=1)
        u_xt = model(z_xt).squeeze(-1)

        du_dt = torch.autograd.grad(
            outputs=u_xt,
            inputs=t_points,
            grad_outputs=torch.ones_like(u_xt),
            retain_graph=True,
            create_graph=True
        )[0].squeeze(-1)

        t_matrix = t_points.repeat(1, M)
        s_matrix = y_time_points.T
        scaled_times = s_matrix * t_matrix

        x_repeat = x_points.unsqueeze(1).expand(N, M, d).reshape(N * M, d)
        s_repeat = scaled_times.reshape(N * M, 1)
        z_samples = torch.cat([x_repeat, s_repeat], dim=1)

        lap_u = compute_torch_laplacian(model, z_samples, d=d).view(N, M)

        memory_integral = t_points.squeeze() * torch.mean(lap_u, dim=1)

        t = t_points.squeeze()
        spatial_product = torch.sin(q * math.pi * x_points[:, 0]) * torch.prod(torch.sin(math.pi * x_points[:, 1:]), dim=1)
        f_xt = (2 * t + ((math.pi ** 2) / 3) * (q ** 2 + d - 1) * t ** 3) * spatial_product

        residual = du_dt - memory_integral - f_xt
        return residual

    def heat_func(model):
        device = next(model.parameters()).device
        if method == 'monte_carlo' and sampling == 'uniform':
            x_points = torch.rand((int(0.25 * num_points), dim)).to(device)
            t_points = x_points[:, -1:]
            spatial_points = x_points[:, :-1]
            y_points = torch.rand((int(0.25 * num_points), 1)).to(device)
            residual = calc_residual(model, spatial_points, t_points, y_points)
            boundary_points = []
            for i in range(dim):
                points_neg = torch.rand((int((0.25 / dim) * num_points), dim))
                points_neg[:, i] = 0

                if i < dim - 1:
                    points_pos = torch.rand((int((0.25 / dim) * num_points), dim))
                    points_pos[:, i] = 1
                    boundary_points.append(points_pos)

                boundary_points.append(points_neg)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual ** 2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        elif method == 'monte_carlo' and sampling == 'sobol':
            soboleng = SobolEngine(dimension=dim)
            soboleng2 = SobolEngine(dimension=1)
            x_points = soboleng.draw(int(0.25 * num_points)).to(device)
            t_points = x_points[:, -1:]
            spatial_points = x_points[:, :-1]
            y_points = soboleng2.draw(int(0.25 * num_points)).to(device)
            residual = calc_residual(model, spatial_points, t_points, y_points)
            boundary_points = []
            for i in range(dim):
                points_neg = soboleng.draw(int((0.25 / dim) * num_points))
                points_neg[:, i] = 0

                if i < dim - 1:
                    points_pos = soboleng.draw(int((0.25 / dim) * num_points))
                    points_pos[:, i] = 1
                    boundary_points.append(points_pos)

                boundary_points.append(points_neg)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual ** 2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        elif method == 'double_sampling' and sampling == 'uniform':
            x_points = torch.rand((int(0.25 * num_points), dim)).to(device)  # Random points in [0, 1]^dim
            t_points = x_points[:, -1:]
            spatial_points = x_points[:, :-1]
            y1_points = torch.rand((int(0.125 * num_points), 1)).to(device)  # Random points in [0, 1]^dim
            y2_points = torch.rand((int(0.125 * num_points), 1)).to(device)  # Random points in [0, 1]^dim
            residual1 = calc_residual(model, spatial_points, t_points, y1_points)
            residual2 = calc_residual(model, spatial_points, t_points, y2_points)
            boundary_points = []
            for i in range(dim):
                points_neg = torch.rand((int((0.25 / dim) * num_points), dim))
                points_neg[:, i] = 0

                if i < dim - 1:
                    points_pos = torch.rand((int((0.25 / dim) * num_points), dim))
                    points_pos[:, i] = 1
                    boundary_points.append(points_pos)

                boundary_points.append(points_neg)
            chosen_boundary_points = torch.cat(boundary_points).to(device)
            u_x = model(chosen_boundary_points)
            ie_loss = torch.mean(residual1 * residual2) + torch.mean(torch.squeeze(u_x) ** 2)
            return ie_loss
        else:
            return None

    return heat_func


def get_heat_memory_loss_func(method, dim=10, num_points=None, device='cpu', q=None):
    q = Q if q is None else q
    assert dim > 1, f'dim must be at least 2'
    print(f'using heat frequency Q={q}')

    if method == 'sps':
        return generate_sps_func(dim=dim, device=device, q=q)
    else:
        approximation_method, sampling = method.split('__')[1],  method.split('__')[2]
        print(approximation_method, sampling)
        return generate_pinn_func(dim=dim, num_points=num_points, method=approximation_method, sampling=sampling, q=q)


def heat_memory_eval_func(model, dim=10, num_chunks=None, q=None):
    q = Q if q is None else q
    num_eval_points = 100_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=0, range_max=1).to(device)
    y = X[:, -1] ** 2 * torch.sin(q * math.pi * X[:, 0]) * torch.prod(torch.sin(math.pi * X[:, 1:-1]), dim=1)
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


def heat_memory_loss_func(model, dim=10):
    num_eval_points = 10_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=-0.1, range_max=1.1, use_sobol=True).to(device)
    y = X[:, -1] ** 2 * torch.prod(X[:, :-1] * (1 - X[:, :-1]), dim=1)
    out = model(X)
    out = torch.squeeze(out)
    l2_relative_error = torch.norm(out - y) / (torch.norm(y) + 1e-5)
    loss = torch.mean((out - y) ** 2)
    return l2_relative_error, loss

