import torch
import torch.nn.functional as F
from torch.quasirandom import SobolEngine
torch.set_default_dtype(torch.float64)


def sep_polys_sum(inp, params):
    single_input = (inp.ndim == 1)
    if single_input:
        inp = inp.unsqueeze(0)

    _, d = inp.shape
    s, d2, k = params.shape
    assert d == d2

    powers = torch.arange(k, device=inp.device, dtype=inp.dtype)
    basis = inp.unsqueeze(-1).pow(powers)
    vals = torch.einsum('ndk,sdk->nsd', basis, params)
    out = vals.prod(dim=2).sum(dim=1)

    return out[0] if single_input else out


def sep_polys_sum_integral(params, start_values, end_values, dim_range=None):
    # range dim includes dim_range[0] and is up to but excludes dim_range[1]!!!
    power_dim = params.shape[2]

    if dim_range is None or (dim_range[0] == 0 and dim_range[1] == params.shape[1]):
        params = params / torch.arange(1, power_dim + 1, device=params.device)
        params = torch.nn.functional.pad(params, (1, 0))

        powers = torch.arange(params.shape[2], device=params.device)

        expanded_inp = torch.unsqueeze(start_values, dim=1)
        start_integral_value = torch.sum(torch.pow(expanded_inp, powers) * params, dim=2)

        expanded_inp2 = torch.unsqueeze(end_values, dim=1)
        end_integral_value = torch.sum(torch.pow(expanded_inp2, powers) * params, dim=2)

        total_value = torch.sum(torch.prod(end_integral_value - start_integral_value, dim=1), dim=0)

        return total_value
    else:
        assert dim_range[1] - dim_range[0] == start_values.shape[0]
        cur_params = params[:, dim_range[0]:dim_range[1], :] / torch.arange(1, power_dim + 1, device=params.device)
        cur_params = torch.nn.functional.pad(cur_params, (1, 0))

        powers = torch.arange(cur_params.shape[2], device=params.device)

        expanded_inp = torch.unsqueeze(start_values, dim=1)
        start_integral_value = torch.sum(torch.pow(expanded_inp, powers) * cur_params, dim=2)

        expanded_inp2 = torch.unsqueeze(end_values, dim=1)
        end_integral_value = torch.sum(torch.pow(expanded_inp2, powers) * cur_params, dim=2)

        total_value = torch.prod(end_integral_value - start_integral_value, dim=1)
        total_value = torch.reshape(total_value, (-1, 1, 1))

        if 0 < dim_range[0] and dim_range[1] < params.shape[1]:
            new_params = torch.cat([params[:, :dim_range[0], :], params[:, dim_range[1]:, :]], dim=1)
        elif dim_range[0] == 0:
            new_params = params[:, dim_range[1]:, :].clone()
        else:
            new_params = params[:, :dim_range[0], :].clone()

        new_params[:, 0:1, :] = new_params[:, 0:1, :].clone() * total_value

        return new_params


def sep_polys_sum_prepend_dims(params, num_dims, start_dim=0):
    assert num_dims > 0
    if start_dim == 0:
        const_pads = torch.zeros((params.shape[0], num_dims, params.shape[2]), device=params.device)
        const_pads[:, :, 0] = 1
        new_params = torch.cat([const_pads, params], dim=1)
    else:
        const_pads = torch.zeros((params.shape[0], num_dims, params.shape[2]), device=params.device)
        const_pads[:, :, 0] = 1
        new_params = torch.cat([params[:, :start_dim, :], const_pads, params[:, start_dim:, :]], dim=1)
    return new_params


def sep_polys_sum_multiply(params1, params2):
    padding = params2.shape[2] - 1
    transposed_reshaped_params2 = torch.reshape(torch.transpose(params2, dim0=0, dim1=1), (-1, 1, params2.shape[2])).flip(dims=(2,))
    all_terms = F.conv1d(params1, transposed_reshaped_params2, groups=params2.shape[1], padding=padding)
    expanded_params = torch.reshape(all_terms, (all_terms.shape[0], params2.shape[1], -1, all_terms.shape[2]))
    transposed_new_params = torch.permute(expanded_params, (0, 2, 1, 3))
    new_params = torch.reshape(transposed_new_params, (-1, params2.shape[1], all_terms.shape[2]))
    return new_params


def sep_polys_sum_differentiate(params, dim=None):
    power_dim = params.shape[2]

    if dim is None:
        if power_dim == 1:
            new_params = torch.nn.functional.pad(params, (0, 1)) * torch.arange(end=power_dim+1, device=params.device)
        else:
            new_params = params * torch.arange(end=power_dim)
        new_params = new_params[:, :, 1:]
    else:
        new_params = params.clone()
        if power_dim == 1:
            new_params[:, dim, 0] = 0
        else:
            new_params[:, dim, :] = new_params[:, dim, :] * torch.arange(end=power_dim, device=params.device)
            new_params[:, dim, 0] = 0
            new_params[:, dim, :] = torch.roll(new_params[:, dim, :], shifts=-1, dims=1)

    return new_params


def sep_polys_sum_add(params1, params2):
    # increase sum dim size
    power_dim1 = params1.shape[2]
    power_dim2 = params2.shape[2]

    if power_dim1 > power_dim2:
        padded_params2 = torch.nn.functional.pad(params2, (0, power_dim1 - power_dim2))
        new_params = torch.cat([padded_params2, params1], dim=0)
        return new_params
    elif power_dim1 < power_dim2:
        padded_params1 = torch.nn.functional.pad(params1, (0, power_dim2 - power_dim1))
        new_params = torch.cat([padded_params1, params2], dim=0)
        return new_params
    else:
        new_params = torch.cat([params1, params2], dim=0)
        return new_params


def sep_polys_sum_plug_val(val, dim, params):
    if params.shape[1] == 1:
        return sep_polys_sum(val.clone(), params)

    power_dim = params.shape[2]
    powers = torch.arange(power_dim, device=params.device)

    power_vals = torch.pow(val, powers)
    plugged_val = power_vals * params[:, dim, :]
    summed_plugged_val = torch.sum(plugged_val, dim=1)
    if 0 < dim < params.shape[1] - 1:
        new_params = torch.cat([params[:, :dim, :], params[:, dim + 1:, :]], dim=1)
    elif dim == 0:
        new_params = params[:, 1:, :].clone()
    else:
        new_params = params[:, :params.shape[1] - 1, :].clone()
    summed_plugged_val = torch.reshape(summed_plugged_val, (-1, 1, 1))
    new_params[:, 0:1, :] = new_params[:, 0:1, :].clone() * summed_plugged_val
    return new_params


def sep_polys_sum_laplace(params, exclude_last_dims=0):
    new_params = None
    for cur_dim in range(params.shape[1] - exclude_last_dims):
        cur_deriv = sep_polys_sum_differentiate(params, dim=cur_dim)
        cur_second_deriv = sep_polys_sum_differentiate(cur_deriv, dim=cur_dim)
        if new_params is None:
            new_params = cur_second_deriv
        else:
            new_params = sep_polys_sum_add(new_params, cur_second_deriv)
    return new_params


def sample_nd_points(n_dimensions, num_points, range_min=-1.0, range_max=1.0, use_sobol=False, seed=None):
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    if use_sobol:
        soboleng = SobolEngine(dimension=n_dimensions, seed=seed)
        length = range_max - range_min
        points = soboleng.draw(num_points)
        points = range_min + points * length
    else:
        points = torch.rand(num_points, n_dimensions, generator=gen) * (range_max - range_min) + range_min
    return points


def create_circulant_matrix(vec):
    return torch.cat([f := vec.flip(0), f[:-1]]).unfold(0, len(vec), 1).flip(0)


def compute_torch_laplacian(model, x, d=None):
    u = model(x)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    if d is None:
        laplacian = sum(torch.autograd.grad(grads[:, i], x, grad_outputs=torch.ones_like(grads[:, i]), create_graph=True)[0][:, i]
                        for i in range(x.shape[1]))
    else:
        laplacian = sum(torch.autograd.grad(grads[:, i], x, grad_outputs=torch.ones_like(grads[:, i]), create_graph=True)[0][:, i]
                        for i in range(d))
    return laplacian


def sep_polys_sum_el_wise_multiply(params1: torch.Tensor, params2: torch.Tensor):
    # multiply 2 sps "element-wise" (each polynomial with the corresponding one of the other sps)
    assert params1.shape == params2.shape, f'this operation supports inputs with same shape'
    s, d, k = params1.shape
    params1_reshaped = params1.reshape(1, s * d, k)
    params2_reshaped = params2.reshape(s * d, 1, k)
    params2_flipped = torch.flip(params2_reshaped, dims=[2])
    result = F.conv1d(params1_reshaped, params2_flipped, padding=k-1, groups=s*d)
    return result.reshape(s, d, 2 * k - 1)


def sep_polys_sum_volt_int(params: torch.Tensor, integrated_dim: int, limit_dim: int):
    int_dim_params = params[:, integrated_dim:integrated_dim + 1, :] / torch.arange(1, params.shape[2] + 1, device=params.device)
    limit_dim_params = params[:, limit_dim:limit_dim + 1, :]

    int_dim_params = torch.nn.functional.pad(int_dim_params, (1, 0))
    limit_dim_params = torch.nn.functional.pad(limit_dim_params, (0, 1))

    # we assume integral starts at 0 and ends at limit_dim
    int_params = sep_polys_sum_el_wise_multiply(int_dim_params, limit_dim_params)
    padded_params = torch.nn.functional.pad(params, (0, int_params.shape[2] - params.shape[2]))
    padded_params[:, limit_dim:limit_dim + 1, :] = int_params.clone()

    # remove integrated dim
    if 0 < integrated_dim < params.shape[1] - 1:
        new_params = torch.cat([padded_params[:, :integrated_dim, :], padded_params[:, integrated_dim + 1:, :]], dim=1)
    elif integrated_dim == 0:
        new_params = padded_params[:, 1:, :].clone()
    else:
        new_params = padded_params[:, :padded_params.shape[1] - 1, :].clone()

    return new_params


def sep_polys_sum_remove_trailing_zeros(params):
    reversed_params = torch.flip(params, dims=[-1])
    is_zero = reversed_params == 0
    cumsum_mask = torch.cumprod(is_zero, dim=-1)
    trailing_zeros_per_vec = cumsum_mask.sum(dim=-1)
    min_trailing_zeros = trailing_zeros_per_vec.min()
    new_params = params[..., :params.shape[2] - min_trailing_zeros].clone()
    return new_params


def sep_polys_sum_grad_prod_squared(params):
    new_params = None
    for cur_dim in range(params.shape[1]):
        cur_deriv = sep_polys_sum_differentiate(params, dim=cur_dim)
        squared_deriv = sep_polys_sum_multiply(cur_deriv, cur_deriv)
        if new_params is None:
            new_params = squared_deriv
        else:
            new_params = sep_polys_sum_add(new_params, squared_deriv)
    return new_params


def lorps_L2_sq(params: torch.Tensor,
                start_vals: torch.Tensor,
                end_vals: torch.Tensor) -> torch.Tensor:
    S, D, K = params.shape
    device = params.device
    dtype = params.dtype
    ks = torch.arange(K, device=device)
    k_idx = ks.view(-1, 1)
    l_idx = ks.view(1, -1)
    sum_idx = k_idx + l_idx
    m = torch.arange(2 * K - 1, device=device, dtype=dtype)
    m1 = m + 1
    a = start_vals.to(dtype).view(D, 1)
    b = end_vals.to(dtype).view(D, 1)
    w = (b ** m1 - a ** m1) / m1
    M_all = w[:, sum_idx]
    A = params.permute(1, 0, 2)
    B = torch.matmul(A, M_all)
    I = torch.matmul(B, A.transpose(-1, -2))
    G = I.prod(dim=0)
    return G.sum()

