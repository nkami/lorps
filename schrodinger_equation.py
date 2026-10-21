import torch

from utils import sep_polys_sum_integral, sep_polys_sum_add, sep_polys_sum_multiply, sep_polys_sum, \
    sep_polys_sum_laplace, sep_polys_sum_grad_prod_squared, sample_nd_points


DOMAIN_ABS_EDGE = 3
# EDGE_VAL = 2
import math
EDGE_VAL = math.exp(-1 * (DOMAIN_ABS_EDGE ** 2 / 2))


# works as infinite well on [-L, L]
def generate_sps_func(dim=10, device='cpu'):
    lin_sum = torch.zeros((dim, dim, 3))
    lin_sum[:, :, 0] = 1
    for i in range(dim):
        lin_sum[i, i, 0] = 0
        lin_sum[i, i, 2] = 1

    boundary_conditions = torch.zeros((1, dim, 3))
    boundary_conditions[0, :, 0] = -1 * DOMAIN_ABS_EDGE ** 2
    boundary_conditions[0, :, 2] = DOMAIN_ABS_EDGE ** 2
    # tmp_bound = sep_polys_sum_multiply(boundary_conditions, boundary_conditions)
    # for i in range(2):
    #     tmp_bound = sep_polys_sum_multiply(tmp_bound, boundary_conditions)
    # boundary_conditions = tmp_bound

    def schrodinger_loss_func(model):
        params = model.params
        cur_boundary_conditions = boundary_conditions.to(params.device)
        params = sep_polys_sum_multiply(params, cur_boundary_conditions)

        start = torch.ones(dim, device=params.device) * -1
        end = torch.ones(dim, device=params.device) * 1

        grad_params = sep_polys_sum_grad_prod_squared(params)
        grad_int = sep_polys_sum_integral(grad_params, start, end) * (DOMAIN_ABS_EDGE ** (-2))

        squared_params = sep_polys_sum_multiply(params, params)
        squared_int = sep_polys_sum_integral(squared_params, start, end)

        tot_loss = grad_int / squared_int

        # norm_params = sep_polys_sum_multiply(params, params)
        # norm_integral = sep_polys_sum_integral(norm_params, start, end) * (DOMAIN_ABS_EDGE ** dim)
        # normalization_penalty = (norm_integral - 1) ** 2

        loss_terms = [tot_loss]
        loss_terms = torch.stack(loss_terms, dim=0)
        loss = torch.sum(loss_terms)

        return loss

    optimized_func = torch.compile(schrodinger_loss_func)
    return optimized_func


# def generate_sps_func(dim=10):
#     lin_sum = torch.zeros((dim, dim, 3))
#     lin_sum[:, :, 0] = 1
#     for i in range(dim):
#         lin_sum[i, i, 0] = 0
#         lin_sum[i, i, 2] = 1
#
#     boundary_conditions = torch.zeros((1, dim, 3))
#     # boundary_conditions[0, :, 0] = -1 * DOMAIN_ABS_EDGE ** 2
#     # boundary_conditions[0, :, 2] = DOMAIN_ABS_EDGE ** 2
#
#     boundary_conditions[0, :, 0] = -1 * DOMAIN_ABS_EDGE ** 2
#     boundary_conditions[0, :, 2] = DOMAIN_ABS_EDGE ** 2
#
#     boundary_conditions2 = torch.zeros((1, dim, 1))
#     boundary_conditions2[0, 0, 0] = EDGE_VAL
#
#     def schrodinger_loss_func(model):
#         params = model.params
#         cur_boundary_conditions = boundary_conditions.to(params.device)
#         cur_boundary_conditions2 = boundary_conditions2.to(params.device)
#         params = sep_polys_sum_multiply(params, cur_boundary_conditions)
#         params = sep_polys_sum_add(params, cur_boundary_conditions2)
#
#         start = torch.ones(dim, device=params.device) * -1
#         end = torch.ones(dim, device=params.device) * 1
#
#         grad_params = sep_polys_sum_grad_prod_squared(params)
#         grad_int = sep_polys_sum_integral(grad_params, start, end) * (DOMAIN_ABS_EDGE ** (-2))
#
#         cur_lin_sum = lin_sum.to(params.device)
#         squared_params = sep_polys_sum_multiply(params, params)
#         cur_square_sum_prod = sep_polys_sum_multiply(cur_lin_sum, squared_params)
#         pot_int = sep_polys_sum_integral(cur_square_sum_prod, start, end) * (DOMAIN_ABS_EDGE ** 2)
#
#         squared_int = sep_polys_sum_integral(squared_params, start, end)
#
#         # print(grad_int, pot_int, squared_int)
#         tot_loss = (grad_int + pot_int) / squared_int
#
#         norm_params = sep_polys_sum_multiply(params, params)
#         norm_integral = sep_polys_sum_integral(norm_params, start, end) * (DOMAIN_ABS_EDGE ** dim)
#         normalization_penalty = (norm_integral - 1) ** 2
#
#         loss_terms = [tot_loss]
#         # print(tot_loss)
#         loss_terms = torch.stack(loss_terms, dim=0)
#         loss = torch.sum(loss_terms)
#
#         return loss
#
#     # optimized_func = torch.compile(schrodinger_loss_func)
#     return schrodinger_loss_func


def get_schrodinger_loss_func(method, dim=10, num_points=None, device='cpu'):

    if method == 'sps':
        return generate_sps_func(dim=dim, device=device)
    else:
        return None


def schrodinger_eval_func(model, dim=10, num_chunks=None):
    # return torch.Tensor([0]), torch.Tensor([0])
    num_eval_points = 100_000
    device = next(model.parameters()).device
    X = sample_nd_points(n_dimensions=dim, num_points=num_eval_points, range_min=-DOMAIN_ABS_EDGE, range_max=DOMAIN_ABS_EDGE).to(device)
    y = torch.prod(torch.special.airy_ai(X - 2.3381), dim=1)
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

    if dim == 1:
        import matplotlib.pyplot as plt
        # Create the plot
        plt.figure()
        x_axis = X.squeeze().cpu().numpy()
        y_axis = out.squeeze().cpu().numpy()

        import numpy as np
        from scipy import special
        sorted_indices = np.argsort(x_axis)
        x_sorted = x_axis[sorted_indices]
        y_sorted = y_axis[sorted_indices]

        x_axis = x_sorted[::100]
        y_axis = y_sorted[::100]
        print(type(x_axis), x_axis.shape)
        print(type(y_axis), y_axis.shape)

        # y_cos, _, _, _ = special.airy(x_axis - 2.3381)
        y_cos = np.exp(-1 * x_axis ** 2 / 2)
        # print(type(y_cos), y_cos.shape)

        # Create the plot
        plt.figure()
        plt.plot(x_axis, y_axis, label='Model Output')
        plt.plot(x_axis, y_cos, label='cos(pi/2 * x)', linestyle='--')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Comparison Plot')
        plt.legend()

        # Save the figure
        plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        import time
        time.sleep(1)


    return l2_relative_error, rmse



if __name__ == '__main__':
    pass




