"""LoRPS benchmark for a nonseparable constant-kernel integral equation.

The benchmark on the unit cube is

    -Delta_z u(z) + eta * integral_[0,1]^d u(y) dy = 1,
    u(z) = 0 on the boundary.

The kernel K(z, y) = 1 is constant.  If v is the Dirichlet torsion
function, -Delta v = 1, and A_d = integral v, then

    u = v / (1 + eta * A_d).

For d >= 2, v has an infinite separated sine expansion.  The reference
solution below evaluates the equivalent heat-semigroup integral, rather
than truncating an exponentially large multi-index sum.

Training is performed after the change of variables z = (x + 1) / 2,
with x in [-1, 1]^d.  In these coordinates the equation is

    -4 * Delta_x u(x) + eta * mean_[ -1,1 ]^d(u) = 1.

This lets the benchmark reuse the hard zero-boundary construction of the
project's ``SPS(..., equation='nonlocal_diffusion')`` model.  The LoRPS
residual norm and the integral term are both evaluated analytically.

Example:

    python constant_kernel_benchmark.py \
        --dimensions 2 5 10 20 --ranks 1 2 4 8 16 32 --k 10

For a fast installation and reference check, run:

    python constant_kernel_benchmark.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from torch.optim import Adam, LBFGS

from models import SPS
from utils import (
    lorps_L2_sq,
    sample_nd_points,
    sep_polys_sum_add,
    sep_polys_sum_integral,
    sep_polys_sum_laplace,
    sep_polys_sum_multiply,
)


torch.set_default_dtype(torch.float64)


@dataclass
class ReferenceInfo:
    """Dimension-specific information computed with the reference solution."""

    dimension: int
    eta: float
    torsion_integral: float
    denominator: float
    quadrature_order: int
    sine_modes: int


@dataclass
class RunResult:
    dimension: int
    rank: int
    k: int
    seed: int
    eta: float
    num_parameters: int
    log_residual_mse: float
    residual_mse: float
    relative_l2_error: float
    rmse: float
    train_seconds: float
    lbfgs_outer_steps: int
    stopped_early: bool
    peak_memory_mb: float
    torsion_integral: float
    reference_denominator: float
    eval_points: int


def _transformed_quadrature(dimension: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights for an integral over [0, infinity)."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if order < 8:
        raise ValueError("reference quadrature order must be at least 8")

    nodes, weights = leggauss(order)
    z = 0.5 * (nodes + 1.0)
    z_weights = 0.5 * weights

    # The first Dirichlet eigenvalue of the d-dimensional cube is pi^2 d.
    # This scale makes the rational map resolve the important time range in
    # every dimension while still covering the infinite tail.
    time_scale = 1.0 / (math.pi**2 * dimension)
    times = time_scale * z / (1.0 - z)
    time_weights = z_weights * time_scale / (1.0 - z) ** 2
    return times, time_weights


def _one_dimensional_heat_solution(
    points: np.ndarray,
    times: np.ndarray,
    sine_modes: int,
    image_terms: int,
    image_switch_time: float,
) -> np.ndarray:
    """Evaluate the 1-D heat solution with initial value one and zero boundary."""

    points = np.asarray(points, dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.empty((points.size, times.size), dtype=np.float64)

    small = times < image_switch_time
    if np.any(small):
        small_times = times[small]
        sigma = np.sqrt(2.0 * small_times)[None, :]
        x = points[:, None]
        image_values = np.zeros((points.size, small_times.size), dtype=np.float64)
        for image_index in range(-image_terms, image_terms + 1):
            shifted_x = x + 2.0 * image_index
            image_values += (
                ndtr((1.0 - shifted_x) / sigma)
                - ndtr(-shifted_x / sigma)
                - ndtr((1.0 + shifted_x) / sigma)
                + ndtr(shifted_x / sigma)
            )
        values[:, small] = image_values

    if np.any(~small):
        odd_modes = np.arange(1, 2 * sine_modes, 2, dtype=np.float64)
        basis = np.sin(math.pi * points[:, None] * odd_modes[None, :]) / odd_modes[None, :]
        decay = np.exp(
            -(math.pi * odd_modes[:, None]) ** 2 * times[None, ~small]
        )
        values[:, ~small] = (4.0 / math.pi) * (basis @ decay)

    # Roundoff in the image and sine representations can produce values just
    # outside the probabilistic range of this heat solution.
    return np.clip(values, 0.0, 1.0)


def reference_solution(
    unit_cube_points: np.ndarray,
    eta: float = 1.0,
    quadrature_order: int = 160,
    sine_modes: int = 128,
    image_terms: int = 3,
    image_switch_time: float = 0.02,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, ReferenceInfo]:
    """Evaluate the nonseparable reference solution on points in [0, 1]^d.

    The torsion function is represented as

        v(z) = integral_0^infinity product_j h(t, z_j) dt,

    where h is the one-dimensional zero-boundary heat solution with initial
    value one.  This is mathematically equivalent to the infinite separated
    sine series, but costs O(N d M Q) instead of enumerating M^d modes.
    """

    points = np.asarray(unit_cube_points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("unit_cube_points must have shape (num_points, dimension)")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("reference points must lie in [0, 1]^d")
    if sine_modes < 8:
        raise ValueError("sine_modes must be at least 8")
    if image_terms < 1:
        raise ValueError("image_terms must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    dimension = points.shape[1]
    times, weights = _transformed_quadrature(dimension, quadrature_order)

    odd_modes = np.arange(1, 2 * sine_modes, 2, dtype=np.float64)
    integrated_heat = (8.0 / math.pi**2) * np.sum(
        np.exp(-(math.pi * odd_modes[:, None]) ** 2 * times[None, :])
        / odd_modes[:, None] ** 2,
        axis=0,
    )
    torsion_integral = float(np.dot(weights, integrated_heat**dimension))
    denominator = 1.0 + eta * torsion_integral

    output = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        flat_points = points[start:stop].reshape(-1)
        heat_values = _one_dimensional_heat_solution(
            flat_points,
            times,
            sine_modes=sine_modes,
            image_terms=image_terms,
            image_switch_time=image_switch_time,
        )
        heat_values = heat_values.reshape(stop - start, dimension, quadrature_order)
        torsion_values = np.dot(np.prod(heat_values, axis=1), weights)
        output[start:stop] = torsion_values / denominator

    info = ReferenceInfo(
        dimension=dimension,
        eta=eta,
        torsion_integral=torsion_integral,
        denominator=denominator,
        quadrature_order=quadrature_order,
        sine_modes=sine_modes,
    )
    return output, info


def solution_params(model: SPS) -> torch.Tensor:
    """Return the polynomial parameters after imposing zero boundary values."""

    dimension = model.params.shape[1]
    boundary = torch.zeros(
        (1, dimension, 3), device=model.params.device, dtype=model.params.dtype
    )
    boundary[0, :, 0] = -1.0
    boundary[0, :, 2] = 1.0
    return sep_polys_sum_multiply(model.params, boundary)


def residual_params(model: SPS, eta: float) -> torch.Tensor:
    """Construct LoRPS parameters for -4 Delta u + eta mean(u) - 1."""

    params = solution_params(model)
    dimension = params.shape[1]
    lower = torch.full(
        (dimension,), -1.0, device=params.device, dtype=params.dtype
    )
    upper = torch.full(
        (dimension,), 1.0, device=params.device, dtype=params.dtype
    )

    laplacian = sep_polys_sum_laplace(params)
    minus_four = torch.ones(
        (1, dimension, 1), device=params.device, dtype=params.dtype
    )
    minus_four[0, 0, 0] = -4.0
    negative_laplacian = sep_polys_sum_multiply(laplacian, minus_four)

    volume = float(2**dimension)
    mean_u = sep_polys_sum_integral(params, lower, upper) / volume
    scalar = (eta * mean_u - 1.0).reshape(1, 1, 1)
    if dimension == 1:
        constant = scalar
    else:
        constant = torch.cat(
            [
                scalar,
                torch.ones(
                    (1, dimension - 1, 1),
                    device=params.device,
                    dtype=params.dtype,
                ),
            ],
            dim=1,
        )
    return sep_polys_sum_add(negative_laplacian, constant)


def residual_mean_square(model: SPS, eta: float) -> torch.Tensor:
    """Compute the exact mean squared equation residual."""

    params = residual_params(model, eta)
    dimension = params.shape[1]
    lower = torch.full(
        (dimension,), -1.0, device=params.device, dtype=params.dtype
    )
    upper = torch.full(
        (dimension,), 1.0, device=params.device, dtype=params.dtype
    )
    return lorps_L2_sq(params, lower, upper) / float(2**dimension)


def log_residual_loss(model: SPS, eta: float) -> torch.Tensor:
    mse = residual_mean_square(model, eta)
    return torch.log(torch.clamp(mse, min=torch.finfo(mse.dtype).tiny))


def initialize_model(model: SPS, dimension: int, rank: int, noise: float) -> None:
    """Initialize non-vanishing, symmetry-compatible separated factors."""

    with torch.no_grad():
        model.params.zero_()
        model.params[:, :, 0] = 1.0

        # Since the model multiplies every factor by (x_j^2 - 1), this sign
        # makes the initial solution positive in the interior.  Its magnitude
        # is distributed over the rank terms without exponentially small
        # products in high dimension.
        model.params[:, 0, 0] = ((-1.0) ** dimension) * 0.05 / rank

        # The target and domain are coordinate-wise even.  Small random even
        # coefficients break rank-term symmetry while respecting that fact.
        for degree in range(2, model.params.shape[2], 2):
            scale = noise / (degree + 1)
            model.params[:, :, degree].normal_(mean=0.0, std=scale)


def evaluate_model(
    model: SPS,
    transformed_points: torch.Tensor,
    reference_values: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[float, float]:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, transformed_points.shape[0], chunk_size):
            stop = min(start + chunk_size, transformed_points.shape[0])
            points = transformed_points[start:stop].to(device)
            outputs.append(model(points).detach().cpu())
    prediction = torch.cat(outputs)
    error = prediction - reference_values
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference_values)
    rmse = torch.sqrt(torch.mean(error**2))
    return float(relative_l2), float(rmse)


def _clone_state_dict(model: SPS) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one(
    dimension: int,
    rank: int,
    k: int,
    seed: int,
    eta: float,
    device: torch.device,
    adam_steps: int,
    learning_rate: float,
    max_duration_minutes: float,
    lbfgs_steps: int | None,
    lbfgs_max_iter: int,
    lbfgs_history: int,
    lbfgs_tolerance_grad: float,
    lbfgs_tolerance_change: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    gradient_clip: float,
    init_noise: float,
    progress_every: int,
) -> tuple[SPS, float, float, float, int, bool]:
    """Train one LoRPS configuration and restore its best residual iterate."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = SPS(s=rank, d=dimension, k=k, equation="nonlocal_diffusion").to(device)
    initialize_model(model, dimension=dimension, rank=rank, noise=init_noise)
    model.train()

    best_loss = math.inf
    best_state = _clone_state_dict(model)
    start_time = time.perf_counter()
    lbfgs_outer_steps = 0
    stopped_early = False

    def record_best(loss_value: float) -> None:
        nonlocal best_loss, best_state
        if math.isfinite(loss_value) and loss_value < best_loss:
            best_loss = loss_value
            best_state = _clone_state_dict(model)

    adam = Adam(model.parameters(), lr=learning_rate)
    for step in range(1, adam_steps + 1):
        adam.zero_grad()
        loss = log_residual_loss(model, eta)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite Adam loss for d={dimension}, s={rank}, step={step}"
            )
        value = float(loss.detach())
        record_best(value)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        adam.step()
        if progress_every > 0 and (step == 1 or step % progress_every == 0):
            print(f"  Adam {step:5d}/{adam_steps}: log residual MSE = {value:.6e}")

    if adam_steps > 0:
        record_best(float(log_residual_loss(model, eta).detach()))

    if lbfgs_steps != 0:
        lbfgs = LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=lbfgs_tolerance_grad,
            tolerance_change=lbfgs_tolerance_change,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            closure_loss = log_residual_loss(model, eta)
            if not torch.isfinite(closure_loss):
                raise FloatingPointError(
                    f"non-finite L-BFGS loss for d={dimension}, s={rank}"
                )
            closure_loss.backward()
            return closure_loss

        step = 0
        stale_steps = 0
        early_stop_reference = float(log_residual_loss(model, eta).detach())
        max_duration_seconds = 60.0 * max_duration_minutes
        while time.perf_counter() - start_time < max_duration_seconds:
            if lbfgs_steps is not None and step >= lbfgs_steps:
                break
            step += 1
            lbfgs.step(closure)
            value = float(log_residual_loss(model, eta).detach())
            record_best(value)

            # PyTorch's convergence checks terminate only the current
            # LBFGS.step() call.  Detect a plateau across outer calls so the
            # wall-clock loop does not repeatedly invoke an already converged
            # optimizer.  Improvements accumulate against a fixed reference;
            # several individually tiny improvements can therefore still
            # reset the patience counter once their sum exceeds min_delta.
            if early_stop_reference - value > early_stop_min_delta:
                early_stop_reference = value
                stale_steps = 0
            else:
                stale_steps += 1

            if progress_every > 0 and (step == 1 or step % progress_every == 0):
                elapsed_minutes = (time.perf_counter() - start_time) / 60.0
                print(
                    f"  L-BFGS {step:5d}: log residual MSE = {value:.6e} "
                    f"({elapsed_minutes:.1f}/{max_duration_minutes:g} min)"
                )
            if early_stop_patience > 0 and stale_steps >= early_stop_patience:
                stopped_early = True
                print(
                    f"  Early stopping L-BFGS after {step} outer steps: "
                    f"log residual improved by at most {early_stop_min_delta:.1e} "
                    f"during {early_stop_patience} consecutive checks"
                )
                break
        lbfgs_outer_steps = step
        stop_reason = "convergence" if stopped_early else "time/step limit"
        print(
            f"  L-BFGS stopped after {step} outer steps and "
            f"{(time.perf_counter() - start_time) / 60.0:.1f} minutes "
            f"({stop_reason})"
        )

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start_time
    final_log_loss = float(log_residual_loss(model, eta).detach())
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory_mb = 0.0
    return (
        model,
        final_log_loss,
        elapsed,
        peak_memory_mb,
        lbfgs_outer_steps,
        stopped_early,
    )


def make_evaluation_data(
    dimension: int,
    eta: float,
    num_points: int,
    seed: int,
    quadrature_order: int,
    sine_modes: int,
    image_terms: int,
    reference_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, ReferenceInfo]:
    """Create shared Sobol points and reference values for one dimension."""

    transformed_points = sample_nd_points(
        n_dimensions=dimension,
        num_points=num_points,
        range_min=-1.0,
        range_max=1.0,
        use_sobol=True,
        seed=seed,
    ).to(dtype=torch.float64, device="cpu")
    unit_cube_points = 0.5 * (transformed_points.numpy() + 1.0)
    values, info = reference_solution(
        unit_cube_points,
        eta=eta,
        quadrature_order=quadrature_order,
        sine_modes=sine_modes,
        image_terms=image_terms,
        chunk_size=reference_chunk_size,
    )
    return transformed_points, torch.from_numpy(values), info


def _write_results(path: Path, results: Iterable[RunResult]) -> None:
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    """Check the reference integral and the exact residual construction in 1-D."""

    z = np.linspace(0.0, 1.0, 101)[:, None]
    reference, info = reference_solution(
        z,
        eta=0.0,
        quadrature_order=192,
        sine_modes=192,
        chunk_size=101,
    )
    exact = 0.5 * z[:, 0] * (1.0 - z[:, 0])
    max_error = float(np.max(np.abs(reference - exact)))
    integral_error = abs(info.torsion_integral - 1.0 / 12.0)
    if max_error > 1e-7 or integral_error > 2e-8:
        raise AssertionError(
            f"reference self-test failed: max_error={max_error}, "
            f"integral_error={integral_error}"
        )

    eta = 1.25
    model = SPS(s=1, d=1, k=1, equation="nonlocal_diffusion")
    # On x in [-1,1], u((x+1)/2)=(1-x^2)/(8*(1+eta/12)).
    with torch.no_grad():
        model.params[0, 0, 0] = -1.0 / (8.0 * (1.0 + eta / 12.0))
    residual = float(residual_mean_square(model, eta))
    # The residual is represented as a sum of cancelling rank-one terms, so
    # its Gram-form squared norm is subject to ordinary floating-point
    # cancellation even though the represented residual is identically zero.
    if abs(residual) > 1e-12:
        raise AssertionError(f"analytic residual self-test failed: MSE={residual}")
    print(
        "Self-test passed: "
        f"reference max error={max_error:.3e}, "
        f"torsion-integral error={integral_error:.3e}, "
        f"analytic residual MSE={residual:.3e}"
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", "--dim", nargs="+", type=int, default=[2, 5, 10, 20])
    parser.add_argument("--ranks", "--s", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--k", type=int, default=10, help="coefficients per univariate polynomial")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--adam-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-duration-minutes",
        type=float,
        default=120.0,
        help="total Adam + L-BFGS training time per (d, s, seed) run",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=None,
        help="optional cap on L-BFGS outer steps; by default only time limits it",
    )
    parser.add_argument("--lbfgs-max-iter", type=int, default=3)
    parser.add_argument("--lbfgs-history", type=int, default=100)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-28)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-29)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=200,
        help="stop after this many L-BFGS outer calls without meaningful improvement; 0 disables",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-10,
        help="minimum cumulative decrease in log residual MSE that resets early-stop patience",
    )
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--init-noise", type=float, default=0.05)
    parser.add_argument("--eval-points", type=int, default=8192)
    parser.add_argument("--eval-chunk-size", type=int, default=2048)
    parser.add_argument("--reference-chunk-size", type=int, default=2048)
    parser.add_argument("--reference-quadrature", type=int, default=160)
    parser.add_argument("--reference-sine-modes", type=int, default=128)
    parser.add_argument("--reference-image-terms", type=int, default=3)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the sweep without training")
    args = parser.parse_args(argv)
    args.dimensions = list(dict.fromkeys(args.dimensions))
    args.ranks = list(dict.fromkeys(args.ranks))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return

    if any(d < 2 for d in args.dimensions):
        raise ValueError("benchmark dimensions must be at least 2; d=1 is only a self-test")
    if any(rank < 1 for rank in args.ranks):
        raise ValueError("all ranks must be positive")
    if args.k < 1:
        raise ValueError("k must be positive")
    if args.max_duration_minutes <= 0.0 or not math.isfinite(args.max_duration_minutes):
        raise ValueError("max-duration-minutes must be positive")
    if args.lbfgs_steps is not None and args.lbfgs_steps < 0:
        raise ValueError("lbfgs-steps cannot be negative")
    if args.lbfgs_tolerance_grad < 0.0 or args.lbfgs_tolerance_change < 0.0:
        raise ValueError("L-BFGS tolerances cannot be negative")
    if args.early_stop_patience < 0:
        raise ValueError("early-stop-patience cannot be negative")
    if args.early_stop_min_delta < 0.0:
        raise ValueError("early-stop-min-delta cannot be negative")

    if any(seed < 0 for seed in args.seeds) or args.evaluation_seed < 0:
        raise ValueError("seeds must be nonnegative")
    if args.adam_steps < 0:
        raise ValueError("adam-steps cannot be negative")
    if args.learning_rate <= 0 or args.lbfgs_max_iter < 1 or args.lbfgs_history < 1:
        raise ValueError("learning-rate, lbfgs-max-iter and lbfgs-history must be positive")
    if min(args.eval_points, args.eval_chunk_size, args.reference_chunk_size) < 1:
        raise ValueError("evaluation counts and chunk sizes must be positive")
    if args.dry_run:
        for d in args.dimensions:
            for s in args.ranks:
                for seed in args.seeds:
                    print(f"constant: d={d}, s={s}, k={args.k}, seed={seed}, budget={args.max_duration_minutes:g} min")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")
    output_dir = args.output_dir or Path(f"results_constant_kernel_{timestamp}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            output_file,
            indent=2,
        )

    print(f"Device: {device}")
    print(f"Results: {output_dir}")
    all_results: list[RunResult] = []

    # Reference points are shared by every rank and seed at a fixed dimension.
    # The nested loops below are the d-by-s benchmark sweep requested for the
    # experiment; seeds form an optional third sweep.
    for dimension in args.dimensions:
        print(f"\nPreparing d={dimension} reference data ...")
        points, reference, reference_info = make_evaluation_data(
            dimension=dimension,
            eta=args.eta,
            num_points=args.eval_points,
            seed=args.evaluation_seed + dimension,
            quadrature_order=args.reference_quadrature,
            sine_modes=args.reference_sine_modes,
            image_terms=args.reference_image_terms,
            reference_chunk_size=args.reference_chunk_size,
        )
        print(
            f"  A_d={reference_info.torsion_integral:.10e}, "
            f"1+eta*A_d={reference_info.denominator:.10e}"
        )

        for rank in args.ranks:
            for seed in args.seeds:
                print(f"\nTraining d={dimension}, s={rank}, k={args.k}, seed={seed}")
                (
                    model,
                    log_mse,
                    elapsed,
                    peak_memory,
                    lbfgs_outer_steps,
                    stopped_early,
                ) = train_one(
                    dimension=dimension,
                    rank=rank,
                    k=args.k,
                    seed=seed,
                    eta=args.eta,
                    device=device,
                    adam_steps=args.adam_steps,
                    learning_rate=args.learning_rate,
                    max_duration_minutes=args.max_duration_minutes,
                    lbfgs_steps=args.lbfgs_steps,
                    lbfgs_max_iter=args.lbfgs_max_iter,
                    lbfgs_history=args.lbfgs_history,
                    lbfgs_tolerance_grad=args.lbfgs_tolerance_grad,
                    lbfgs_tolerance_change=args.lbfgs_tolerance_change,
                    early_stop_patience=args.early_stop_patience,
                    early_stop_min_delta=args.early_stop_min_delta,
                    gradient_clip=args.gradient_clip,
                    init_noise=args.init_noise,
                    progress_every=args.progress_every,
                )
                relative_l2, rmse = evaluate_model(
                    model,
                    transformed_points=points,
                    reference_values=reference,
                    device=device,
                    chunk_size=args.eval_chunk_size,
                )
                result = RunResult(
                    dimension=dimension,
                    rank=rank,
                    k=args.k,
                    seed=seed,
                    eta=args.eta,
                    num_parameters=sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    log_residual_mse=log_mse,
                    residual_mse=math.exp(log_mse),
                    relative_l2_error=relative_l2,
                    rmse=rmse,
                    train_seconds=elapsed,
                    lbfgs_outer_steps=lbfgs_outer_steps,
                    stopped_early=stopped_early,
                    peak_memory_mb=peak_memory,
                    torsion_integral=reference_info.torsion_integral,
                    reference_denominator=reference_info.denominator,
                    eval_points=args.eval_points,
                )
                all_results.append(result)
                _write_results(output_dir / "results.csv", all_results)
                print(
                    f"  relative L2={relative_l2:.6e}, RMSE={rmse:.6e}, "
                    f"residual MSE={result.residual_mse:.6e}, time={elapsed:.1f}s"
                )

                if not args.no_save_models:
                    checkpoint_path = output_dir / f"d_{dimension}_s_{rank}_seed_{seed}.pt"
                    torch.save(
                        {
                            "state_dict": _clone_state_dict(model),
                            "dimension": dimension,
                            "rank": rank,
                            "k": args.k,
                            "eta": args.eta,
                            "metrics": asdict(result),
                        },
                        checkpoint_path,
                    )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"\nFinished {len(all_results)} runs. Results saved to {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
