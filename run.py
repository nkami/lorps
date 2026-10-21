from __future__ import annotations

import argparse
import csv
from datetime import datetime
from functools import partial
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.optim import Adam, LBFGS

from models import get_model_setup
import convolution_equation as convolution
import fredholm_equation as fredholm
import heat_memory_equation as heat
import nonlocal_diffusion_equation as nonlocal_diffusion
import schrodinger_equation as schrodinger

torch.set_default_dtype(torch.float64)

ARCHITECTURES = ("pinn", "fls", "qres", "pinnsformer")
SAMPLERS = ("monte_carlo__uniform", "monte_carlo__sobol", "double_sampling__uniform")
ALL_METHODS = ["sps"] + [f"{arch}__{sampler}" for arch in ARCHITECTURES for sampler in SAMPLERS]
EQUATIONS = ("convolution", "nonlocal_diffusion", "heat", "fredholm", "schrodinger")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--equation", required=True, choices=(*EQUATIONS, "constant", "constant_kernel"))
    parser.add_argument("--dimensions", "--dim", nargs="+", type=int, default=[20])
    parser.add_argument("--ranks", "--s", nargs="+", type=int, default=None, help="SPS ranks; default 20")
    parser.add_argument("--k", type=int, default=None, help="SPS coefficients per factor; default 8 (degree k-1)")
    parser.add_argument("--q", "--qs", nargs="+", type=int, choices=range(1, 6), default=None, help="heat frequencies; default 1")
    parser.add_argument("--alpha", "--alphas", nargs="+", type=float, default=None, help="Fredholm alpha values; default 75")
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=["sps"], help="sps, a baseline, or all (13 methods)")
    parser.add_argument("--max-duration-minutes", type=float, default=120., help="wall-clock budget per run, including compilation/evaluation")
    parser.add_argument("--adam-steps", type=int, default=999, help="preserves the old switch to L-BFGS at step 1000")
    parser.add_argument("--lbfgs-steps", type=int, default=None, help="optional cap on L-BFGS outer calls")
    parser.add_argument("--lbfgs-max-iter", type=int, default=3)
    parser.add_argument("--lbfgs-history", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None, help="override the architecture's Adam learning rate")
    parser.add_argument("--num-points", type=int, default=None, help="override baseline training sample count")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-chunks", type=int, default=None, help="must divide 100000; defaults to the architecture setting")
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--eager", action="store_true", help="unwrap compiled SPS losses (useful for CPU smoke tests)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without training or creating files")
    args = parser.parse_args(argv)
    if args.q is not None and args.equation != "heat":
        parser.error("--q is only valid for heat")
    if args.alpha is not None and args.equation != "fredholm":
        parser.error("--alpha is only valid for fredholm")
    if any(d < (3 if args.equation == "fredholm" else 2) for d in args.dimensions):
        parser.error("dimensions must be at least 2 (at least 3 for Fredholm)")
    if args.ranks is not None and any(s < 1 for s in args.ranks):
        parser.error("ranks must be positive")
    if args.k is not None and args.k < 1:
        parser.error("k must be positive")
    if args.adam_steps < 0 or (args.lbfgs_steps is not None and args.lbfgs_steps < 0):
        parser.error("optimizer step counts cannot be negative")
    if args.max_duration_minutes <= 0 or not math.isfinite(args.max_duration_minutes):
        parser.error("max-duration-minutes must be finite and positive")
    if min(args.eval_every, args.lbfgs_max_iter, args.lbfgs_history) < 1:
        parser.error("evaluation interval and L-BFGS settings must be positive")
    if args.eval_chunks is not None and (args.eval_chunks < 1 or 100000 % args.eval_chunks):
        parser.error("eval-chunks must be a positive divisor of 100000")
    if args.learning_rate is not None and (args.learning_rate <= 0 or not math.isfinite(args.learning_rate)):
        parser.error("learning-rate must be finite and positive")
    if args.num_points is not None and args.num_points < 1:
        parser.error("num-points must be positive")
    if any(seed < 0 for seed in args.seeds) or args.evaluation_seed < 0:
        parser.error("seeds must be nonnegative")
    if args.alpha is not None and any(not math.isfinite(a) for a in args.alpha):
        parser.error("alpha must be finite")
    methods = []
    for method in args.methods:
        if method == "all":
            methods.extend(ALL_METHODS)
            continue
        if method == "lorps":
            method = "sps"
        if method in ARCHITECTURES:
            method += "__monte_carlo__uniform"
        if method not in ALL_METHODS:
            parser.error(f"unsupported method {method!r}; choose sps or an architecture with one of {SAMPLERS}")
        methods.append(method)
    args.methods = list(dict.fromkeys(methods))
    if "sps" not in args.methods and (args.ranks is not None or args.k is not None):
        parser.error("--ranks and --k require SPS among the selected methods")
    if args.equation == "schrodinger" and args.methods != ["sps"]:
        parser.error("the supplied model setups support only SPS for schrodinger")
    return args


def planned_runs(args):
    """Yield independent configurations; rank sweeps do not duplicate baselines."""
    dimensions = dict.fromkeys(args.dimensions)
    seeds = dict.fromkeys(args.seeds)
    frequencies = dict.fromkeys(args.q or [1]) if args.equation == "heat" else [None]
    alphas = dict.fromkeys(args.alpha or [75.]) if args.equation == "fredholm" else [None]
    for d, q, alpha, method in product(dimensions, frequencies, alphas, args.methods):
        ranks = dict.fromkeys(args.ranks or [20]) if method == "sps" else [None]
        for s, seed in product(ranks, seeds):
            yield dict(equation=args.equation, dimension=d, rank=s,
                       k=(args.k or 8) if method == "sps" else None,
                       q=q, alpha=alpha, method=method, seed=seed)


def equation_functions(config, device, num_points, eager):
    name = config["equation"]
    modules = {"convolution": convolution, "fredholm": fredholm,
               "heat": heat, "nonlocal_diffusion": nonlocal_diffusion, "schrodinger": schrodinger}
    prefix = "heat_memory" if name == "heat" else name
    module = modules[name]
    kwargs = dict(method=config["method"], dim=config["dimension"], num_points=num_points, device=device)
    if name == "heat":
        kwargs["q"] = config["q"]
    elif name == "fredholm":
        kwargs["alpha"] = config["alpha"]
    loss = getattr(module, f"get_{prefix}_loss_func")(**kwargs)
    evaluate = getattr(module, f"{prefix}_eval_func")
    if name == "heat":
        evaluate = partial(evaluate, q=config["q"])
    if eager:
        loss = getattr(loss, "_torchdynamo_orig_callable", loss)
    return loss, evaluate


def run_directory(root, config):
    tag = config["equation"]
    if config["q"] is not None:
        tag += f"_q_{config['q']}"
    if config["alpha"] is not None:
        tag += f"_alpha_{config['alpha']:.17g}"
    shape = f"d_{config['dimension']}"
    if config["rank"] is not None:
        shape += f"_s_{config['rank']}_k_{config['k']}"
    return root / tag / shape / config["method"] / f"seed_{config['seed']}"


def train_one(args, config, device, output_dir):
    torch.manual_seed(config["seed"])
    arch = config["method"].split("__")[0]
    model, lr, num_points, chunks = get_model_setup(arch, config["equation"], config["dimension"],
                                                  s=config["rank"], k=config["k"])
    model = model.to(device)
    lr = args.learning_rate if args.learning_rate is not None else lr
    if arch != "sps" and args.num_points is not None:
        num_points = args.num_points
    chunks = args.eval_chunks if args.eval_chunks is not None else chunks
    loss_func, eval_func = equation_functions(config, device, num_points, args.eager)
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved = dict(config, device=str(device), dtype="float64", torch_version=str(torch.__version__),
                    learning_rate=lr, num_points=num_points, eval_chunks=chunks,
                    num_parameters=sum(p.numel() for p in model.parameters()),
                    adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps,
                    lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_history=args.lbfgs_history,
                    max_duration_minutes=args.max_duration_minutes, eager=args.eager,
                    evaluation_seed=args.evaluation_seed, eval_every=args.eval_every)
    (output_dir / "configuration.json").write_text(json.dumps(resolved, indent=2))
    adam = Adam(model.parameters(), lr=lr)
    lbfgs = LBFGS(model.parameters(), lr=1., max_iter=args.lbfgs_max_iter,
                  history_size=args.lbfgs_history, tolerance_grad=1e-28,
                  tolerance_change=1e-29, line_search_fn="strong_wolfe")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    best = None
    step = 0
    adam_done = lbfgs_done = 0
    peak_memory_mb = 0.
    start = time.perf_counter()

    def closure(optimizer):
        optimizer.zero_grad()
        loss = loss_func(model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at update {step + 1}")
        loss.backward()
        return loss

    def measure(stage):
        nonlocal best
        # Monitoring uses a fixed seed and preserves the training RNG state.
        # Baseline residuals require autograd for spatial derivatives.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.evaluation_seed)
            value = float(loss_func(model).detach())
            torch.manual_seed(args.evaluation_seed)
            with torch.no_grad():
                relative, rmse = eval_func(model, dim=config["dimension"], num_chunks=chunks)
        row = dict(step=step, stage=stage, cur_time=time.perf_counter()-start,
                   loss=value, rel_l2_error=float(relative), rmse=float(rmse),
                   memory=peak_memory_mb)
        if not all(math.isfinite(row[key]) for key in ("loss", "rel_l2_error", "rmse")):
            raise FloatingPointError(f"nonfinite loss or evaluation at update {step}")
        if best is None or value < best["loss"]:
            best = row.copy()
            if not args.no_save_models:
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                configuration=resolved, metrics=best), output_dir / "best.pt")
        print(f"  step={step} {stage} loss={value:.6g} rel_L2={row['rel_l2_error']:.6g} elapsed={row['cur_time']/60:.2f} min", flush=True)
        return row

    status = "time_limit"
    error = None
    last_measured = -1
    with (output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "stage", "cur_time", "loss", "optimizer_loss", "rel_l2_error", "rmse", "memory"])
        writer.writeheader()
        try:
            writer.writerow(measure("initial")); stream.flush(); last_measured = 0
            while time.perf_counter()-start < args.max_duration_minutes*60:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if adam_done < args.adam_steps:
                    stage = "adam"
                    value = closure(adam)
                    adam.step(); adam_done += 1
                else:
                    if args.lbfgs_steps is not None and lbfgs_done >= args.lbfgs_steps:
                        status = "step_limit"
                        break
                    stage = "lbfgs"
                    value = lbfgs.step(lambda: closure(lbfgs)); lbfgs_done += 1
                if device.type == "cuda":
                    peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated(device)/1024**2)
                step += 1
                if step % args.eval_every == 0:
                    writer.writerow(measure(stage)); stream.flush(); last_measured = step
                else:
                    writer.writerow(dict(step=step, stage=stage, cur_time=time.perf_counter()-start,
                                         loss="", optimizer_loss=float(value.detach()), rel_l2_error="", rmse="",
                                         memory=peak_memory_mb))
            if last_measured != step:
                writer.writerow(measure("final")); stream.flush()
        except FloatingPointError as exc:
            status, error = "nonfinite_loss", str(exc)
            print(error, flush=True)
    result = dict(config, status=status, error=error, adam_updates=adam_done, lbfgs_updates=lbfgs_done,
                  train_seconds=time.perf_counter()-start,
                  memory=peak_memory_mb,
                  num_points=num_points, best_step=best["step"] if best else None,
                  loss=best["loss"] if best else None,
                  l2_rel_error=best["rel_l2_error"] if best else None, rmse=best["rmse"] if best else None)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dispatch = argparse.ArgumentParser(add_help=False)
    dispatch.add_argument("--equation")
    selected, remaining = dispatch.parse_known_args(argv)
    if selected.equation in ("constant", "constant_kernel"):
        from constant_kernel_benchmark import main as constant_main
        return constant_main(remaining)
    args = parse_args(argv)
    plans = list(planned_runs(args))
    print(f"Planned runs: {len(plans)}", flush=True)
    if args.dry_run:
        for config in plans:
            print(json.dumps(config))
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = args.output_dir or Path(f"results_{args.equation}_{datetime.now():%Y_%m_%d__%H_%M_%S_%f}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}. Choose a new --output-dir.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "configuration.json").write_text(json.dumps(vars(args), indent=2, default=str))
    results = []
    for config in plans:
        print(f"\nRunning {json.dumps(config)} on {device}", flush=True)
        results.append(train_one(args, config, device, run_directory(root, config)))
        with (root / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nFinished. Results: {root / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
