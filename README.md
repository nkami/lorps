# LoRPS experiments

Run commands below from this directory in your existing LoRPS environment.
`run.py` is the common entry point. `sps` remains the internal model name;
`--methods lorps` is accepted as an alias.

## Main LoRPS equations

```bash
python run.py --equation convolution --seeds 0 --device cuda
python run.py --equation nonlocal_diffusion --seeds 0 --device cuda
python run.py --equation heat --q 1 --seeds 0 --device cuda
python run.py --equation fredholm --alpha 1 25 50 75 --seeds 0 --device cuda
```

For the five-seed runs, replace `--seeds 0` with `--seeds 0 1 2 3 4`.
Set `--max-duration-minutes` explicitly to change the 120-minute limit.

## Heat frequency experiment

```bash
python run.py --equation heat --q 1 2 3 4 5 --dimensions 20 --ranks 20 --k 8 --seeds 0 --device cuda --max-duration-minutes 120
```

Q selects the projection file, the Q-dependent forcing coefficient, and the
exact reference solution. Each loss captures its own Q, including baseline
losses. Frequencies 1 through 5 have supplied projection files.

## Infinite-rank benchmark

```bash
python run.py --equation constant --dimensions 2 5 10 20 --ranks 1 2 4 8 16 32 --k 10 --seeds 0 --device cuda --max-duration-minutes 120
```

