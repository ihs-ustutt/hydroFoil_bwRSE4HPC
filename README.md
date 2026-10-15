# hydrofoil-opt

[![Build](https://github.com/ihs-ustutt/hydroFoil_bwRSE4HPC/actions/workflows/ci.yml/badge.svg)](https://github.com/ihs-ustutt/hydroFoil_bwRSE4HPC/actions)

`hydrofoil-opt` is a case plugin for [`hydroflow-opt`](https://github.com/thomasisensee/hydroflow-opt).

## Runtime prerequisites

The package can be installed and unit-tested without CFD software. Running a real hydrofoil evaluation requires a separately provisioned Linux environment with [`dtOO`](https://github.com/ihs-ustutt/dtOO), foamlib, OpenFOAM, and an MPI launcher. Set up those tools as usual before invoking `hydroflow-opt`.

Check the currently active shell before running a real case:

```bash
hydrofoil-opt check-runtime
```

This prints a checklist for the dtOO Python bindings, OpenFOAM tools, MPI launcher, and relevant environment variables.

## Example

Install the orchestration package and this plugin in the same environment. Then run either explicit candidates or the pygmo island optimizer via

```bash
hydroflow-opt check examples/hydrofoil_candidate.toml
hydroflow-opt run examples/hydrofoil_candidate.toml
```

or

```bash
hydroflow-opt check examples/hydrofoil_optimization.toml
hydroflow-opt optimize examples/hydrofoil_optimization.toml
```
respectively.

The resource shape is explicit. For example, four two-rank CFD evaluations need at least eight CPUs:

```toml
[resources]
available_cpus = 8
concurrent_evaluations = 4
mpi_ranks = 2
threads_per_rank = 1
```

The local backend then uses `mpiexec -n <mpi_ranks>` for parallel solves.

For Slurm execution, add:

```toml
[execution]
backend = "slurm"

[resources]
memory_mib_per_evaluation = 1536
```

and run `hydroflow-opt` inside an existing `sbatch` or `salloc` allocation. The allocation may span nodes, but each candidate must fit on one node. Preparation, laminar solve, turbulence configuration, turbulent solve, and finalization each run as separate Slurm job steps. Python stages use one task; each parallel OpenFOAM solver stage is launched directly as:

```text
srun --exclusive --nodes=1 --ntasks=<mpi_ranks> \
  --cpus-per-task=<threads_per_rank> --cpu-bind=cores --mem=1536M --mpi=pmix \
  simpleFoam -parallel
```

There is no nested `mpiexec` in Slurm mode. Local execution continues to use `mpiexec -n <mpi_ranks>`. For four islands with two one-thread MPI ranks per candidate, request eight Slurm tasks with one CPU each:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
```

Before a production run, verify that the cluster supports direct Open MPI launching with `srun --mpi=list`; the available plugin list must include `pmix`.
