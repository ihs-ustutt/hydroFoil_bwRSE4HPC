# flow-opt-hydrofoil

`flow-opt-hydrofoil` is the hydrofoil case plugin for
[`flow-opt`](../flow-opt). It has no Pyro5 dependency and does not implement
its own scheduler or worker pool. `flow-opt` starts an isolated worker for
each candidate and owns the global resource budget.

## Runtime prerequisites

The package can be installed and unit-tested without CFD software. Running a
real hydrofoil evaluation requires a separately provisioned Linux environment
with dtOO, foamlib, OpenFOAM, and an MPI launcher. Set up those tools as usual
before invoking `flow-opt`; this package deliberately does not load them until
its worker executes a candidate.

Check the currently active shell before running a real case:

```bash
flow-opt-hydrofoil check-runtime
```

This prints a compact checklist for the dtOO Python bindings, OpenFOAM tools,
MPI launcher, and relevant environment variables. It does not run a CFD case.

## Example

Install the orchestration package and this plugin in the same environment.
Then run either explicit candidates or the pygmo island optimizer:

```bash
flow-opt check examples/hydrofoil.toml
flow-opt optimize examples/hydrofoil.toml
```

The resource shape is explicit. For example, four two-rank CFD evaluations
need at least eight CPUs:

```toml
[resources]
available_cpus = 8
concurrent_evaluations = 4
mpi_ranks = 2
threads_per_rank = 1
```

The local worker uses `mpiexec -n <mpi_ranks>` and never passes
`--oversubscribe`. A future Slurm backend will launch this same worker with
`srun` and owns placement inside the job allocation.
