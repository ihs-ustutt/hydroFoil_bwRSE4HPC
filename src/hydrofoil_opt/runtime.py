"""Runtime bridge to the hydrofoil CFD implementation.

The external dtOO, foamlib, and OpenFOAM runtime is deliberately imported only
when a worker is actually evaluating a candidate.  This keeps the plugin
installable and unit-testable on a development laptop.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any


def evaluate(request: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate one candidate with the established hydrofoil physics code.

    The legacy implementation is loaded only here while its geometry and CFD
    implementation is being split further.  It contains no Pyro dependency;
    the process boundary and all scheduling now belong to hydroflow-opt .
    """

    scratch_dir = Path(request["context"]["scratch_dir"])
    scratch_dir.mkdir(parents=True, exist_ok=True)
    resources = request["context"]["resources"]
    mpi_ranks = int(resources["mpi_ranks"])
    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be at least one")
    os.environ["TMPDIR"] = str(scratch_dir)
    os.environ["FLOW_OPT_MPI_RANKS"] = str(mpi_ranks)
    module = _load_implementation()
    parameters = request["candidate"]["parameters"]
    candidate_id = request["candidate"]["id"]
    fitness, extra, state, history = module.runHydFoil(
        [
            float(parameters["alpha_1"]),
            float(parameters["alpha_2"]),
            float(parameters["t_mid"]),
        ],
        candidate_id,
        solver_launcher=_solver_launcher(request, mpi_ranks),
    )
    return float(fitness), {
        "state": state,
        "fitness_components": extra,
        "history": history,
    }


def _solver_launcher(request: dict[str, Any], mpi_ranks: int) -> list[str]:
    """Return the backend-provided command prefix for ``simpleFoam``."""

    execution = request["context"].get("execution")
    if execution is None:
        return ["mpiexec", "-n", str(mpi_ranks)] if mpi_ranks > 1 else []
    launcher = execution.get("mpi_launcher")
    if not isinstance(launcher, list) or not all(
        isinstance(item, str) and item for item in launcher
    ):
        raise TypeError("context.execution.mpi_launcher must be a string list")
    return list(launcher)


def _load_implementation() -> Any:
    source = _implementation_path()
    if not source.exists():
        raise RuntimeError(
            "hydrofoil implementation is unavailable; install from the "
            "hydrofoil-opt source checkout with its CFD runtime"
        )
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location(
        "hydrofoil_implementation", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load hydrofoil implementation from {source}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _implementation_path() -> Path:
    bundled = files("hydrofoil_opt").joinpath("_legacy", "hydroFoil.py")
    if bundled.is_file():
        return Path(bundled)
    return Path(__file__).resolve().parents[2] / "hydroFoil.py"
