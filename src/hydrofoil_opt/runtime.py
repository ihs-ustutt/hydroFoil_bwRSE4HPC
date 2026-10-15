"""File-based stages bridging hydroflow-opt to the hydrofoil CFD code."""

import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

from hydroflow_opt.models import EvaluationResult
from hydroflow_opt.results import write_result

from hydrofoil_opt.case import case_directory


def prepare(request_path: Path) -> None:
    """Build and decompose a case, ready for the laminar solve."""
    from foamlib import FoamCase

    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = request["candidate"]
    scratch = Path(request["context"]["scratch_dir"])
    os.environ["TMPDIR"] = str(scratch)
    module = _load_implementation()
    foil = module.hydFoil(**candidate["parameters"], state=candidate["id"])
    foil.Geometry()
    foil.GeometryMesh()
    foil.Mesh()
    directory = case_directory(scratch, candidate["id"])
    if not directory.is_dir():
        raise RuntimeError(
            f"geometry did not create expected case: {directory}"
        )
    case = FoamCase(directory)
    ranks = request["context"]["resources"]["mpi_ranks"]
    if ranks > 1:
        case.decompose_par_dict["numberOfSubdomains"] = ranks
        case.decompose_par_dict["method"] = "scotch"
        case.decompose_par()
    case.turbulence_properties["RAS"]["turbulence"] = False
    case.control_dict["endTime"] = 500
    case.control_dict["writeInterval"] = 500
    (scratch / "history.json").write_text(
        json.dumps({"Start Time": time.time()}), encoding="utf-8"
    )


def configure_turbulent(request_path: Path) -> None:
    """Continue from the laminar solution with turbulence enabled."""
    from foamlib import FoamCase

    request = json.loads(request_path.read_text(encoding="utf-8"))
    directory = case_directory(
        Path(request["context"]["scratch_dir"]), request["candidate"]["id"]
    )
    case = FoamCase(directory)
    case.turbulence_properties["RAS"]["turbulence"] = True
    case.control_dict["endTime"] = 1000
    case.control_dict["writeInterval"] = 1000


def finalize(request_path: Path, result_path: Path) -> None:
    """Reconstruct, evaluate, persist the result, then clean the case."""
    import numpy as np
    from foamlib import FoamCase

    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_id = request["candidate"]["id"]
    scratch = Path(request["context"]["scratch_dir"])
    directory = case_directory(scratch, candidate_id)
    if request["context"]["resources"]["mpi_ranks"] > 1:
        FoamCase(directory).reconstruct_par()
    fitness, head, force, efficiency = _load_implementation().evaluate_case(
        str(directory), candidate_id
    )
    result = EvaluationResult.success(
        candidate_id,
        float(np.asarray(fitness).item()),
        metadata={
            "state": candidate_id,
            "fitness_components": {
                "dHMean": head.tolist(),
                "FMean": float(force),
                "eta": efficiency.tolist(),
            },
            "history": json.loads(
                (scratch / "history.json").read_text(encoding="utf-8")
            ),
        },
    )
    write_result(result_path, result)
    try:
        shutil.rmtree(directory)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not clean completed case %s", directory, exc_info=True
        )


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
