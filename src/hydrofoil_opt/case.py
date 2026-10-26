"""Case declaration for hydroflow-opt without importing CFD dependencies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hydroflow_opt.models import (
    Candidate,
    EvaluationPaths,
    EvaluationPlan,
    EvaluationStage,
    ParameterSpace,
    ResourceRequest,
    StageResources,
)


class HydrofoilCase:
    """Expose hydrofoil parameters and its staged evaluation workflow."""

    def parameter_space(self, options: dict[str, Any]) -> ParameterSpace:
        """Return the physical optimization bounds for the hydrofoil."""
        del options
        return ParameterSpace(
            names=("alpha_1", "alpha_2", "t_mid"),
            lower_bounds=(150.0, 155.0, 0.01),
            upper_bounds=(170.0, 175.0, 0.1),
        )

    def evaluation_plan(
        self,
        candidate: Candidate,
        paths: EvaluationPaths,
        resources: ResourceRequest,
    ) -> EvaluationPlan:
        """Describe preparation, two solves, and finalization."""
        worker = (sys.executable, "-m", "hydrofoil_opt.worker")
        request = str(paths.request_path)
        directory = case_directory(paths.scratch_dir, candidate.id)
        solver = ("simpleFoam",)
        if resources.mpi_ranks > 1:
            solver += ("-parallel",)
        solver_resources = StageResources(
            resources.mpi_ranks, resources.threads_per_rank
        )
        return EvaluationPlan(
            stages=(
                EvaluationStage(
                    "prepare",
                    (*worker, "prepare", request),
                    paths.scratch_dir,
                ),
                EvaluationStage(
                    "solve-laminar",
                    solver,
                    directory,
                    solver_resources,
                ),
                EvaluationStage(
                    "configure-turbulent",
                    (*worker, "configure-turbulent", request),
                    directory,
                ),
                EvaluationStage(
                    "solve-turbulent",
                    solver,
                    directory,
                    solver_resources,
                ),
                EvaluationStage(
                    "finalize",
                    (*worker, "finalize", request, str(paths.result_path)),
                    paths.scratch_dir,
                ),
            )
        )


def case_directory(scratch_dir: Path, candidate_id: str) -> Path:
    """Return the directory produced by dtOO's case label and state."""
    return scratch_dir / f"of_{candidate_id}"
