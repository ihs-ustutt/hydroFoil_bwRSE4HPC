"""Case declaration consumed by flow-opt without importing CFD dependencies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from flow_opt.models import ParameterSpace


class HydrofoilCase:
    """Expose the hydrofoil's parameters and isolated worker command."""

    def parameter_space(self, options: dict[str, Any]) -> ParameterSpace:
        """Return the physical optimization bounds for the hydrofoil."""

        del options
        return ParameterSpace(
            names=("alpha_1", "alpha_2", "t_mid"),
            lower_bounds=(150.0, 155.0, 0.01),
            upper_bounds=(170.0, 175.0, 0.1),
        )

    def worker_command(
        self,
        request_path: Path,
        result_path: Path,
    ) -> list[str]:
        """Run each evaluation in a new Python interpreter."""

        return [
            sys.executable,
            "-m",
            "flow_opt_hydrofoil.worker",
            str(request_path),
            str(result_path),
        ]
