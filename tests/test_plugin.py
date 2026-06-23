import json

from flow_opt_hydrofoil.case import HydrofoilCase
from flow_opt_hydrofoil.worker import main


def test_case_exposes_the_hydrofoil_parameter_space():
    space = HydrofoilCase().parameter_space({})
    assert space.names == ("alpha_1", "alpha_2", "t_mid")
    assert space.lower_bounds == (150.0, 155.0, 0.01)


def test_worker_returns_a_structured_failure_without_cfd_runtime(tmp_path):
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "id": "test-1",
                    "parameters": {
                        "alpha_1": 160.0,
                        "alpha_2": 165.0,
                        "t_mid": 0.05,
                    },
                },
                "context": {
                    "scratch_dir": str(tmp_path / "scratch"),
                    "resources": {"mpi_ranks": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    assert main([str(request_path), str(result_path)]) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["candidate_id"] == "test-1"
    assert result["status"] == "failed"
