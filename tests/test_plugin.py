import json

import flow_opt_hydrofoil.runtime_check as runtime_check
from flow_opt_hydrofoil.case import HydrofoilCase
from flow_opt_hydrofoil.cli import main as cli_main
from flow_opt_hydrofoil.runtime_check import CheckResult
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


def test_runtime_check_reports_missing_runtime(monkeypatch, tmp_path):
    implementation = tmp_path / "hydroFoil.py"
    implementation.write_text("# test implementation\n", encoding="utf-8")

    def fake_import_module(name):
        if name == "dtOOPythonSWIG":
            raise ImportError("missing dtOO")
        return object()

    monkeypatch.setattr(runtime_check, "import_module", fake_import_module)
    monkeypatch.setattr(runtime_check, "which", lambda name: None)
    monkeypatch.setattr(
        runtime_check, "_implementation_path", lambda: implementation
    )

    results = runtime_check.run_checks()
    report = runtime_check.format_report(results, environ={})

    assert runtime_check.failed_required_count(results) == 3
    assert "FAIL import dtOOPythonSWIG" in report
    assert "FAIL executable simpleFoam" in report
    assert "Runtime check failed: 3 required check(s) failed." in report


def test_runtime_check_passes_when_requirements_are_available(
    monkeypatch, tmp_path
):
    implementation = tmp_path / "hydroFoil.py"
    implementation.write_text("# test implementation\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_check, "import_module", lambda module_name: object()
    )
    monkeypatch.setattr(
        runtime_check, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        runtime_check, "_implementation_path", lambda: implementation
    )

    results = runtime_check.run_checks()
    report = runtime_check.format_report(
        results,
        environ={"FLOW_OPT_MPI_RANKS": "2"},
    )

    assert runtime_check.failed_required_count(results) == 0
    assert "OK   import dtOOPythonSWIG" in report
    assert "FLOW_OPT_MPI_RANKS=2" in report
    assert "Runtime check passed." in report


def test_runtime_cli_returns_nonzero_for_failed_checks(monkeypatch):
    monkeypatch.setattr(
        "flow_opt_hydrofoil.cli.run_checks",
        lambda: [CheckResult("import dtOOPythonSWIG", False)],
    )

    assert cli_main(["check-runtime"]) == 1


def test_runtime_cli_returns_zero_for_successful_checks(monkeypatch):
    monkeypatch.setattr(
        "flow_opt_hydrofoil.cli.run_checks",
        lambda: [CheckResult("import dtOOPythonSWIG", True)],
    )

    assert cli_main(["check-runtime"]) == 0
