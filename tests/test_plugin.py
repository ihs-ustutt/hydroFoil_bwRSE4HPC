import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydroflow_opt.backends import SlurmBackend, SubprocessBackend
from hydroflow_opt.cases import case_from_name
from hydroflow_opt.cli import main as core_main
from hydroflow_opt.config import load_config
from hydroflow_opt.models import (
    Candidate,
    EvaluationPaths,
    EvaluationPlan,
    ResourceRequest,
)

import hydrofoil_opt.runtime as runtime
import hydrofoil_opt.runtime_check as runtime_check
from hydrofoil_opt.case import HydrofoilCase, case_directory
from hydrofoil_opt.cli import main as cli_main
from hydrofoil_opt.runtime_check import CheckResult
from hydrofoil_opt.worker import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.toml")))
def test_current_core_discovers_plugin_and_checks_examples(path):
    assert isinstance(case_from_name("hydrofoil"), HydrofoilCase)
    assert core_main(["check", str(path)]) == 0


@pytest.mark.parametrize("ranks", [1, 2])
def test_plan_and_backend_launchers(tmp_path, monkeypatch, ranks):
    case = HydrofoilCase()
    assert case.parameter_space({}).names == ("alpha_1", "alpha_2", "t_mid")
    paths = EvaluationPaths(
        tmp_path, tmp_path, tmp_path / "request.json", tmp_path / "result.json"
    )
    resources = ResourceRequest(4, 1, ranks, 2, 1536)
    stages = case.evaluation_plan(
        Candidate("test", {}), paths, resources
    ).stages
    assert [stage.name for stage in stages] == [
        "prepare",
        "solve-laminar",
        "configure-turbulent",
        "solve-turbulent",
        "finalize",
    ]
    config = replace(
        load_config(EXAMPLES / "hydrofoil_candidate.toml"), resources=resources
    )
    local = SubprocessBackend(config, case)
    slurm = SlurmBackend(config, case)
    monkeypatch.setattr(SlurmBackend, "validate_environment", lambda *_: None)
    for index, stage in enumerate(stages):
        parallel = index in (1, 3)
        assert stage.resources.processes == (ranks if parallel else 1)
        assert stage.resources.threads_per_process == (2 if parallel else 1)
        command = local.launch_command(stage)
        if parallel:
            expected = ["simpleFoam"]
            if ranks > 1:
                expected = ["mpiexec", "-n", "2", "simpleFoam", "-parallel"]
            assert command == expected
        else:
            assert command[:3] == [
                sys.executable,
                "-m",
                "hydrofoil_opt.worker",
            ]
        command = slurm.launch_command(stage)
        assert command[0] == "srun"
        assert f"--ntasks={stage.resources.processes}" in command
        assert "--mem=1536M" in command
        assert "mpiexec" not in command
        assert ("--mpi=pmix" in command) == (parallel and ranks > 1)


@pytest.fixture
def staged_runtime(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    directory = case_directory(scratch, "test")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "candidate": {
                    "id": "test",
                    "parameters": {
                        "alpha_1": 160.0,
                        "alpha_2": 165.0,
                        "t_mid": 0.05,
                    },
                },
                "context": {
                    "scratch_dir": str(scratch),
                    "resources": {"mpi_ranks": 2},
                },
            }
        )
    )
    calls = []
    case = SimpleNamespace(
        turbulence_properties={"RAS": {}},
        control_dict={"startFrom": "latestTime"},
        decompose_par_dict={},
        decompose_par=lambda: calls.append("decompose"),
        reconstruct_par=lambda: calls.append("reconstruct"),
    )

    class Foil:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def Geometry(self):
            calls.append("geometry")

        def GeometryMesh(self):
            calls.append("geometry-mesh")

        def Mesh(self):
            directory.mkdir()
            calls.append("mesh")

    def evaluate(directory_arg, state):
        assert directory_arg == str(directory)
        assert state == "test"
        calls.append("evaluate")
        return 1.25, np.array(1.0), 2.0, np.array(0.8)

    monkeypatch.setitem(
        sys.modules, "foamlib", SimpleNamespace(FoamCase=lambda path: case)
    )
    monkeypatch.setattr(
        runtime,
        "_load_implementation",
        lambda: SimpleNamespace(hydFoil=Foil, evaluate_case=evaluate),
    )
    monkeypatch.setenv("TMPDIR", str(scratch))
    return request, directory, case, calls


@pytest.mark.parametrize("ranks", [1, 2])
def test_runtime_stages_and_result_before_cleanup(
    staged_runtime, ranks, monkeypatch
):
    request, directory, case, calls = staged_runtime
    data = json.loads(request.read_text())
    data["context"]["resources"]["mpi_ranks"] = ranks
    request.write_text(json.dumps(data))
    assert main(["prepare", str(request)]) == 0
    assert calls[1:4] == ["geometry", "geometry-mesh", "mesh"]
    assert ("decompose" in calls) == (ranks > 1)
    assert case.control_dict["endTime"] == 500
    assert case.turbulence_properties["RAS"]["turbulence"] is False
    main(["configure-turbulent", str(request)])
    assert case.control_dict == {
        "startFrom": "latestTime",
        "endTime": 1000,
        "writeInterval": 1000,
    }
    assert case.turbulence_properties["RAS"]["turbulence"] is True
    result = request.parent / "result.json"
    original_cleanup = runtime.shutil.rmtree

    def cleanup(path):
        assert json.loads(result.read_text())["objective"] == 1.25
        original_cleanup(path)

    monkeypatch.setattr(runtime.shutil, "rmtree", cleanup)
    main(["finalize", str(request), str(result)])
    output = json.loads(result.read_text())
    assert output["metadata"]["state"] == "test"
    assert output["metadata"]["fitness_components"] == {
        "dHMean": 1.0,
        "FMean": 2.0,
        "eta": 0.8,
    }
    assert "Start Time" in output["metadata"]["history"]
    assert ("reconstruct" in calls) == (ranks > 1)
    assert not directory.exists()


def test_finalization_failure_keeps_case(staged_runtime):
    request, directory, case, _ = staged_runtime
    runtime.prepare(request)

    def fail():
        raise RuntimeError("reconstruction failed")

    case.reconstruct_par = fail
    result = request.parent / "result.json"
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        main(["finalize", str(request), str(result)])
    assert directory.exists()
    assert not result.exists()


@pytest.mark.parametrize("failed_stage", [None, "solve-laminar"])
def test_core_executes_plan_and_stops_on_failure(tmp_path, failed_stage):
    config = replace(
        load_config(EXAMPLES / "hydrofoil_candidate.toml"),
        run_dir=tmp_path / "run",
        scratch_dir=tmp_path / "scratch",
    )
    candidate = config.candidates[0]

    class StubCase(HydrofoilCase):
        def evaluation_plan(self, candidate, paths, resources):
            plan = super().evaluation_plan(candidate, paths, resources)
            stages = []
            for stage in plan.stages:
                code = "print('stage ran')"
                if stage.name == "prepare":
                    directory = case_directory(paths.scratch_dir, candidate.id)
                    code += (
                        "; from pathlib import Path; "
                        f"Path({str(directory)!r}).mkdir()"
                    )
                if stage.name == "finalize":
                    result = {
                        "candidate_id": candidate.id,
                        "status": "success",
                        "objective": 1.25,
                    }
                    code += (
                        "; from pathlib import Path; "
                        f"Path({str(paths.result_path)!r})"
                        f".write_text({json.dumps(result)!r})"
                    )
                if stage.name == failed_stage:
                    code = "raise SystemExit(3)"
                stages.append(
                    replace(stage, command=(sys.executable, "-c", code))
                )
            return EvaluationPlan(tuple(stages))

    result = SubprocessBackend(config, StubCase()).evaluate(candidate)
    assert result.status == ("failed" if failed_stage else "success")
    stages_dir = config.run_dir / "evaluations" / candidate.id / "stages"
    if failed_stage:
        assert result.metadata["failed_stage"] == failed_stage
        assert len(list(stages_dir.iterdir())) == 2
        assert case_directory(
            config.scratch_dir / candidate.id, candidate.id
        ).exists()
    else:
        assert result.objective == 1.25
        assert len(result.timings) == 5
        assert len(list(stages_dir.iterdir())) == 5


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

    assert runtime_check.failed_required_count(results) == 5
    assert "FAIL import dtOOPythonSWIG" in report
    assert "FAIL executable simpleFoam" in report
    assert "INFO executable srun" in report
    assert "Runtime check failed: 5 required check(s) failed." in report


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
        environ={"TMPDIR": "/scratch"},
    )

    assert runtime_check.failed_required_count(results) == 0
    assert "OK   import dtOOPythonSWIG" in report
    assert "TMPDIR=/scratch" in report
    assert "Runtime check passed." in report


def test_runtime_cli_returns_nonzero_for_failed_checks(monkeypatch):
    monkeypatch.setattr(
        "hydrofoil_opt.cli.run_checks",
        lambda: [CheckResult("import dtOOPythonSWIG", False)],
    )

    assert cli_main(["check-runtime"]) == 1


def test_runtime_cli_returns_zero_for_successful_checks(monkeypatch):
    monkeypatch.setattr(
        "hydrofoil_opt.cli.run_checks",
        lambda: [CheckResult("import dtOOPythonSWIG", True)],
    )

    assert cli_main(["check-runtime"]) == 0


def test_cleanup_failure_keeps_successful_result(staged_runtime, monkeypatch):
    request, directory, _, _ = staged_runtime
    runtime.prepare(request)

    def fail_cleanup(path):
        raise OSError("cleanup failed")

    monkeypatch.setattr(runtime.shutil, "rmtree", fail_cleanup)
    result = request.parent / "result.json"
    runtime.finalize(request, result)
    assert json.loads(result.read_text())["status"] == "success"
    assert directory.exists()
