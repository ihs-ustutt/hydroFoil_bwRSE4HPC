"""Exercise legacy coupling-face selection without loading the CFD runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "labels",
    [
        ["xyz_meshBlock_0", "xyz_meshBlock_1", "xyz_meshBlock_2"],
        ["xyz_meshBlock_1", "xyz_meshBlock_2", "xyz_meshBlock_3"],
    ],
)
def test_outer_region_couples_actual_first_and_last_blocks(labels):
    source = (
        Path(__file__).resolve().parents[1]
        / "src/hydrofoil_opt/_legacy/hydroFoil.py"
    )
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "Mesh"
    )
    faces = []
    valid_faces = {
        label + suffix
        for label in labels
        for suffix in ("->NORTH", "->WEST", "->EAST")
    }

    def face(label):
        assert label in valid_faces
        return label

    class Geometry:
        def __init__(self, label):
            self.label = label

        def getLabel(self):
            return self.label

        def segmentConstUPercent(self, value):
            return None

        def segmentConstWPercent(self, value):
            return None

    class Geometries(dict):
        def getIndices(self, pattern):
            return labels

        def getLabel(self, index):
            return index

    class CouplingComplete(Exception):
        pass

    def observer():
        raise CouplingComplete

    model = SimpleNamespace(
        addIfRegionToGmshModel=lambda geometry: geometry,
        getRegionByTag=lambda tag: tag,
        tagPhysical=lambda *args: None,
        getDtGmshRegionByPhysical=lambda label: SimpleNamespace(
            addFace=lambda face, orientation: faces.append(face)
        ),
        getDtGmshFaceByPhysical=face,
    )
    volume = SimpleNamespace(
        jInit=lambda *args: None,
        getModel=lambda: model,
    )
    dtOO = SimpleNamespace(
        map3dTo3d=SimpleNamespace(
            MustDownCast=lambda geometry: geometry,
            DownCast=lambda geometry: geometry,
        ),
        gmshBoundedVolume=lambda: volume,
        jsonPrimitive=lambda value: value,
        bVOAddInternalEdge=observer,
    )
    namespace = {"dtOO": dtOO}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    foil = SimpleNamespace(
        aG=Geometries(
            {label: Geometry(label) for label in [*labels, "xyz_channel"]}
        ),
        bV=SimpleNamespace(set=lambda volume: None),
    )
    # Stop at the observer: all coupling faces have now been added.
    with pytest.raises(CouplingComplete):
        namespace["Mesh"](foil)
    assert faces == [
        *(label + "->NORTH" for label in labels),
        labels[0] + "->WEST",
        labels[-1] + "->EAST",
    ]
