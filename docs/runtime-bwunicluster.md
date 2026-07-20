# bwUniCluster dtOO Runtime Setup

This document records the current cluster-specific setup for running the
hydrofoil case with `hydroflow-opt`. It is based on the dtOO [author's build recipe](https://github.com/ihs-ustutt/dtOO/issues/67).
The result is expected to be:

- dtOO and third-party libraries installed in `~/dtOO-install`
- dtOO source code in `~/dtOO`
- foamFine source code in `~/foamFine`
- a Python 3.13 virtual environment in `~/py313-dtoo`

The setup is not a generic dtOO installation guide. It assumes the listed
bwUniCluster modules are available.

## 1. Create the Python Environment

```bash
module load devel/python/3.13.3-gnu-14.2
python -m venv ~/py313-dtoo
source ~/py313-dtoo/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy==2.1.2 \
  pygmo \
  oslo.concurrency \
  scikit-learn
```

Install `hydroflow-opt` and the hydrofoil plugin from source:

```bash
python -m pip install -e ~/path/to/hydroflow-opt
python -m pip install -e "~/path/to/hydroFoil_bwRSE4HPC[runtime]"
```

## 2. Prepare the Shell Environment

Store this as `~/pe` or another clearly named environment script:

```bash
#!/bin/bash

for module_name in cae/openfoam/v2412 devel/python/3.13.3-gnu-14.2
do
  module load "${module_name}"
done

foamInit

export DTOO_EXTERNLIBS=~/dtOO-install
export FOAMFINE_DIR=~/foamFine

export PYTHONPATH=${DTOO_EXTERNLIBS}/tools:${PYTHONPATH}
export PYTHONPATH=${DTOO_EXTERNLIBS}/scripts/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${DTOO_EXTERNLIBS}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${DTOO_EXTERNLIBS}/lib64:${LD_LIBRARY_PATH}

source ~/py313-dtoo/bin/activate

alias dtoo-cmake-conf="\
  cmake \
  -DPython3_EXECUTABLE=python3 \
  -DCMAKE_INSTALL_PREFIX=${DTOO_EXTERNLIBS} \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .."

export OSLO_LOCK_PATH=~/oslo-lock
```

Activate it before building or running:

```bash
source ~/pe
```

## 3. Clone dtOO and foamFine

```bash
git clone https://github.com/ihs-ustutt/dtOO.git ~/dtOO
cd ~/dtOO
git submodule init
git submodule update

git clone https://github.com/ihs-ustutt/foamFine.git ~/foamFine
```

For reproducible runs, record the commits:

```bash
git -C ~/dtOO rev-parse HEAD
git -C ~/foamFine rev-parse HEAD
```

## 4. Compile foamFine

```bash
source ~/pe
cd ~/foamFine/of
wmake all
```

## 5. Compile dtOO Third-Party Dependencies

```bash
source ~/pe
cd ~/dtOO/dtOO-ThirdParty

_extraConfig="\
  --extra-cmake=pythonocc-core@-DCMAKE_CXX_FLAGS=-I${DTOO_EXTERNLIBS}/include \
  --extra-cmake=moab@-DMOAB_DEP_LIBRARIES=$(find ${DTOO_EXTERNLIBS}/ | grep -P '.*libcgns.so$') \
"

for dependency in \
  cgns \
  muparser \
  openmesh \
  openvolumemesh \
  nlohmann_json \
  occt \
  pythonocc-core \
  gmsh \
  mpfr-4.1.0 \
  cgal \
  moab
do
  sh buildDep -i "${DTOO_EXTERNLIBS}" -o "${dependency}" \
    ${_extraConfig} --tee
done
```

## 6. Compile dtOO

```bash
source ~/pe
cd ~/dtOO
mkdir -p build
cd build
dtoo-cmake-conf
make -j8 install
```

## 7. Validate the Runtime

Run the dtOO test suite:

```bash
source ~/pe
cd ~/dtOO/build
ctest
```

Run small import checks:

```bash
python -c "import dtOOPythonSWIG as dtOO; print('dtOO ok')"
python -c "import pyDtOO; print('pyDtOO ok')"
python -c "from dtOOPythonApp.builder import ofOpenFOAMCase_turboMachine; print('dtOOPythonApp ok')"
python -c "import foamlib; print('foamlib ok')"
command -v simpleFoam
command -v mpiexec
```

Finally run the hydrofoil plugin runtime check:

```bash
hydrofoil-opt check-runtime
```

Only after this passes should a real hydrofoil run be attempted:

```bash
hydroflow-opt check examples/hydrofoil.toml
hydroflow-opt optimize examples/hydrofoil.toml
```

## Notes

- The current setup is tied to the loaded Python module. If dtOO is built with
  Python 3.13, run `hydroflow-opt` with the same Python environment.
- `LD_LIBRARY_PATH` and `PYTHONPATH` are part of the runtime contract. Missing
  paths usually show up as import errors for `dtOOPythonSWIG` or OpenCASCADE
  libraries.
- The Docker image `atismer/dtoo-opensuse:stable` is not equivalent to this
  setup. It currently exposes older Python versions and should not be assumed
  to run this `hydroflow-opt` workflow.
