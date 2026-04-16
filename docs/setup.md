# Setup — building and running HiPy

The canonical recipe lives in `.github/workflows/test.yml` and the
`Dockerfile`. This doc is the hand-runnable version, validated locally
against `main` on 2026-04-16 (40-test sample from the C++ backend,
plus `compile.py` through the MLIR backend).

Two backends, two setups:

- **C++ backend** (`hipy/cppbackend/`) — required for `pytest test` and
  `check_prints`. Standard pip deps + CMake + Arrow (via pyarrow).
- **MLIR backend** (`hipy/mlirbackend/`) — required for `compile.py`.
  Needs LingoDB's `lingodbbridge` wheel, which is **not on PyPI**.

## 1. System prerequisites

```
Python 3.12 (the Dockerfile pins python:3.12-bookworm)
cmake
ccache        # used as the CMake compiler launcher
build-essential (gcc/g++ ≥ C++20)
```

On Debian/Ubuntu:

```bash
sudo apt update && sudo apt install -y cmake ccache build-essential
```

## 2. Python venv + dependencies

Always use a `.venv` at the repo root — CMake is pointed at its
`python` binary so pyarrow / pybind11 lookups resolve against this
interpreter.

```bash
cd /path/to/hipy
python3 -m venv .venv
source .venv/bin/activate
pip install pytest numpy pandas pyarrow networkx scipy pybind11 \
            jinja2 scikit-learn
python -c "import pyarrow; pyarrow.create_library_symlinks()"
```

`create_library_symlinks()` is mandatory: the CMake build links against
`libarrow.so` / `libarrow_python.so` (unsuffixed), but pyarrow only
ships the versioned files (`libarrow.so.<N>00`). The symlinks are
idempotent — if they already exist from a prior install, leave them.

## 3. LingoDB wheels (only needed for `compile.py`)

`lingodbbridge` is published as a local wheel, not on PyPI. Obtain the
`lingodb` and `lingodb_bridge` wheels for your Python version and
install them into the venv:

```bash
pip install /path/to/lingodb-<version>-py3-none-any.whl
pip install /path/to/lingodb_bridge-<version>-<abi>.whl
```

**Gotcha:** these wheels pin older `pyarrow` (22.0.0) and `numpy`
(1.26.4). They will silently downgrade whatever pip installed in §2.
The pre-existing pyarrow symlinks from §2 still point at working
libraries after the downgrade — don't delete them.

If you only care about `pytest` and not `compile.py`, skip this step.

## 4. Configure the C++ backend

The backend uses an out-of-tree CMake build. Configure once per
checkout:

```bash
cd cppbackend
touch standalone.cpp         # placeholder; compiler rewrites it per run
export HIPY_STANDALONE_SOURCE=$(pwd)
cmake -B /tmp/hipy-generator \
      -DPYTHON_EXECUTABLE=/path/to/hipy/.venv/bin/python .
export HIPY_STANDALONE_BUILD=/tmp/hipy-generator
cd ..
```

- `HIPY_STANDALONE_SOURCE` — directory containing `standalone.cpp`
  (the file the compiler overwrites with generated code).
- `HIPY_STANDALONE_BUILD` — CMake build directory. The backend runs
  `cmake --build $HIPY_STANDALONE_BUILD --target standalone` for every
  compile, so pick a persistent path (ccache relies on it to short-cut
  rebuilds).
- `-DPYTHON_EXECUTABLE=...` must point at the venv's `python`, else
  CMake's `pyarrow.get_include()` shell-out uses the wrong interpreter
  and the Arrow headers are invisible.

See `docs/cpp-backend.md` for what each build artifact is.

## 5. Running `pytest`

```bash
export HIPY_STANDALONE_SOURCE=/path/to/hipy/cppbackend
export HIPY_STANDALONE_BUILD=/tmp/hipy-generator
export PYTHONPATH=".:$PYTHONPATH"
source .venv/bin/activate
pytest test                       # full suite
pytest test/test_hello_world.py   # single file
pytest test/ -k fannkuch          # filter
HIPY_DEBUG=1 pytest test/         # enable Context sanity checks
```

The first test invocation builds the C++ generator binary (slow,
~30–60s). Subsequent runs reuse `HIPY_STANDALONE_BUILD` + ccache and
are fast.

See `docs/tests.md` for the `check_prints` idiom and `not_constant(...)`.

## 6. Running `compile.py` (MLIR backend)

`compile.py` emits an MLIR module (via `hipy/mlirbackend/`) for a
single function in a user file. No executable — the MLIR is printed
(or written to the optional fourth argument).

```
python compile.py <file> <function> <arg_types_json> [function_suffix] [fallback|no_fallback] [output_file]
```

`arg_types_json` is a JSON list of `"str"` / `"int"` / `"float"` / `"date"`
(see `compile.py:33-45` for the full set).

- `function_suffix` — sets `hipy.config.function_suffix` (prepended with
  an underscore) so generated function names can be disambiguated when
  compiling several variants into the same module. See `compile.py:23`.
- `fallback` (default) vs `no_fallback` — toggles whether
  `hipy.compiler.compile(..., fallback=...)` is allowed to insert the
  generic Python-fallback path. `compile.py:21`.
- Any input `<file>` not ending in `.py` is copied to a sibling `.py`
  path before `importlib` loads it (`compile.py:47-54`) so hosts that
  pass unextensioned files still work.

Example (validated):

```bash
cat > /tmp/sample.py <<'PY'
import hipy

@hipy.compiled_function
def add_one(x: int) -> int:
    return x + 1
PY

PYTHONPATH=.:$PYTHONPATH .venv/bin/python compile.py \
    /tmp/sample.py add_one '["int"]'
```

Expected output:

```
module {
  func.func @add_one(%arg0: i64) -> i64 {
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.addi %arg0, %c1_i64 : i64
    return %0 : i64
  }
}
```

Only the subset documented in `docs/mlir-backend.md §4` lowers —
composite types, closures, and most shim libraries will assert.

## 7. CI reference

`.github/workflows/test.yml` runs inside the `ghcr.io/lingo-db/hipy-dev`
container, which provides LingoDB deps pre-built. It mirrors the
steps above verbatim — when you change the local setup, update the
workflow too (and vice versa).

The container has no `lingodbbridge`-related steps because the test
suite only needs the C++ backend. `compile.py` is not exercised in CI.

## 8. Gotchas

- **Always activate the `.venv` before running CMake.** If you configure
  with the system Python, pyarrow paths resolve against system pyarrow
  (or not at all) and the link step fails on Arrow symbols.
- **`HIPY_STANDALONE_SOURCE` / `HIPY_STANDALONE_BUILD` must be set for
  every shell.** They're not persisted anywhere. A missing
  `HIPY_STANDALONE_BUILD` manifests as the first `check_prints` hanging
  or CMake complaining about an empty build dir.
- **The pyarrow/numpy downgrade from the lingodb wheels is sticky.**
  If you later `pip install -U pyarrow`, rerun `lingodbbridge` and it
  will assert on ABI mismatch. Keep 22.0.0 pinned while `compile.py`
  matters to you.
- **`touch standalone.cpp` is not cargo-cult.** CMake's `add_executable`
  requires the source file to exist at configure time; the compiler
  later overwrites it. Skip the touch and configure fails.
- **ccache is required by the CMakeLists.txt**, not optional — it's set
  as `CMAKE_CXX_COMPILER_LAUNCHER` (see `cppbackend/CMakeLists.txt:5`).
  A missing `ccache` binary manifests at build time, not configure
  time; install it before running step 4.
