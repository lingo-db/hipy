# HiPy — knowledge base for Claude

This repo has a dedicated documentation tree at **`docs/`** that was
written to brief you on the codebase. Consult it before exploring the
source tree blindly — every major component has a doc page with the
relevant file paths, extension seams, and gotchas.

Also at the repo root: **`hipy_paper.md`** — a condensed summary of the
OOPSLA'24 paper that is HiPy's scientific foundation. Read it first if
you need the *why* behind a design decision.

## How to navigate the knowledge base

Start at **`docs/index.md`** — it lists every doc file and the suggested
reading order. The docs are organized by logical component:

| Doc | What it covers |
|---|---|
| `docs/index.md` | Index + reading order |
| `docs/setup.md` | **How to build + run** — venv, pyarrow symlinks, pytest, `compile.py`. Go here first if the repo isn't set up yet on this machine. |
| `docs/extending.md` | **How to add things** — shims, virtual types, IR builtins, passes, backends. Go here first when the task is "add X". |
| `docs/ir.md` | IR types, SSA, ops, modules |
| `docs/compiler.md` | AST rewrite (cogen) + IR-gen loop |
| `docs/context.md` | `GeneratorContext` runtime |
| `docs/value.md` | Virtual-object model |
| `docs/intrinsics.md` | Library-author API |
| `docs/runtime-glue.md` | Decorators, closures, test helpers |
| `docs/standard-library.md` | Every `hipy/lib/` shim |
| `docs/optimizations.md` | Passes in `hipy/opt/` |
| `docs/cpp-backend.md` | IR → C++ + Arrow runtime |
| `docs/mlir-backend.md` | Prototype MLIR backend (stub) |
| `docs/tests.md` | Test layout + `check_prints` idiom |

Each doc ends with a **Gotchas** section capturing non-obvious
invariants. When a test fails unexpectedly or an edit breaks something,
check the gotchas first.

## Workflow hints

- **New feature / bugfix**: skim the relevant component doc(s), then
  look at the source. The docs point to specific files and often
  specific methods.
- **Adding a library shim, virtual type, IR builtin, or optimization
  pass**: `docs/extending.md` is a cookbook — follow the checklist there.
- **Writing tests**: `docs/tests.md` describes the `check_prints` idiom
  and the `not_constant(...)` pattern used to defeat constant folding.
- **Debugging a compile error from the C++ backend**: `docs/cpp-backend.md`
  explains the direct-compile flow (pybind11 / Arrow include + link
  discovery) and where the generated source lands.
  
## Running the test suite

    Use the venv's pytest directly. Each test compiles its C++ into a
    fresh tempdir, so there is no build directory to configure:

        PYTHONPATH="." ./.venv/bin/pytest test

    - Do **not** prepend `$PYTHONPATH` — pass `PYTHONPATH="."` verbatim.
    - `HIPY_STANDALONE_SOURCE` defaults to the repo's `cppbackend/`
      (header location); don't set it.
    - Add `-n auto` to compile/run tests in parallel (pytest-xdist).

    Single file / filter:

        PYTHONPATH="." ./.venv/bin/pytest test/test_hello_world.py
        PYTHONPATH="." ./.venv/bin/pytest test -k fannkuch

    The main agent should invoke ./.venv/bin/pytest (not pytest via an activated venv) so a single approval covers repeated runs, and should drop
     the source .venv/bin/activate && export ... preamble that triggered extra prompts.


## Keeping the docs fresh

If a change makes a doc inaccurate (renamed function, new IR op,
changed pipeline ordering), update the doc in the same change. See the
*Maintenance notes* section at the bottom of `docs/index.md` for a
quick mapping from change type to doc file.
