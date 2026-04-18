# HiPy documentation index

HiPy is a Python-to-IR compiler that extracts high-level Python semantics
into a statically-typed IR suitable for data processing. It is the
artifact of the OOPSLA'24 paper *"Making Pythons out of Snakes: Deeply
Embedding Python's Data Processing Libraries."*

This directory is a **knowledge base for contributors and AI assistants**.
Each file documents one logical component. Read `hipy_paper.md` in the
repo root first for the scientific foundation; these docs focus on the
codebase.

## Start here

- **`setup.md`** — Prerequisites, `.venv`, CMake configure, env vars,
  running `pytest` and `compile.py`. Read this before touching the code
  if the repo isn't already built on your machine.
- **`extending.md`** — How to grow HiPy. Library shims, new virtual
  types, IR builtins, optimization passes, new backends. The primary
  reference for "I want to add X."

## Component references

| File | Covers | Key source files |
|---|---|---|
| [`setup.md`](setup.md) | Build + run recipe: venv, pyarrow symlinks, CMake, env vars, `pytest`, `compile.py` | `Dockerfile`, `.github/workflows/test.yml`, `cppbackend/CMakeLists.txt` |
| [`ir.md`](ir.md) | The IR data model — types, SSA, ops, modules, serialization | `hipy/ir.py` |
| [`compiler.md`](compiler.md) | AST rewrite (cogen) and the full IR-generation pipeline | `hipy/compiler.py` |
| [`context.md`](context.md) | The `Context` runtime: event streams, transactions, control-flow (`_if`/`_while`/`_for`/`_try`), dispatch methods | `hipy/context.py` |
| [`value.md`](value.md) | Virtual-object model: `Type`, `Value`, `ValueHolder`, universal values (lambdas, modules, classes, functions, methods), `CValue` folding | `hipy/value.py` |
| [`intrinsics.md`](intrinsics.md) | Library-author API: `call_builtin`, `bind`, `isa`, `typeof`, `only_implemented_if`, etc. | `hipy/intrinsics.py` |
| [`runtime-glue.md`](runtime-glue.md) | Decorators, `HLCFunction`/`HLCMethod`, closure machinery, test helpers | `hipy/__init__.py`, `decorators.py`, `function.py`, `internal_values.py`, `binding.py`, `interpreter.py`, `test_utils.py` |
| [`standard-library.md`](standard-library.md) | The `hipy/lib/` shims — builtins (int/str/list/dict/…), `_tabular` (column/table), numpy, pandas, sklearn, math, statistics, pickle, urllib, collections | `hipy/lib/**` |
| [`optimizations.md`](optimizations.md) | Pattern rewriter, canonicalization, DCE, inlining, `eager_free`, array/tabular fusion | `hipy/opt/**` |
| [`cpp-backend.md`](cpp-backend.md) | The C++ backend — IR→C++ codegen, Arrow integration, numpy ndarray, runtime headers, CMake build | `hipy/cppbackend/**`, `cppbackend/*.h` |
| [`mlir-backend.md`](mlir-backend.md) | Prototype MLIR backend via LingoDB — subset-only, research-only | `hipy/mlirbackend/__init__.py` |
| [`tests.md`](tests.md) | Test layout, `check_prints` idiom, `not_constant` usage | `test/**` |

## Reading order for new contributors

1. `hipy_paper.md` (repo root) — the *why*.
2. `setup.md` — get it building and tests green.
3. `ir.md` — the data model.
4. `compiler.md` — Python → program generator.
5. `context.md` — how the generator runs.
6. `value.md` — virtual objects.
7. `intrinsics.md` — library-author API.
8. `runtime-glue.md` — decorators and closures.
9. `standard-library.md` — concrete patterns.
10. `optimizations.md` — IR transformations.
11. `cpp-backend.md` — lowering to runnable code.
12. `tests.md` — exercising your work.
13. `extending.md` — adding your own component.

## Conventions in these docs

- Code references use `file_path:line_number` so the editor can
  jump directly.
- Class and function names are `monospace`.
- Paper section references (§4.5 etc.) point to `hipy_paper.md`.
- Every "Gotchas" section captures non-obvious invariants; if a test
  mysteriously fails, check those first.

## Maintenance notes

These docs were written against a snapshot of the codebase; when you
make non-trivial changes, update the relevant doc in the same PR.
Specifically:

- New IR ops → `ir.md`.
- New optimization pass → `optimizations.md` + pipeline ordering in its §6.
- New virtual type → `value.md` and/or `standard-library.md`.
- New shim module → `standard-library.md`.
- New backend → new file + link here.
- New language construct (rewriter case) → `compiler.md` §3.
- Change to build deps, env vars, or CI → `setup.md` + sync
  `.github/workflows/test.yml`.

The `extending.md` cookbook should grow with any new "how do I add X?"
story a contributor asks.
