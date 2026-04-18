# Tests — `test/`

Pytest test tree covering HiPy end-to-end. Every test compiles a
`@hipy.compiled_function` through the full pipeline (cogen → IR →
optimization → C++ backend → execute) and asserts the stdout matches an
expected string.

The canonical test helper is `hipy.binding.check_prints(fn,
expected_str, fallback=False, debug=None)` (see `runtime-glue.md §5`).
Every top-level test file imports it as `from hipy.interpreter import
check_prints` (note: the import path in tests is
`hipy.interpreter.check_prints`, which re-exports from `hipy.binding`).

## 1. Layout

```
test/
├── test_alias.py               — value aliasing / escape analysis
├── test_bool.py                — boolean ops + _const_bool folding
├── test_builtin_funcs.py       — len, sum, min, max, print, repr, sorted
├── test_bytes.py               — bytes literal and ops
├── test_call.py                — HLCFunction dispatch, recursion, kwargs
├── test_call_py.py             — fallback path: calls into CPython via pyobj
├── test_dict.py                — dict mutation, iteration
├── test_for.py                 — for loops (const-for + abstract-for)
├── test_global_const.py        — hipy.global_const pinning
├── test_hello_world.py         — smoke test
├── test_if.py                  — if/elif/else, branch merging, early return
├── test_lambda.py              — 3-variant lambda, closures, bind
├── test_list.py                — list mutation, _concrete_list
├── test_namedtuple.py          — collections.namedtuple (_specialized_named_tuple)
├── test_none.py                — VoidValue handling
├── test_numeric.py             — int/float arithmetic, casts
├── test_print.py               — print formatting
├── test_range.py               — range iter
├── test_statistics.py          — hipy.lib.statistics
├── test_string.py              — str methods, _const_str folding
├── test_tabular.py             — hipy.lib._tabular primitives
├── test_try.py                 — try/except with closure records
├── test_tuple.py               — heterogeneous records + constiter
├── test_while.py               — while loops, fix-point type inference
│
├── examples/                   — end-to-end benchmarks from the paper
│   ├── test_check_iban.py
│   ├── test_fannkuch.py        — classic bench (uses math.fact, list swap)
│   ├── test_levenshtein.py
│   ├── test_schulze.py         — Schulze method (multi-dim arrays)
│   └── test_taq.py             — trade-and-quote (pandas workload)
│
├── numpy/                      — numpy shim
│   ├── examples/
│   ├── test_array_construction.py
│   ├── test_array_ops.py       — elementwise add/sub/mul/div fusion
│   ├── test_array_reshape.py
│   ├── test_array_subscript.py
│   ├── test_math_funcs.py      — sin/cos/sqrt/log/exp/isnan
│   └── test_scalar_types.py    — int64/float64 casts
│
├── opt/                        — optimization passes
│   ├── test_array_patterns.py  — fusion of array.compute
│   ├── test_dead_code_elimination.py
│   ├── test_eager_free.py      — verifies ir.Free insertion
│   ├── test_inline.py          — inlining + subsequent DCE
│   └── test_pattern_rewriter.py — homegrown fold-const-add pattern exercising the rewriter
│
├── pandas/                     — pandas shim
│   ├── examples/
│   ├── test_df.py              — from_dict / constructor / merge / groupby
│   ├── test_series.py          — Series ops + index coherence
│   └── test_series_str.py      — .str accessor
│
├── sklearn/
│   └── test_lm_inference.py    — LinearRegression / LogisticRegression predict
│
└── urllib/
    └── test_parse.py
```

## 2. Idiomatic test shape

Every test file follows this pattern:

```python
import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
# plus optional: import hipy.lib.<module>  — required to register the shim

@hipy.compiled_function
def fn_some_name():
    x = not_constant(5)        # defeat constant folding
    print(x + 3)

def test_some_name():
    check_prints(fn_some_name, """8""")
```

Key conventions:

- **`not_constant(val)`** (from `hipy/test_utils.py` → exported as
  `hipy.interpreter.not_constant`) wraps the value and forces abstract
  materialization via `value.as_abstract(_context)`. Without it, every
  literal folds at generation time and you end up testing the constant
  folder, not the runtime.
- **Type annotations** like `t: bool = True` are used to *prevent*
  compile-time folding of a specific literal. Contrast with `t = True`
  which yields a `_const_bool`. See `test_if.py::fn_if_tmp_vars_ok` for
  the pattern.
- **Expected strings are triple-quoted** and typically start with a
  leading newline. `check_prints` strips whitespace for comparison.
- **Shim imports are required.** `import hipy.lib.numpy` (side-effect:
  registers in `mocked_modules`) must happen before compiling a function
  that uses numpy. Most test files import the shim explicitly at the top.
- **`raw_module(pd)`** gets around the shim for cases where the test
  needs the *real* pandas module (e.g. for the test harness's own
  expected-value computation). See `test/pandas/test_df.py` header.

## 3. Using `check_prints`

```python
check_prints(fn_hello_world, """
hello world
hello world""")
```

- Runs `hipy.compiler.compile(fn, fallback=fallback, debug=debug)`.
- Invokes the C++ backend's `run(fn_name, module)`.
- Captures stdout and compares (leading/trailing whitespace stripped) to
  the expected string.
- `fallback=True` (the default in some tests) allows the generator to fall
  back to `pyobj` on unsupported ops. Pass `fallback=False` to assert that
  everything compiles natively.
- `debug=True` is equivalent to setting `HIPY_DEBUG=1`; enables the
  generator's sanity checks.

### Failure modes

- If the generated C++ fails to compile, CMake prints errors and
  `check_prints` raises. Grep the test output for `error:`.
- If the binary compiles but its output doesn't match, the diff is shown.
- If compilation succeeds but the expected string is wrong, it's a test
  bug — fix the string. Don't silently widen the match.

## 4. Pattern-rewriter tests

`test/opt/test_pattern_rewriter.py` doubles as a live usage example:

```python
class FoldConstAdd(RewritePattern):
    def rewrite(self, op, rewriter):
        match op:
            case ir.CallBuiltin(name="scalar.int.add",
                                args=[ir.SSAValue(producer=ir.Constant(v=a)),
                                      ir.SSAValue(producer=ir.Constant(v=b))]):
                rewriter.replace_with(op, ir.Constant, v=a+b, t=op.result.type)
                return True
        return False
```

Compile `fn`, run the pattern rewriter, assert `str(module)` contains
the expected folded constant. This is the pattern you follow when adding
a new optimization pass — see `optimizations.md` for the driver API.

## 5. `examples/` — paper benchmarks

Directly correspond to the workloads in the OOPSLA'24 paper §6
("Evaluation"):

- **`test_fannkuch.py`** — canonical list-permutation microbenchmark.
  Exercises `range`, `for`, `while`, nested loops, list indexing,
  integer arithmetic.
- **`test_levenshtein.py`** — edit distance; dynamic programming over
  a 2D list.
- **`test_schulze.py`** — Schulze method for voting; matrix workload
  over numpy arrays.
- **`test_taq.py`** — trade-and-quote; pandas groupby / join /
  timestamp arithmetic.
- **`test_check_iban.py`** — string operations; exercises the IBAN
  checksum algorithm.

When you add a new end-to-end feature that the paper doesn't cover,
the `examples/` dir is where it should be exercised.

## 6. Running

```bash
pytest test/                   # full suite
pytest test/test_hello_world.py::test_hello_world
pytest test/ -k "fannkuch"     # filter
HIPY_DEBUG=1 pytest test/      # enable debug asserts in Context
```

The suite assumes the C++ backend is buildable — see
`cpp-backend.md §9` for CMake setup. Missing Apache Arrow / pybind11
manifests as `check_prints` failing on the first test that invokes the
backend.

## 7. Gotchas

- **`check_prints` is the only production-grade runner.** It lives in
  `hipy/interpreter.py` and drives the C++ backend. Don't invent
  alternative runners for new tests.
- **Every shim test must import its shim module** before compiling, or
  the generator silently uses the real library and tests accidentally
  pass via CPython fallback. If a test starts failing after "refactoring"
  a shim, check that the import is still there.
- **`not_constant(…)` is the scalpel for testing the IR path.** Without
  it, most tests reduce to a constant-fold and exercise nothing. Use it
  liberally on every input.
- **Type annotations prevent folding.** `x: int = 5` produces an abstract
  int; `x = 5` produces a `_const_int(5)`. Useful when you want to test
  a specific IR path without `not_constant`.
- **`HIPY_DEBUG=0`** disables `context.seen` (duplicate-wrap detection) —
  production uses this; tests usually leave it on for safety.
