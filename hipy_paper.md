# HiPy — Condensed Paper Reference

Source: *HiPy: Extracting High-Level Semantics from Python Code for Data Processing*
(Jungmair, Engelke, Giceva — PACMPL/OOPSLA 2024, DOI 10.1145/3689737).
LaTeX lives in `./oopsla-paper/main.tex`. This file is a working summary, written for
future maintenance of the repo; it paraphrases the paper and maps its concepts onto
the code.

## 1. Problem and Goal

- Python is pervasive in data analytics / engineering / ML, but its dynamic
  semantics make it a black box for data-processing systems. That blocks
  **logical optimizations** (predicate pushdown, parallelization, constant
  folding across UDF/query boundaries) and forces slow interpreter execution.
- Prior work either:
  - speeds up Python (PyPy, Pyston, GraalPy, Cython, Nuitka) — does not expose
    high-level semantics;
  - specializes a subset (Codon, Numba, Hope, Tuplex) — narrow domain, usually
    compiles just above LLVM, cannot express high-level operators;
  - supports embedded DSLs via AST compilation, tracing, or virtualization
    (TorchScript, torch.fx, Autograph, Grizzly, AFrame, Weld) — captures only
    DSL-related operations on traced objects, not the whole program.
- **HiPy's goal:** extract the *entire* high-level semantics of general Python
  functions into a statically-typed, high-level IR usable for domain-specific
  optimization and code generation, while keeping Python's exact semantics and
  supporting full Python via a fine-grained fallback to CPython.

## 2. Core Methodology — "Generate a Program Generator"

### 2.1 Conceptual basis (adapted 1st Futamura projection)

- Classical: `P_C := mix(int, P)` — partial evaluation of an interpreter on a
  program yields a compiled program.
- Python-adapted: split every object `O` into a static part `O^S` (methods,
  static attrs) and dynamic part `O^D` (runtime values). A function
  specializes on the static parts:
  `f_spec := mix(Python, f, env, args^S)`.
- Instead of writing a monolithic `mix` for Python, use **cogen** — transform
  a program `P` into a program generator `pgen` that emits IR when run:
  `f_pgen := cogen(Python, f)`.
- `cogen` applied to a class produces a **virtual class**; instances are
  **virtual objects** whose methods emit IR instead of computing values.

### 2.2 Key design decisions

- **Explicit annotations only** — only decorated entry functions are captured;
  entry arg types are assumed known at IR-gen time. Writes to globals/modules
  are out of scope.
- **High-level, statically-typed IR** — simple enough for back-ends, high-level
  enough for logical opts (arrays, tables, columns as first-class types).
- **Modular, plain-Python implementation** — library support is idiomatic Python,
  so library authors need no compiler background.
- **Fine-grained fallback to CPython** — unsupported features / diverging types
  are transparently lowered to `pyobj` ops; much finer than PyTorch's function-level
  ignore.
- **Careful handling of mutable objects** — explicit transitions + post-mortem
  escape analysis to avoid desync between multiple representations of one
  logical object.

## 3. Target IR (`hipy/ir.py`)

- Module structure: `imports` (like Python `import`), inline Python source
  (for fallback bodies), and a list of **functions**.
- Function = args + single result; body is a list of SSA operations ending in
  `return`.
- Very few core op kinds: constants, `py_import`/`py_get`/`py_set`/`py_call`
  (Python interop), `if`, and a catch-all **`builtin`** op keyed by string
  (e.g. `"int.add"`, `"string.substr"`, `"list.sort"`, `"array.compute"`,
  `"table.map"`, `"table.filter"`, ...). Back-ends implement whatever set they
  need; anything unsupported falls back.
- **No φ-nodes.** Control flow is only the structured `if` (with two nested op
  lists yielding results) and built-ins that take callback functions
  (e.g. `range.iter`, `list.sort`, `table.apply_row_wise`). Keeps IR / analysis
  simple but still compiles to efficient machine code.
- Types (see Table in §4 of paper):
  `pyobj`, `void`, `bool`, `int` (variable-width), `iN` (fixed-width),
  `f32`/`f64`, `str` (bytes+str unified), `record[name:type,...]`,
  `list[t]`, `dict[k,t]`, `array[shape x type]`, `column[type]`,
  `table[name:type,...]`, `function_ref(...)->t`.
- `pyobj` is the opaque reference-counted CPython object — anchor for the
  fallback path.

## 4. Implementation (`hipy/`)

### 4.1 `cogen` transformation (`hipy/compiler.py`, `hipy/decorators.py`)

Four stages applied when a function is decorated:
1. Retrieve source with `inspect`, parse with `ast`.
2. Desugar constructs (e.g. list comprehensions → explicit loops).
3. Rewrite the AST per the transformation rules (paper Table, §5).
4. `compile()` back to a Python function that now takes an extra `ctx` param
   of type **`GeneratorContext`**.

Representative rewrites (paraphrased):
- `var` → `ctx.get_var("var")`
- `l op r` → `ctx.binary_op(⟦l⟧, ⟦r⟧, "__op__", "__rop__")`
- `f(v…, k=x…)` → `ctx.call(⟦f⟧, [⟦v⟧…], {"k":⟦x⟧,…})`
- `if cond: ibody else: ebody` → wrap each branch as nested fn
  `f_if`/`f_else`; call `ctx._if(⟦cond⟧, [changed…], f_if, f_else)`.
- `for e in l: body` → `ctx._for(⟦l⟧, [read_only…], [iter_vals…], f_loop)`.
- `try/except` → analogous to `if` with `ctx._try`.
- `return e` at top level → `return ⟦e⟧`; nested returns → raise
  `EarlyReturn(⟦e⟧)`, caught at call site. Avoids modelling Python-level
  non-local control flow in IR.
- Lambdas: produce three variants — `f_pgen` (called from generator code),
  `f_closure` (extra `__closure__` record arg, materialized into an IR
  function when passed to a built-in), and `f_py` (unrewritten, used if the
  lambda must be passed to CPython). Bound vars are captured via
  `__closure__["c"]` for the latter two.

### 4.2 Virtual objects & types (`hipy/value.py`, `hipy/internal_values.py`)

- `VirtualType` — abstract `ir_type()` + `construct(ir.Value)`; holds high-level
  type info not in the IR (e.g. `range` vs `tuple` both map to `record`).
- `VirtualObject` — abstract base; every instance corresponds to one
  `ir.Value` via `__irval__()`.
- Method annotations:
  - `@pgen` — raw: not rewritten; body emits IR explicitly via `ctx.gen_builtin`,
    raises `NotImplementedError` to trigger fallback.
  - `@cogen` — rewritten by the transformer; can call other `@cogen`/`@pgen`
    methods. Use when the logic can be expressed in terms of already-defined
    virtual ops (e.g. `__iadd__` calling `+`).
  - `@cogen_class` — marks a class whose members get the appropriate treatment.
- Examples: `_int`, `_list`, `_const_int`, `_concrete_list`, `pyobject`,
  `IntType`, `PyObjType`, ...

### 4.3 Fine-grained fallback

Triggered by:
1. **Unannotated Python function / unsupported module** — call routed through
   `py_call` on the runtime-imported module.
2. **`NotImplementedError` inside a `@pgen` method** — caught by
   `GeneratorContext`, arguments converted via `__topython__()`, operation
   re-emitted on `pyobj`.
3. **Diverging types in control flow** — if the two branches yield different
   types, consolidate to `pyobj`.

Fallback granularity: default is the finest op that failed, but if we are
inside a function/method that has a Python equivalent (e.g. `list.append`),
the whole method is retried as a Python call.

`binary_op` semantics in `GeneratorContext`:
```
try left.__op__(right, self)
except NotImpl: try right.__rop__(left, self)
except NotImpl: convert left to pyobject and retry
```

### 4.4 Eager execution / lazy IR (`hipy/internal_values.py` — `_const_*`, `_concrete_*`)

- Virtual objects normally hold exactly one IR value; **unmaterialized**
  subclasses lift that invariant — they hold Python-side state (e.g.
  `_concrete_list` stores a Python `list` of virtual objects;
  `_const_int` stores a Python `int`).
- `__irval__()` on an unmaterialized object *materializes* it: emits the
  required IR ops and transitions the object to its materialized class.
- Enables:
  - Inconsistently typed containers like `[1, "a"]` without immediate `pyobj`
    fallback (common because deconstructed shortly after).
  - Constant folding (`1+1` stays as `_const_int(2)`).
- Contrast with lightweight modular staging: we don't flag values as
  "staged", eager execution is an *optimization* for selected methods — avoids
  cross-stage persistence problems.

### 4.5 Transitions of mutable objects — `VirtualObjectHolder`

Three state classes: Unmaterialized → Materialized → `pyobject`.

For mutable objects (lists, dicts, ...) the same logical value may need to
exist as both a concrete and a `pyobject` representation at different moments.
Naive impl: two virtual objects ⇒ updates to one don't reflect in the other.

**Solution:** wrap every virtual object in a `VirtualObjectHolder` whose
reference is swapped on transition. Because cogen rewrites all code, we
control every access, so the indirection is transparent.

### 4.6 State management — transactions + type inference

- IR gen has side-effects (emitting ops, mutating unmaterialized state, running
  transitions). Branches that may not actually execute (both sides of `if`,
  loop bodies during fixpoint, try/except) are wrapped in a
  `with ctx.transaction() as T:` that captures changes and commits or aborts.
- `if` yields: both branches evaluated in separate transactions; side-effects
  merged afterwards; resulting IR is one `if` op.
- **Type inference** only needed where CFG edges merge: after `if`, and across
  loop iterations. Each virtual object provides `__merge__(other)` returning
  `(common_type, conv_left, conv_right)`; if no merge is possible, falls back
  to `pyobject`.
- Loops: fixpoint iteration — run body, merge, convert pre-loop values,
  discard side-effects; only when types stabilize is the loop actually emitted.

### 4.7 Post-mortem escape analysis (`hipy/opt/eager_free.py` likely related)

Problem: nested mutables + partial conversion desync. Example:
```python
inner = [0]; outer = [inner]
bar(outer)       # outer → pyobj
print(outer)     # [[42]]
print(inner)     # would print [0] — inconsistent!
```
Cannot be detected cheaply up-front. Algorithm:
1. Generate IR optimistically while recording `Alias`, `Nested`, `Conversion`,
   `Usage` events with per-object **stable identifiers** (tuple of call-site
   locations — identifies an allocation site across generation reruns).
2. Build a Union-Find over objects plus a special `invalid` node.
   - `Nest(a,b)` → union(a,b)
   - `Convert(x)` → union(x, invalid)
   - `Usage(x)` → if `Find(x)==Find(invalid)`, `x` should have been converted
     up-front → record it in `early_conversions`.
3. If any detected, rerun IR generation with those sites forced to convert
   eagerly; repeat until fixpoint.

### 4.8 `intrinsics` module (`hipy/intrinsics.py`)

Helpers callable from `@cogen` code that remove the need to hand-write
`@pgen` boilerplate:
- `intrinsics.call_builtin(name, result_type, args)`
- `intrinsics.not_implemented()` (raises, triggers fallback)
- `intrinsics.isa(obj, type)`
- `intrinsics.bind(fn, arg_types)` — materialize a lambda into an IR function
- `intrinsics.to_python(v)`
- `intrinsics.undef(type)`

This is how most of the standard-library support is written.

## 5. Supported libraries (`hipy/lib/`)

Partial coverage is fine — anything unsupported falls back. Implemented
today:

| Module | Coverage |
|---|---|
| `builtins` | bool, int, float, str, list, dict, tuple, range, print, min, max, sum, sorted, … |
| `numpy` | `ndarray`, `int64`, `float64`, `ones`, `zeros`, `array`; element-wise ops compile to `array.apply` / `array.binary_op` |
| `pandas` | `Series`, `DataFrame`, `RangeIndex`, `Index`, `MultiIndex`; semantically exact; `DataFrame.apply` avoids building per-row `Series` when only subscript is used (unmaterialized series → lookups at generation time) |
| `scikit-learn` | `LinearRegression`, `LogisticRegression`, `Pipeline`, `KMeans`, `MinMaxScaler` — `pickle.loads` unpickles at generation time, dispatches to `cls.__loadfrom__(live_obj)` which produces an unmaterialized virtual model |
| `urllib` | `urlparse` |
| `statistics` | `mean`, `stdev` |
| `scipy` | `special.erf` |
| `pickle` | `loads` |

`builtins` is injected into the global namespace of every transformed
function so names resolve to virtual classes automatically.

## 6. End-to-end pipeline (`hipy/opt/`, `hipy/cppbackend/`, `cppbackend/`)

### 6.1 Optimization passes

- **General**: DCE, function inlining, canonicalization (e.g. pack/unpack
  record elimination) — `hipy/opt/inline.py`, `eliminate_dead_code.py`,
  `canonicalization.py`, `pattern_rewriter.py`.
- **Array fusion** (`array_patterns.py`): lower `array.binary_op` /
  `array.unary_op` into `array.compute(arrays…, scalar_fn)`; fuse chains.
- **Tabular rewrites** (`tabular_patterns.py`, `dccg.py`): rewrite
  column-centric ops (get_column / column.apply / set_column / filter_by_col)
  into table-centric `table.map` / `table.filter` / joins; then extract a
  relational-operator tree for query-style optimizations (predicate pushdown,
  etc.) and fuse pipelines data-centrically à la Neumann 2011.
- `rewrite_cpp.py` — pre-backend lowering.

### 6.2 C++ back-end (`hipy/cppbackend/`, runtime in `cppbackend/*.h`)

- Standalone C++ emitted via string templates (see `hipy/cppbackend/templates/`).
- Also emits code that sets up the embedded CPython interpreter (imports,
  inlined Python defs for fallback bodies).
- Runtime uses C++ stdlib for primitives + custom runtime for:
  - strings with extra methods (`cppbackend/builtin.h`);
  - n-dim arrays with stride/view support (`builtin_numpy.h`);
  - tabular data as **Apache Arrow** columnar buffers (`builtin_arrow.h`);
  - date/time (`builtin_date.h`);
  - `pybind11` for CPython interop.
- Design not locked to C++: `hipy/mlirbackend/` is a stub for a future MLIR
  back-end that could unlock richer optimizations.

## 7. Evaluation highlights

Hardware: Xeon Gold 6430, Ubuntu 24.04, GCC 11. Single-threaded. All
experiments run once for warmup + 3 measured runs, median reported, stdev <5%.

- **Data science** (Logs, Zillow, TPCx-AI UC1/UC10 vs pandas/Tuplex/Modin,
  ±Numba/Cython): HiPy (Opt) is 1.8×–18× faster. Zillow benefits from
  string-heavy C++ gen; Logs from filter pushdown enabled by table-centric
  rewrite. Tuplex beats HiPy on Zillow 2× (hand-tuned exactly for that
  benchmark).
- **Scalar UDFs** called from a C++ table scan (IBAN validation, Levenshtein,
  URL parsing, linear-regression inference): always faster than embedded
  CPython; `lm` reaches 2200× because inference compiles down to a few FP
  ops that inline into the scan.
- **Numerical** (Blackscholes, Haversine, laplace, centdiff vs numpy/Numba/Weld):
  matches numpy on vectorized, beats Numba on `laplace` (4× over CPython via
  high-level fusion); Numba still wins `centdiff` (530× vs HiPy 110×).
- **General Python** (mandelbrot, fannkuch, taq, schulze, telco vs
  PyPy/GraalPy/Codon/Cython): 1.7×–27× speedups on the 3 fully supported
  benchmarks, often beating Codon/PyPy/Cython. Mandelbrot and Telco are
  heavy-fallback (complex / decimal not implemented yet) — perf stays roughly
  at CPython level, confirming that frequent fallback does not catastrophically
  regress.

## 8. Limitations

- **Currently unsupported Python features**: coroutines, class definitions
  (in user code), type aliases, `with`, pattern matching, generators, scoping
  keywords (`nonlocal`/`global`), string templates, decorators.
- **Pathological generation time** possible in the fixpoint loops — handled by
  iteration counters + fallback.
- **Assumes no monkey-patching** — if methods/classes are mutated at runtime,
  extracted IR can diverge from actual semantics. Not typical in data pipelines.

## 9. Developer / user experience notes

- End users keep using plain CPython for prototyping and debugging; HiPy is
  only engaged at compile time. Bugs at the C++ layer are rare.
- Everything (incl. library replacements) is written in idiomatic Python; the
  AST rewrite preserves source locations so Python's debugger works on the
  transformed generator code.
- Deliberately trades a bit of compile time for maintainability vs compilers
  written in C++ (e.g. Codon).

## 10. Security trade-offs

- No automatic fallback + whitelisted builtins ⇒ only IR-expressible ops are
  generated — safe, less compatible.
- Trusted-library list + Python's `audit` hooks ⇒ more compatibility with
  controllable risk. Sandboxing Python fully is infeasible (C extensions,
  `gc`, `sys`, etc.), so VM/container isolation remains the belt-and-braces
  option.

## 11. Repo-to-paper cheat sheet

| Paper concept | Code |
|---|---|
| `cogen` decorator / AST rewrite | `hipy/decorators.py`, `hipy/compiler.py` |
| `GeneratorContext` (`_if`, `_for`, `_try`, `binary_op`, `call`, `transaction`, `gen_builtin`) | `hipy/context.py` |
| IR data model (`Module`, `Function`, `Value`, `Op`, types) | `hipy/ir.py` |
| Virtual object / type base classes | `hipy/value.py` |
| Built-in virtual `_int`, `_list`, `pyobject`, `_const_*`, `_concrete_*` | `hipy/internal_values.py` |
| `intrinsics` helpers | `hipy/intrinsics.py` |
| CPython embedding at runtime | `hipy/binding.py`, `cppbackend/*.h`, `hipy/interpreter.py` |
| Transformed-function runtime | `hipy/function.py` |
| Standard-library shims | `hipy/lib/{builtins.py,numpy,pandas,sklearn,urllib,math.py,statistics.py,scipy,pickle.py,collections,_tabular.py}` |
| Optimization passes | `hipy/opt/{inline,canonicalization,eliminate_dead_code,eliminate_dead_symbols,pattern_rewriter,array_patterns,tabular_patterns,dccg,eager_free,rewrite_cpp}.py` |
| C++ back-end (IR → C++) | `hipy/cppbackend/{cppir.py,templates/}` |
| C++ runtime (Arrow, numpy views, pybind11, string ops, dates, JSON) | `cppbackend/*.h` |
| MLIR back-end stub | `hipy/mlirbackend/` |
| Tests mirroring paper features | `test/test_*.py` (+ `test/{numpy,pandas,sklearn,urllib,opt,examples}/`) |
| Driver / CLI | `compile.py`, `Dockerfile` |

## 12. Things to remember when maintaining

- **Keep semantics strict.** A major selling point vs Grizzly/Weld/TorchScript
  is that Python semantics are preserved; fallback is always the safe
  escape hatch. Do not introduce "close enough" shortcuts without an
  explicit opt-in.
- **Prefer `@cogen` + `intrinsics` over `@pgen`.** Raw `@pgen` is powerful
  but easy to get wrong (esp. around transactions and transitions).
- **Mutable types need `__topython__` and participate in escape analysis.**
  Any new mutable virtual class must: implement `__topython__`; update the
  `VirtualObjectHolder` on transition; emit `Nest`/`Convert`/`Usage` events
  where applicable.
- **`__merge__` is required** on any type that can flow through merging CFG
  edges (if/loop). Missing it ⇒ silent `pyobject` fallback on type divergence.
- **IR extensions are cheap** — add a new `builtin` name + back-end handler,
  fall back otherwise. Avoid adding new top-level IR ops unless they change
  structure (like `if`).
- **Tests in `test/` are the canonical behavioral spec** and cover feature
  families (`test_if`, `test_lambda`, `test_for`, `test_tabular`, …). New
  behavior should come with a matching test.
- **Paper evaluation workloads** live under `test/examples/` (inferred from
  names; verify before citing) — useful regression benchmarks.
