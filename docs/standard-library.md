# Standard library shims — `hipy/lib/`

**Files:** `hipy/lib/builtins.py`, `hipy/lib/_tabular.py`, `hipy/lib/math.py`,
`hipy/lib/statistics.py`, `hipy/lib/pickle.py`, `hipy/lib/collections/`,
`hipy/lib/urllib/`, `hipy/lib/scipy/`, `hipy/lib/numpy/`, `hipy/lib/pandas/`,
`hipy/lib/sklearn/`.

Every module in `hipy/lib/` is a **replacement** for a real Python package.
The loader (`Context.get_by_name` / `PythonModule`) consults
`hipy.mocked_modules` — a name → module dict populated at import time when a
shim calls `hipy.register(sys.modules[__name__])` with
`__HIPY_MODULE__ = "<real_module>"` set at module top.

> If you add a new shim: `__HIPY_MODULE__ = "foo"` at the top, `hipy.register(sys.modules[__name__])` at the bottom. Otherwise the generator silently uses the real `foo`.

Two authoring tools dominate these files:
- `@hipy.compiled_function` — ordinary Python body, rewritten to `_context.*`
  calls by the cogen (see `compiler.md`).
- `@hipy.raw` + `intrinsics.call_builtin(...)` — the function receives
  `_context` and emits IR directly.

The first is preferred; the second is reserved for methods that need to
inspect `ValueHolder` structure, pattern-match on `Value` subclasses, or
thread closures through `intrinsics.bind`.

## 1. `hipy/lib/builtins.py` — the universe of built-in virtual types

`__HIPY_MODULE__ = "builtins"`. This 1900-line module is the substrate every
other shim builds on; if you are adding a new primitive type, you are almost
certainly extending this file.

### 1.1 Scalar types (immutable)

| Virtual class | IR type | Flags | Notes |
|---|---|---|---|
| `bool` | `ir.bool` | `MUTABLE=False`, `NESTED=False` | `__create__` calls `_context._to_bool(val)`. Has `_const_bool` const variant. |
| `int` | `ir.int` | `MUTABLE=False`, `NESTED=False` | Arithmetic → `scalar.int.<op>`; compare → `scalar.int.compare.<op>`. Has `_const_int`. |
| `float` | `ir.f64` | `MUTABLE=False`, `NESTED=False` | Same shape as int. `scalar.float.<op>` / `scalar.float.compare.<op>`. `_const_float`. |
| `str` | `ir.string` | `MUTABLE=False`, `NESTED=False` | Methods like upper/lower/find/substr/split/startswith/contains → `scalar.string.<op>` builtins. Has `_const_str` + a nested `str._iterator`. Also *hosts* the `HLCGeneratorFunctionValue._iterator` class (per the note in `value.md`). |
| `bytes` | `ir.string` | Same as str | Same IR type under the hood; `_const_bytes`. |

Every `_const_*` class:
- Inherits from `CValue` (for arithmetic/compare folding; see `value.md §6`),
- Sets `__HIPY_MATERIALIZED__ = False`,
- Stores the Python value (`cval` / `_cval`),
- Overrides `__abstract__` to emit `ir.Constant(ir_type, val)` when forced to
  materialize.

This is the same pattern every new scalar type should follow.

### 1.2 Mutable containers

**`list`** — IR type `ir.ListType(element_type)`.
- `__create__` (`@hipy.raw`) converts any iterable by appending into a fresh
  abstract list.
- `append`, `__getitem__`, `__setitem__`, `__len__`, `__contains__`, `pop`,
  `extend`, `sort` → `list.<op>` builtins.
- `__merge__` unifies element types. An empty list with `AnyType` element
  is absorbed by the other side's element type (so `if c: xs = [] else: xs = [1,2]`
  typechecks to `list[int]`).
- `_concrete_list` is an immaterialized compile-time snapshot: stores
  `items: list[ValueHolder]`; `__abstract__` materializes by emitting an
  empty list + `append` per item. Exposes `__constiter__` so `for x in [a,b,c]`
  unrolls at generation time.
- Nested `list._iterator` for abstract iteration.

**`dict`** — IR type `ir.DictType(key_type, value_type)`.
- `__getitem__` / `__setitem__` → `dict.get` / `dict.set`; `__contains__` → `dict.contains`.
- Iteration over keys, items, values via `dict.iter_keys`, `dict.iter_items`,
  `dict.iter_values` and a nested `_iterator` class per view.
- `_concrete_dict` holds a Python dict at compile time plus `_to_insert`
  (deferred dynamic inserts). `__abstract__` emits `dict.create` + inserts;
  falls back to abstract dict when any insert value is dynamic.

**`tuple`** — IR type `ir.RecordType([_elt0, _elt1, …])`. Immutable but
*heterogeneous*: stores `_elts: list[ValueHolder]` + `_element_types`. Key
iteration hook is `__constiter__` (yields each `ValueHolder` at generation
time).

**`set`** — `set.<op>` builtins; same shape as list/dict.

### 1.3 Python-side / fallback types

**`object`** — the `pyobj` catch-all (`__HIPY_MUTABLE__=True`, `NESTED=True`).
- IR type `ir.PyObjType`.
- All binary operators forward to `python.operator.<op>` builtins.
- `__hipy_getattr__` / `__hipy_setattr__` emit `ir.PyGetAttr` / `ir.PySetAttr`
  (**not** the `CallBuiltin` path — these are dedicated ops).
- `__call__` (`@hipy.raw`) tries to introspect the source of a known callable
  for return-type inference, then falls back to emitting `ir.PythonCall`.
- `__merge__` trivially identity-merges.

This is what every `to_python(val)` ultimately lands in — every other virtual
type has a `__topython__` that emits IR to produce an `ir.PyObjType` SSA value
wrapped in `object(...)`.

### 1.4 Pseudo-container types

**`range`, `slice`, `enumerate`** — all three are constructed via
`static_object[...]` (see `value.md §5`) so they're immutable records of
their three fields. `__iterate__` dispatches to `range.iter` / iteration over
a wrapped iterable. `enumerate` internally uses a single-slot mutable list
`[0]` as its counter — exemplifies the compile-time "capture a counter in a
list" trick.

### 1.5 Module-level helpers

`print(*args)` → stringifies + emits `dbg.print`. `repr(val)` tries
`__hipy__repr__` first (non-fallback), otherwise hands off to CPython.
`len(v)` delegates to `v.__len__`. `sum(l)`, `min(…)`, `max(…)`, `sorted(…)`
are all `@hipy.compiled_function`s built on iteration + comparison + list
append. `ord(c)` constant-folds when `c` is a `_const_str`.

### 1.6 The CValue → `_super_<op>` fall-through

Every `_const_*` inherits `CValue`. Its dunders check whether *both* sides
have `cval`; if so, they evaluate the Python expression at generation time.
If not, they call `self._super_<op>(other)` — a handle set up by
`@hipy.classdef` that goes to the *non-folded* implementation on the parent
class, which emits the `scalar.<type>.<op>` builtin. This is why
`_const_int(5) + _const_int(3)` folds to `8` at compile time but
`_const_int(5) + int_from_runtime` routes through `int.__add__` →
`scalar.int.add`.

## 2. `hipy/lib/_tabular.py` — columns, rows, tables

`__HIPY_MODULE__ = "_tabular"`. This is not a real stdlib; it's HiPy's
internal relational backbone used by numpy, pandas, and any future
table-oriented library.

- **`row`** — a `RecordType` with named columns. `__getitem__(name)` (name must
  be const) → `ir.RecordGet`.
- **`column`** — `ir.ColumnType(scalar_type)`. Stores `_element_type`.
  `sequential(num_rows)` static method emits a sequential integer column.
- **`table`** — `ir.TableType(columns_list)`. Immutable
  (`__HIPY_MUTABLE__=False`). Operations all return *new* tables with updated
  schemas:

| Method | Emits |
|---|---|
| `get_column(name)` | `table.get_column` |
| `select_columns(names)` | `table.select` (name list as `attributes`) |
| `set_column(name, col)` | `table.set_column` |
| `filter_by_column(c)` | `table.filter` |
| `sort(by, ascending)` | `table.sort` |
| `get_slice(s)` | `table.slice` |
| `apply_row_wise(fn)` | `table.apply_row_wise_scalar` — uses `intrinsics.bind(fn, [row_type])` to produce a callback function-ref |
| `join_inner(other, left_on, right_on, left_keep, right_keep)` | `table.join_inner` |

The `apply_row_wise` pattern is the archetype for every higher-order
table operation: bind the Python callback to a row-typed argument, pass the
resulting `MaterializedFunction` into the builtin call, let the backend
invoke it per row. See `intrinsics.md §5`.

## 3. Small stdlib shims

These are worth skimming as minimal examples of the shim pattern:

- **`hipy/lib/math.py`** (`"math"`) — only `fact` at present, via
  `intrinsics.call_builtin("math.fact", int, [n])`.
- **`hipy/lib/statistics.py`** (`"statistics"`) — `mean`, `stdev` as plain
  compiled functions that iterate the input.
- **`hipy/lib/pickle.py`** (`"pickle"`) — `loads(data)` uses
  `_context.get_corresponding_class(py_cls)` + the class's
  `__from_constant__` method to *unpickle at generation time*. The pickled
  object's concrete fields become IR constants (rather than calling a real
  `pickle.loads` at runtime). This is how sklearn models enter the compiled
  program.
- **`hipy/lib/urllib/parse.py`** (`"urllib.parse"`) — demonstrates the
  `static_object` / `NetlocResultMixin` pattern for a result type with
  computed properties. `urlsplit`/`urlparse` emit `"urllib.parse.urlsplit"`
  builtins; the result types expose `__hipy_getattr__` for `scheme`,
  `netloc`, `username`, etc.
- **`hipy/lib/scipy/special/__init__.py`** (`"scipy.special"`) — only `erf`.
- **`hipy/lib/collections/__init__.py`** (`"collections"`) — `namedtuple` is
  a factory: at generation time it synthesizes a fresh
  `_specialized_named_tuple` class (subclass of `hipy.test_utils._named_tuple`)
  bound to the given field names, and returns it as an `HLCClassValue`.

## 4. `hipy/lib/numpy/` — NumPy

`__HIPY_MODULE__ = "numpy"`.

### 4.1 Scalar types

- **`int64`** — IR `ir.i64`. Same method set as `int`, with
  `scalar.int.<op>` builtins. `_const_int64` for constants.
- **`float64`** — IR `ir.f64`. `scalar.float.<op>` builtins. `_const_float64`.

Type coercion helpers `_convert_to_dtype(value, expected_dtype)` and
`_convert_to_np_type(value)` handle mixed int/float arithmetic.

### 4.2 `ndarray`

- IR type `ir.ArrayType(dtype.ir_type(), shape)`. Shape is a tuple; entries
  may be `None` for dynamic dimensions.
- `shape` attribute is a `ValueHolder` over a runtime tuple; dynamic
  entries are fetched via `array.dim` builtin.
- Operators (`__add__`, `__mul__`, …) go through `_element_wise(other, fn)`
  which emits `array.binary_op` or `array.apply_scalar` with a bound
  callback.
- `reshape(shape)` → `array.reshape`. `__getitem__`/`__setitem__` handle
  int indexing and slicing; empty/ones/zeros produce fresh arrays.

### 4.3 `_concrete_ndarray`

Immaterialized nested-list snapshot. `__create__` walks the list to infer
shape + dtype; `__abstract__` lowers the nested list to an IR array via
`array.from_nested_list`.

### 4.4 Module functions

`empty(shape, dtype=float64)`, `ones(...)`, `zeros(...)`, `zeros_like(a)`,
`array(nested_list)`, `sin`, `cos`, `arcsin`, `sqrt`, `log`, `exp`,
`isnan`. Float/scalar functions use an internal `_float_function` helper
that applies either to a scalar (`scalar.float.<op>`) or across an array's
elements (`array.apply_scalar` with a bound lambda).

### 4.5 `hipy/lib/numpy/random.py`

Eleven lines. `rand(*shape)` allocates an empty float64 array and fills it
via `array.fill` with a lambda that calls the `random.rand` builtin
element-wise. Use as a template when adding further random functions.

## 5. `hipy/lib/pandas/` — Series, DataFrame, groupby

`__HIPY_MODULE__ = "pandas"`. Sits on top of `_tabular.table`/`column` and
numpy scalars.

### 5.1 `Index` / `MultiIndex` / `RangeIndex`

Wrap a `column` plus metadata (`name`, dtype). Store optional
`_concrete_values` (compile-time Python list) for fast const lookups.
`_columns()` exposes the column dict for joins.

### 5.2 `DataFrame`

Mutable (`__HIPY_MUTABLE__=True`). Holds:
- `_table` — the underlying `_tabular.table`,
- `index` — Index / RangeIndex / MultiIndex,
- `_col_versions: dict[str,int]` — per-column version counter bumped on each
  mutation,
- `_col_types: dict[str, Type]`.

`__abstract__` returns `ValueHolder.AbstractViaPython` — DataFrame cannot be
lowered to a single SSA; it must go through pyobj on any context that
demands an SSA representation. This is why DataFrame flows stay on the
compile-time side as long as possible.

Key methods: `__getitem__(colname)` returns a Series; `__setitem__(colname,
val)` bumps the version, calls `_table.set_column`. `merge`, `fillna`,
`reset_index`, `apply(func, axis=0/1)`, `groupby(by)`. `axis=1` applies use
an internal `_row_to_series` adapter.

### 5.3 `Series`

Mutable. Holds `_data` (column), `index`, optional `_df` (parent), `name`,
`_version` (snapshot of parent's version for `name`), `_concrete_values`.

- `_data` getter: if parent version matches `_version`, re-fetch column
  from parent; else use local copy. This is how Series stays coherent with
  mutations on its parent DataFrame.
- `_concrete_values`-backed materialization: lazy; materializes to a column
  only on first access.
- `.str` / `.dt` return `_StringMethods` / `_DateMethods` accessor objects
  (chainable).

### 5.4 GroupBy

`DataFrameGroupBy` / `DataFrameGroupBySeriesGroupBy` are immutable views.
`agg(func_or_dict)` emits `table.aggregate` with per-column handlers
(sum/min/mean via `zero(col_type)` + reducer lambda). Returns a new
DataFrame with aggregated columns.

### 5.5 `.iloc`

`_iLocDFIndexer` / `_iLocSeriesIndexer` — positional indexers. Slice →
returns a DataFrame/Series; int → returns a row-as-Series or scalar.

### 5.6 Module-level `merge`

Handles inner/left joins on columns and/or indexes. For `how="left"`,
integer columns are promoted to float first so NaNs can be introduced. Emits
`table.join_inner` or `table.join_left`.

## 6. `hipy/lib/sklearn/` — trained models as compile-time constants

Every sklearn wrapper follows the same pattern:

- Inherit from `static_object["field1", "field2", …]` with fields for the
  fitted attributes (`coef_`, `intercept_`, `feature_names_in_`, …) plus
  hyperparameters.
- Implement `__from_constant__(py_obj, context)` — inspect a **fitted
  scikit-learn object** loaded at generation time and lift its arrays via
  `context.perform_call(HLCFunctionValue(np.array), ...)` and scalars via
  `context.constant(...)`. This is what makes sklearn models survive
  `pickle.loads` through HiPy's pickle shim.
- Implement `__topython__` — rebuild a real sklearn object from the IR
  values so a fallback path still works.
- Implement `predict` / `transform` as inlined pure-Python logic (element-wise
  dot products, distance computations, argmax). No sklearn runtime
  dependency in compiled code.

Shipped wrappers:

| Submodule | Class | Algorithm |
|---|---|---|
| `sklearn/pipeline.py` | `Pipeline` | chains `.transform()` through all but last step, then `.predict()` on the last |
| `sklearn/cluster/_kmeans.py` | `KMeans` | inline Euclidean distance to each centroid, argmin |
| `sklearn/linear_model/_base.py` | `LinearRegression` | `intercept_ + sum(e[i] * coef_[i])` per row |
| `sklearn/linear_model/_logistic.py` | `LogisticRegression` | per-class logit, argmax (multiclass) or sign (binary) |
| `sklearn/preprocessing/_data.py` | `MinMaxScaler` | `(col - data_min_[i]) * scale_[i]` per column |

These are the paper's "pickled-then-lifted" examples — inspect them when
adding a new model wrapper.

## 7. The representative IR builtin catalog

Approximate inventory of `CallBuiltin` `fn_name` strings observed across
the shims. Useful when writing a new shim (prefer an existing prefix) or a
new backend (these are what you'll be handling in the IR):

- **Scalars:** `scalar.int.<op>` (add, sub, mul, div, mod, pow, shl, shr,
  and, or, xor), `scalar.int.compare.<lt|le|gt|ge|eq|ne>`,
  `scalar.float.<op>`, `scalar.float.compare.<op>`, `scalar.bool.<op>`,
  `scalar.string.<upper|lower|find|substr|split|startswith|contains|len>`,
  `scalar.int.from_string`, `scalar.float.from_int`, `scalar.int.to_python`.
- **Collections:** `list.<append|at|set|len|pop|extend|iter>`,
  `dict.<get|set|contains|iter_keys|iter_items|iter_values|create>`,
  `set.<add|contains|len|iter>`.
- **Iteration:** `range.iter`, `list.iter`, `dict.iter_keys`, `string.iter`.
- **Tables:** `table.get_column`, `table.set_column`, `table.select`,
  `table.filter`, `table.sort`, `table.slice`, `table.apply_row_wise_scalar`,
  `table.aggregate`, `table.join_inner`, `table.join_left`.
- **Arrays:** `array.create_empty`, `array.fill`, `array.apply_scalar`,
  `array.binary_op`, `array.reshape`, `array.copy`, `array.get`, `array.set`,
  `array.from_nested_list`, `array.dim`.
- **Misc:** `math.fact`, `urllib.parse.urlsplit`, `random.rand`, `dbg.print`,
  `try_or_default`, `python.operator.<op>`, `python.create_dict`,
  `python.create_list`, `python.tuple_from_list`.

Backends must implement every `fn_name` a user's program can reach. A new
builtin introduced by a shim needs a matching handler in the C++ backend
(see `cpp-backend.md`).

## 8. Gotchas for shim authors

- **Forgetting `hipy.register(sys.modules[__name__])`** is the #1 failure mode.
  The module file loads, `@compiled_function`s and classes are visible to
  pytest, but the HiPy generator never substitutes them — it silently uses
  the real library. Always grep for `hipy.register(` when cloning a new shim.
- **`intrinsics.call_builtin(..., side_effects=False)`** is what lets the
  optimizer reorder / DCE the op. Arithmetic is side-effect-free; anything
  that writes `print`, files, or RNG state must keep the default `True`.
- **Row-wise callbacks need `intrinsics.bind`, not a naked lambda.** Passing
  a Python lambda directly to `call_builtin` will raise — `bind` materializes
  it into an `ir.FunctionRef`, which is what the builtin handlers expect.
- **`__HIPY_MATERIALIZED__ = False`** is mandatory on any `_const_*` /
  `_concrete_*` class; otherwise the ValueHolder won't know the value still
  has a compile-time representation and will force early materialization.
- **DataFrame `__abstract__` returns `AbstractViaPython`.** If you add a new
  mutable type that cannot be lowered to a single SSA, use the same sentinel.
  The generator will then always go through `__topython__` when a single
  SSA is required.
- **sklearn `__from_constant__` runs at generation time, not runtime.** Its
  `context` argument is the `Context` — use it to build IR-side values from
  the already-fitted Python model. Parameters that depend on runtime data
  don't belong here.
