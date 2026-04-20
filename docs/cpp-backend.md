# C++ backend — `hipy/cppbackend/` + `cppbackend/*.h`

**Files:**
- `hipy/cppbackend/__init__.py` (~1100 LOC) — the code generator driver
- `hipy/cppbackend/templates/standalone.cpp` — C++ `int main()` template
- `hipy/cppbackend/templates/udf_eval.cpp` — Arrow-IPC UDF-benchmark template
- `cppbackend/builtin.h` — core C++ runtime (containers, strings, helpers)
- `cppbackend/builtin_commons.h` — `bound_fn<>` closure trampoline
- `cppbackend/builtin_arrow.h` — Apache Arrow integration (Column, Table, builders, accessors)
- `cppbackend/builtin_numpy.h` — `ndarray<T,D>` view + materialize
- `cppbackend/builtin_date.h` — date helpers
- `cppbackend/datastructures.h` — hand-rolled `JoinHashTable`, aggregation HT, table builder
- `cppbackend/CMakeLists.txt` — compile rules for the generated program

This backend takes the post-optimization IR module (see `optimizations.md`)
and emits a complete C++ program that, when compiled and run, executes the
HiPy function.

## 1. End-to-end pipeline

1. `compile.py` (CLI) or `hipy/binding.py::check_prints(fn, expected, …)`
   calls `hipy.compiler.compile(fn, arg_types, fallback, debug)` to produce
   an `ir.Module`.
2. The module is further optimized (canonicalize, inline, DCE, array/tabular
   fusion, `eager_free`, `eliminate_dead_symbols`).
3. `hipy.cppbackend.run(fn_name, module)` constructs a `CPPBackend(fn_name,
   module)` and calls its `run()`:
   - `generate_module()` emits forward declarations and function bodies
     for every `ir.Function` in the module.
   - The result is interpolated into the Jinja2 template
     `templates/standalone.cpp` producing a single `standalone.cpp` source.
4. `write_compile_run_cpp` writes the source to a fresh
   `tempfile.TemporaryDirectory`, assembles a compile command directly
   (no CMake) by querying `sysconfig`, `pybind11`, `pyarrow` for
   include / lib paths, and invokes the system C++ compiler once per
   test. `ccache` and `ld.lld` are picked up automatically if in
   `PATH`.
5. The resulting binary is executed as a subprocess with `PYTHONPATH`
   threaded through; stdout/stderr/returncode are captured and returned.
   The tempdir is deleted on scope exit — tests are fully independent,
   so `pytest -n auto` parallelizes safely.

> Tests that use `binding.check_prints` cover this whole loop end-to-end.
> The "standalone binary" pattern is what makes HiPy's integration tests
> authoritative: they run real C++, not just IR.

## 2. `CPPBackend` — the code generator (hipy/cppbackend/__init__.py)

One large class. Members of note:

| Member | Role |
|---|---|
| `fn_name`, `module` | Entry function + IR module. |
| `unique_id` | Monotonic counter for local-var naming (`val_0`, `val_1`, …). |
| `enable_python`, `enable_arrow`, `enable_numpy` | Feature flags surfaced to the template as `PY_ENABLED` / `ARROW_ENABLED` / `NUMPY_ENABLED` macros so the header includes and pybind11 setup are conditional. |
| `global_constants` | Constant string/tuple/record values hoisted to file scope. |
| `global_py_constants` | Python-side constants (imported module globals etc.) declared statically + initialized once at main entry. |
| `generate_type(ir_type)` | Dispatches IR types to C++ types (§3). |
| `generate_op(op)` | Emits one SSA op to C++. Delegates to `generate_builtin` for `ir.CallBuiltin`. |
| `generate_builtin(op)` | ~700-line `match` over `op.name` — the mapping from every IR builtin name to its C++ idiom. Grouped by namespace (scalar, list, dict, string, array, table, column, python, dbg, …). |
| `generate_function(fn)` | Renders function signature + body via Jinja2. Always void return at the module level; values flow through output parameters for tables/columns that own heap data. |
| `generate_module()` | Forward-declare all functions in dependency order, then emit bodies. |
| `run()` | Assembles template inputs + renders. |

Two helper free functions:

- `get_column_builder(t)`, `get_column_accessor(t)` — map element IR types
  to the matching Arrow-builder / Arrow-accessor C++ class
  (`Int64ColumnBuilder`, `StrColumnBuilder`, `BoolColumnAccessor`, …).
- `write_compile_run_cpp(code, debug)` — writes the rendered source to
  a tempdir, invokes the compiler directly, runs the binary, captures
  output.

## 3. Type lowering

`generate_type(ir_type)` dispatches:

| IR type | C++ type |
|---|---|
| `ir.VoidType` | `uint8_t` (one-byte placeholder) |
| `ir.IntType` | `int64_t` |
| `ir.IntegerType(width=w)` | `int{w}_t` |
| `ir.FloatType(width=32)` | `float` |
| `ir.FloatType(width=64)` | `double` |
| `ir.BoolType` | `bool` |
| `ir.StringType` | `std::string` |
| `ir.ListType(E)` | `std::shared_ptr<std::vector<C++(E)>>` |
| `ir.DictType(K,V)` | `std::shared_ptr<std::unordered_map<C++(K), C++(V)>>` |
| `ir.RecordType(ms)` | `std::tuple<C++(t1), C++(t2), …>` — keyed lookup compiles to `std::get<i>(tpl)` |
| `ir.ArrayType(E, shape)` | `std::shared_ptr<builtin::ndarray<C++(E), D>>` where `D=len(shape)` |
| `ir.ColumnType` | `std::shared_ptr<builtin::tabular::Column>` |
| `ir.TableType` | `std::shared_ptr<builtin::tabular::Table>` |
| `ir.FunctionRefType(args, res)` | `std::add_pointer<C++(res)(C++(args)…)>::type` (raw function pointer) |
| `ir.FunctionRefType(args, res, closure=C)` | `builtin::bound_fn<FnPtrType, C++(C)>` — see §4 |
| `ir.PyObjType` | `py::object` |

Key traits:
- Heap-owned data (list, dict, array, column, table) is always
  `std::shared_ptr`. This pairs with the `ir.Free` ops inserted by
  `eager_free` — dropping the last shared_ptr reference runs the
  destructor.
- Records map to tuples. `MakeRecord` with keyed members becomes
  `std::make_tuple(v1, v2, …)` with a fixed index convention that matches
  `ir.RecordGet`'s lookup.
- Function refs with closures use a specialized template `bound_fn`; see
  `builtin_commons.h`.

## 4. Closures — `builtin::bound_fn`

`builtin_commons.h` defines (roughly):

```cpp
template<typename FnPtr, typename Closure>
struct bound_fn {
    FnPtr fn;
    Closure closure;
    template<typename... Args>
    auto operator()(Args... args) { return fn(args..., closure); }
};
```

Every closure-carrying `ir.FunctionRef` is emitted as a `bound_fn{fn,
closure_tuple}`. Callers invoke it exactly like a function — the closure
is appended to the argument list as the callee's final parameter. This is
the C++-level equivalent of `hipy/internal_values.py::lambda_in_closure`
(see `runtime-glue.md §4`).

Compare to what the IR already encoded: `FunctionRef` ops carry a
`closure` field, and every function with a non-empty closure type has an
implicit trailing record argument. The backend just honors that
convention.

## 5. `CallBuiltin` dispatch (the builtin catalog)

`generate_builtin(op)` is a single large `match op.name`. Non-exhaustive
mapping, grouped by prefix:

- **`scalar.<int|float>.<op>`** → C++ binary operators (`+`, `-`, `*`, `%`,
  `/`), `std::pow`, `std::sqrt`, etc. Compare variants (`.compare.lt`, `.eq`,
  …) emit `<`, `==`.
- **`scalar.<int|float|bool>.<to_python|from_python>`** →
  `py::int_(v)` / `v.cast<int64_t>()` etc.
- **`scalar.string.<upper|lower|find|rfind|substr|split|contains|startswith|len>`** →
  helpers in `builtin::string::` (defined in `builtin.h`).
- **`scalar.int.from_string`** → `std::stoll(s)`; `scalar.float.from_string`
  → `std::stod(s)`.
- **`list.<append|at|set|len|iter|pop>`** → `vec->push_back`, `(*vec)[i]`,
  `vec->size()`, index-loop.
- **`dict.<get|set|contains|iter_keys|iter_items|iter_values|create>`** →
  `unordered_map` operations.
- **`array.<create_empty|reshape|fill|apply_scalar|binary_op|compute|get|set|create_view|dim|from_nested_list>`** →
  `ndarray<>::` methods from `builtin_numpy.h`.
- **`table.<get_column|select|set_column|filter|sort|slice|length|filter_by_func|compute|add_index_column|join_inner|join_left|aggregate|apply_row_wise_scalar|from_dict>`** →
  `Table::` and `ColumnBuilder` calls from `builtin_arrow.h` (plus the
  hand-rolled `JoinHashTable` / `AggregationHT` in `datastructures.h`).
- **`column.<apply_scalar|binary_op|filter|aggregate|unique|isin_column|sequential|length>`** →
  chunk-wise Arrow compute + custom column-builder emission.
- **`python.operator.<add|sub|…|eq|lt|gt|contains|is_>`** → pybind11 operator
  overloads on `py::object`.
- **`python.<create_list|create_dict|create_slice|get_none|tuple_from_list>`** →
  pybind11 constructors.
- **`dbg.print`** → `std::cout` / `std::endl` emission.
- **`range.iter` / `while.iter`** — handled here via the generic
  builtin dispatch (no separate lowering pass).

Every new IR builtin a library author introduces (via
`intrinsics.call_builtin`) must get a handler here; otherwise `run()`
fails at generation time with an unrecognized-name error.

## 6. Templates

### 6.1 `standalone.cpp`

Header of the template (placeholder syntax):

```cpp
#define PY_ENABLED {{py_enabled}}
#define ARROW_ENABLED {{arrow_enabled}}
#define NUMPY_ENABLED {{numpy_enabled}}
#include "builtin.h"
#if ARROW_ENABLED==1
#include "builtin_arrow.h"
#endif
#if NUMPY_ENABLED==1
#include "builtin_numpy.h"
#endif
#include <iostream>
…
#if PY_ENABLED==1
#include<pybind11/embed.h>
namespace py = pybind11;
#endif
{{global_constants}}
{{global_py_decl}}

{{method_definitions}}
int main(){
    #if PY_ENABLED==1
    py::scoped_interpreter guard{};
    auto mainModule=py::module_::import("__main__");
    {{init_python}}
    #endif
    #if ARROW_ENABLED==1
    arrow::py::import_pyarrow();
    #endif
    {{global_py_init}}
    {{fn}}();
    {{global_py_deinit}}
    return 0;
}
```

- `{{init_python}}` iterates `module.imports` + `module.py_functions` and
  injects `import X as Y` / `exec("def foo(...): …")` lines so any
  pyobj-level calls at runtime see the same module state CPython saw.
- `{{method_definitions}}` is every generated C++ function.
- `{{fn}}` is the entry function name — called with no arguments at main.
  Args for the HiPy function are loaded as module-scope globals in
  `init_python` before the call (this is the pattern `compile.py` relies
  on for CLI inputs).

### 6.2 `udf_eval.cpp`

A specialization used for the paper's UDF benchmarks: it loads a column
from an Arrow IPC file, times a loop that invokes the compiled UDF on
every row, writes a result column, then prints a JSON blob with `runtime`
(seconds) and `res` (the result column serialized). No Python driver
currently invokes this template — it is retained as documentation of the
UDF-benchmark shape.

## 7. Runtime headers

### 7.1 `builtin.h` + `builtin_commons.h`

Defines `builtin::string::*` helpers (upper/lower/find/substr/split/
startswith), `bound_fn<>`, `std::hash<std::tuple<...>>` specialization
(XOR-combine of member hashes so tuples work as `unordered_map` keys),
and basic arithmetic / stream emission helpers.

### 7.2 `builtin_arrow.h`

Real Apache Arrow integration. Key classes:

- **`Column`** — `std::shared_ptr<arrow::ChunkedArray>` wrapper.
  Methods: `iterate<Accessor>(fn)`, `iterateZipped<A1,A2>(fn)` (handles
  misaligned chunks), `byIndex<A>(i)`, `unique()`, `isIn(other)`,
  `to_python()` via `arrow::py::wrap_chunked_array`.
- **`GenericColumnAccessor<ArrowType>`** — templated typed read from an
  Arrow array. Specializations for every primitive type and for strings
  (`array->GetString(i)`) + `ListColumnAccessor<Elt>` for nested lists.
- **`ColumnBuilder<ArrowBuilder>`** — append-based column construction
  with chunk rollover at 20k rows. Specializations: `Int64ColumnBuilder`,
  `Float64ColumnBuilder`, `StrColumnBuilder`, `BoolColumnBuilder`, etc.
- **`Table`** — `std::shared_ptr<arrow::Table>` wrapper.
  `from_columns(vector<pair<name,Column>>)`, `to_python()` via
  `arrow::py::wrap_table`, `iterateBatches(fn)` via
  `arrow::TableBatchReader`, `sort`/`filter`/`slice` delegating to
  Arrow's compute kernels, `load(path)` for Arrow-IPC files.

All relational operations are expressed as Arrow compute calls or
chunked iteration — there is no bespoke columnar storage.

### 7.3 `builtin_numpy.h`

`ndarray<T,D>` is a **view**: pointer, shape, strides, offset. Heap-held
through `std::shared_ptr<ndarray_data<T>>`.
- `byIndices(array<int64_t, D>)` — strided load.
- `reshape(shape)` — tries to reuse strides; otherwise materializes.
- `create_view(view_info)` — handles fancy indexing and slices.
- `materialize()` — produce a C-contiguous fresh copy.
- `to_numpy()` — wrap in `py::array_t<T>` (materializes if view).

### 7.4 `builtin_date.h`

Minimal date helpers (parsing, comparison). Used by date columns.

### 7.5 `datastructures.h`

Hand-rolled:
- `JoinHashTable<KeysT, ValuesT>` — multi-map; `insert → build (sort
  buckets) → find` pattern suited to batch joins.
- `AggregationHashTable<KeysT, ValueT>` — `contains`-or-insert, then
  per-key mutable accumulator.
- `TableBuilder` plumbing (chained column builders).

Kept separate from Arrow because Arrow's compute kernels don't cover
these exact patterns efficiently.

## 8. Direct compile + output

`write_compile_run_cpp` invokes the system C++ compiler directly (no
CMake). The command, assembled in `_build_compile_command`, is:

```
[ccache] $CXX -std=c++20 -march=native -fvisibility=hidden {-O0 -g | -O3 -DNDEBUG}
    [-fuse-ld=lld if ld.lld is in PATH]
    -I{pybind11.get_include()}
    -I{sysconfig.get_path('include')}
    -I{pyarrow.get_include()}
    -I{HIPY_STANDALONE_SOURCE}        # header bundle (builtin.h, …)
    /tmp/hipy-cpp-XXXX/standalone.cpp
    -o /tmp/hipy-cpp-XXXX/standalone
    -L{pyarrow.get_library_dirs()[0]} -Wl,-rpath,{same}
    -larrow -larrow_compute -larrow_python
    -L{sysconfig LIBDIR} -Wl,-rpath,{same}
    -lpython{sysconfig LDVERSION}
```

Relevant env vars:
- `HIPY_STANDALONE_SOURCE` — header location (defaults to repo's
  `cppbackend/`). There is no longer a build directory.
- `CXX` — compiler override; defaults to `g++`, falls back to `c++`.

`ccache` and `ld.lld` are optional and picked up via `shutil.which`.

## 9. Lifecycle example — pandas GROUP BY

End-to-end for `df.groupby("a").agg("sum")`:

1. Frontend emits `table.aggregate(table, init_fn, agg_fn, finalize_fn,
   group_by=…, input=…, output=…)` via the pandas shim.
2. `opt/tabular_patterns` normalizes any column-level prep.
3. `generate_builtin` dispatches `table.aggregate` directly, emitting
   column builders as locals, a `std::unordered_map` for the
   aggregation HT, an `arrow::TableBatchReader` loop, key-tuple
   construction per row, insert-or-update with the init/agg functions
   inlined as `ir.Call` ops, a final scan loop that calls `finalize_fn`
   per key, appends to builders, then `Table::from_columns`.
4. `eager_free` has already inserted `ir.Free` after the last use of the
   original table so its shared_ptr drops as soon as the scan completes.

All of this runs with no Python on the hot path — pandas is entered only
via `.to_python()` at the final boundary, which calls
`arrow::py::wrap_table()` to produce a zero-copy pyarrow.Table that
pandas can wrap.

## 10. Gotchas

- **`unordered_map` with tuple keys needs a `std::hash` specialization.**
  `builtin.h` provides one via recursive XOR-combine. If you extend the
  IR with a new non-trivially-hashable aggregate, add a matching hash
  specialization.
- **`std::shared_ptr` + shared ownership between columns.** A `Column`
  can be shared across tables; `eager_free` only drops *one* reference.
  This is safe but means heap peak usage is sensitive to when the last
  consumer runs.
- **`py::scoped_interpreter`.** Only one lives, at main(). If a generated
  program triggers pybind11 fallbacks, the interpreter is already up.
  `arrow::py::import_pyarrow()` must be called after the interpreter is
  initialized — the template does this in order.
- **Arrow chunk boundaries.** `iterate()` hides them for one column but
  `iterateZipped()` must resync; when you add a new multi-column
  operation, use `iterateZipped` rather than re-opening two `iterate`s.
- **`ir.Free` order matters.** `eager_free` is the only pass that inserts
  Frees. If you add a pass *after* `eager_free` that duplicates uses,
  you'll read freed memory; add the `Free`s yourself or move your pass
  before `eager_free`.
- **Missing builtin → silent runtime.** Adding a new `intrinsics.call_builtin(fn_name, …)` in a shim **must** be paired with a `case "fn_name":` in `generate_builtin`. Otherwise `run()` raises at codegen time with a "no handler" error. Grep for the existing ones when adding your own.
- **Global Python constants are initialized once, at main entry.** If
  your new builtin needs a Python object at generation time, put it in
  `global_py_constants` via `constant(...)` / `import_pymodule(...)`;
  don't reach into Python from inside a hot loop.
- **CMake caches aggressively.** If the emitted source changes but CMake
  doesn't rebuild, delete the build directory or bump the timestamp.
  `write_compile_run_cpp` touches the source file to help, but clean
  rebuilds are sometimes necessary after a large backend change.
