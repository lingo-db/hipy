# Runtime glue — decorators, function handles, closures, interpreter helpers

**Files:**
- `hipy/__init__.py`
- `hipy/decorators.py`
- `hipy/function.py`
- `hipy/internal_values.py`
- `hipy/binding.py`
- `hipy/interpreter.py`
- `hipy/test_utils.py`

These modules are the **seams** between ordinary Python and the HiPy machinery
— the decorators a user applies, the small wrappers that trigger AST
transformation on demand, and the helpers that test/driver code uses to
exercise a compiled function.

## 1. `hipy/__init__.py` — public surface

```python
from hipy.decorators import compiled_function, raw, classdef, raw_generator
mocked_modules = {}
def register(module): mocked_modules[module.__HIPY_MODULE__] = module
class global_const:
    def __init__(self, value): self.value = value
```

- `mocked_modules` is a name → module dict used by
  `Context.get_by_name` / `PythonModule` to swap in HiPy's replacements for
  `numpy`, `pandas`, `sklearn`, etc. Every `hipy/lib/*` replacement module
  declares `__HIPY_MODULE__ = "numpy"` (or similar) and calls
  `hipy.register(<module>)` to install itself.
- `global_const(value)` is a marker object. When a generator sees a module
  attribute that is a `global_const`, it yields `context.constant(val)` —
  i.e. it's a way to tell HiPy "treat this Python-level value as a compile-time
  constant." Used sparingly to pin specific globals to the generation-time
  path. See tests/test_global_const.py for usage.

## 2. `hipy/decorators.py` — user-facing decorators

Four decorators, each wrapping a callable:

| Decorator | Produces | When to use |
|---|---|---|
| `@hipy.compiled_function` / `@hipy.compiled_function` | `HLCFunction(pyfunc)` — lazy, AST-rewritten on first call | Default: methods in user / library code that should be cogen-transformed. |
| `@hipy.raw` | `HLCFunction(fn, compiled_fn=fn)` — skips AST rewrite | For methods that emit IR directly via `_context.*`. If the function doesn't accept `_context`, the decorator wraps it in a thunk that strips it. |
| `@hipy.classdef` | Class metaprogramming that wraps every `HLCFunction` member in `HLCMethod` via a patched `__init__` | Mandatory on every virtual `Value` subclass. Also preserves super-method access via `_super_<method>` attribute rebinding. |
| `@hipy.raw_generator` | `GeneratorFunction(pyfunc)` | For the (experimental) generator-function support — see `HLCGeneratorFunctionValue` in `hipy/value.py`. |

`@classdef` implementation details worth remembering:
- Iterates `cls.__dict__` collecting attributes that are `HLCFunction`
  instances.
- Replaces each with a `lambda self, *a, **kw: getattr(self, name)(*a, **kw)` —
  so calling `instance.method(...)` dispatches through the instance dict
  below, not through the class.
- Replaces `__init__` to, after running the original init, set
  `_super_<name>` (saved previous binding) and `<name>` (a fresh
  `HLCMethod(func, self)`) on `self` for every wrapped method.
- Marks the class with `cls.__hipy__ = True` so
  `Context.get_by_name` can recognize it.

The `_super_<name>` attribute is the anchor that `CValue._cvalue_binary_op`
uses to call the parent-class implementation after constant folding fails.

## 3. `hipy/function.py` — `HLCFunction`, `HLCMethod`, `GeneratorFunction`

Thin wrappers with lazy AST rewrite:

```python
class HLCFunction:
    def __init__(self, pyfunc, compiled_fn=None):
        self.pyfunc = pyfunc
        self.compiled_fn = compiled_fn
    def get_compiled_fn(self):
        if self.compiled_fn is None:
            self.compiled_fn = stage_and_compile(self.pyfunc)
        return self.compiled_fn
    def __call__(self, *args, **kwargs): return self.pyfunc(*args, **kwargs)
    def get_name(self): return self.pyfunc.__name__
```

- Calling the object directly (`HLCFunction(...)(...)` in ordinary Python)
  runs the **original, unrewritten** function. This is what lets users debug
  with plain CPython.
- Calling through `get_compiled_fn()(...)` runs the rewritten version, which
  expects a `_context` kwarg. This is what `Context.perform_call` invokes.
- `HLCMethod(func, self_value)` pre-binds the receiver; the generator calls
  `func.get_compiled_fn()(self_value, *args, **kwargs, _context=ctx)`.
- `GeneratorFunction` is the same shape for `@raw_generator`.

Note: `GeneratorFunction.get_compiled_fn` has a latent bug — it references
`self.compiled_fn` without initializing it in `__init__`. Treat as
incomplete; usable paths go through `HLCGeneratorFunctionValue` which
does the staging separately.

## 4. `hipy/internal_values.py` — closures and closure-bound lambdas

Two `Value` subclasses plus a `create_closure` factory:

### `closure_record` (a `Value`)
A variadic record keyed by name, with special handling for void-typed
fields (stored separately in `is_void` rather than taking up a record slot).
`ir_type()` is:
- `ir.void` if empty,
- the single field's IR type if len==1,
- `ir.RecordType([...])` otherwise.

Implements `__getitem__` (for reading closure variables from generated code)
and `__hipy_get_type__`. Does NOT implement `__topython__` — raising
`NotImplementedError` so callers know closure records stay inside HiPy.

### `lambda_in_closure` (a `Value`)
The pair `(closure_record, staged_fn)` — a lambda bound to its closure. Its
`__call__` invokes the staged fn with `__closure__=self._closure_value`.
Produced by `create_closure` below.

### `create_closure(closure_dict, _context)`
Takes a Python dict of `{name: ValueHolder}` and returns a `ValueHolder`
wrapping a `closure_record`:

- For values that are already-materialized or mutable or contain nested
  objects: keep as-is.
- For purely constant values (immutable, unmaterialized, no nested): wrap
  in `MaterializedConstantValue` — these stay at generation time and don't
  take up an IR slot.
- For `LambdaValue`s: recursively convert to `lambda_in_closure` via
  `bind_staged` — so nested lambdas capture their closures correctly.

`create_closure` is used by `Context._try` to build the closure records for
the try-and-except bodies. For `_if`/`_while`/`_for` the rewriter passes
inputs explicitly as arguments, so no closure record is built there.

## 5. `hipy/interpreter.py` — test runner

Contains `check_prints(fn, expected_str, fallback=False, debug=None)` —
used by tests to compile `fn` through the C++ backend, run it, and assert
stdout == `expected_str`. The `HIPY_DEBUG` env var toggles debug.

## 6. `hipy/interpreter.py` — in-process fallback

```python
@hipy.raw
def not_constant(value, _context):
    return value.as_abstract(_context)
```

Single helper. Calling `interpreter.not_constant(x)` forces `x` to
materialize as an abstract IR value — useful in tests or library code to
defeat constant folding when the generated code path is the one being
exercised.

## 7. `hipy/test_utils.py` — test plumbing

```python
@hipy.classdef
class _named_tuple(_tuple):
    __HIPY_MUTABLE__ = False
    # NamedTupleType, __hipy_getattr__, __hipy__repr__ ...
```

A virtual named-tuple used by the test suite (e.g. `test/test_namedtuple.py`).
Inherits from `hipy.lib.builtins.tuple`. Exposes:
- `NamedTupleType(typename, field_names, element_types, cls=None)` — IR type
  is a `ir.RecordType` keyed by `_elt{i}`.
- `__hipy_getattr__` — attribute access becomes subscript by field index.
- `__hipy__repr__` — emits IR to build the `"Point(x=1, y=2)"` string.

Useful as a minimal example of **building a new virtual class** that's
concrete enough to exercise records, closures (via `_cvalue_binary_op`), and
repr plumbing.

## 8. Gotchas for maintainers

- **Decorators run at import time.** Adding a new decorator must not touch
  the source AST; `stage_and_compile` needs access to the original source
  via `inspect.getsource`. Keep decorators wrapper-only.
- **`@hipy.raw` can opt in or out of `_context`.** If your function
  signature already has `_context`, the decorator uses it directly; if not,
  a thunk strips the kwarg so you can write the function like a normal
  Python function. Choose consciously — raw methods that emit IR should
  take `_context`.
- **Adding a new library mock: declare `__HIPY_MODULE__ = "<name>"` at module
  top, then call `hipy.register(<this_module>)`.** Without this step
  `mocked_modules` won't see it and the original package will be used.
- **`create_closure` is the only supported way to build a closure record**
  that is handed to IR. Building one by hand via `closure_record(...)` won't
  perform the `MaterializedConstantValue` folding and can produce larger IR
  record types than needed.
- **`binding.check_prints` is the idiomatic test helper.** It runs the
  compiler *and* the C++ backend, so it catches integration failures the
  IR-level tests miss. Every test file under `test/` uses it.
- **`HIPY_DEBUG=0` suppresses debug assertions.** `check_prints` reads this.
  Production runs without it get a small speedup from skipping the
  `context.seen` duplicate-detection check.
