# Intrinsics — helper API for library authors

**File:** `hipy/intrinsics.py`

`intrinsics` is the ergonomic face of `Context` for library authors writing
library shims in `hipy/lib/`. Instead of hand-rolling `@pgen` methods that
call `_context.call_builtin` / construct types / etc., you write plain
`@hipy.compiled_function` code that *imports* `hipy.intrinsics` and calls
these helpers. The helpers themselves are `@hipy.raw` functions (so they
receive `_context` and can emit IR directly) but they present a
value-oriented API to their callers.

Rule of thumb: **prefer `@cogen` + `intrinsics` over `@pgen`.** Raw `@pgen`
is still needed for methods that manipulate virtual-object internals
(unmaterialized state, transitions, `__topython__`, `__merge__`), but the
vast majority of stdlib shims are simple constant-fold-or-emit-builtin
helpers that these intrinsics cover.

## 1. `call_builtin(fn_name, return_type, args, side_effects=True, attributes=None)`

Emits an `ir.CallBuiltin` — the catch-all high-level op in the IR. Accepts:

- `fn_name` — a `CValue` with a `str` cval (e.g. `"string.split"`).
- `return_type` — a class (`str`, `int`) or a `TypeValue` from `typeof(...)`.
  `None`/`VoidValue` means void result.
- `args` — a `_concrete_list` of `ValueHolder`s; materialized to SSA in
  order.
- `side_effects` — if `False`, DCE / canonicalization may drop / reorder
  the op.
- `attributes` — optional extra metadata (constant-folded via `constify`).
  Useful for e.g. column names on table ops.

This is how most stdlib IR production looks from library code:

```python
@hipy.compiled_function
def upper(self):
    return intrinsics.call_builtin("string.upper", str, [self], side_effects=False)
```

## 2. Type introspection and construction

| Function | Purpose |
|---|---|
| `typeof(value)` | Returns a `TypeValue` wrapping the value's current `__hipy_get_type__`. Use when a builtin needs to echo a runtime-derived type. |
| `create_type(cls, *args)` | Build a `TypeValue` by calling `cls.__hipy_create_type__(*constified_args)`. Needed to construct parameterized types from a library signature (`list[int]`, `array[shape x f64]`). |
| `isa(value, cls)` | Constant-folded `isinstance` equivalent. Accepts `HLCClassValue` or `TypeValue`; compares via `__hipy_get_type__` or direct `isinstance` on the underlying `Value`. |
| `isoftype(value, type)` | Like `isa` but strict on `Type` equality. |
| `try_narrow(value, type)` | If `isa(value, type)` already, return value; else try `value.__narrow_type__(value, type)` and silently keep the original on `NotImplementedError`. Used by library code that wants to refine the type when possible but not force. |

`constify(arg)` is the internal helper that turns a `ValueHolder` holding a
constant/list/tuple/dict of constants into a plain Python value (or
`None` if the value isn't constant). Used by `call_builtin`, `create_type`,
`bind`, `create_record` when they need a compile-time Python value.

## 3. Fallback control

| Function | Purpose |
|---|---|
| `not_implemented()` | Raise `NotImplementedError`. Use inside a `@pgen` or `@raw` method to signal "this specialization doesn't apply" — the outer dispatch will try another overload or fall back. |
| `only_implemented_if(*conds)` | Each arg must be a `CValue(bool)`. If any is False, raises `NotImplementedError`. Lets you write guard clauses like `intrinsics.only_implemented_if(isa(arg, int))` at the top of a specialization. |
| `error(msg)` | Constant-folded `raise RuntimeError(msg)` — fires at generation time, not runtime. Use for compile-time checks. |

## 4. IR plumbing

| Function | Purpose |
|---|---|
| `to_python(value)` | Force `value` to its `pyobj` form. Same as `context.to_python` but wrapped as a library-friendly call. |
| `annotate_object_abstract_path(value, abstract_path)` | Tag a `pyobj` value (an `hipy.lib.builtins.object`) with a constant string path. Used by MLIR lowering to stash the original dotted Python path of a fallback-imported module so the backend can re-resolve it without scanning imports. |
| `import_pymodule(name)` | Emits `ir.PyImport` and wraps the result as a `pyobj`. Useful for library code that needs to import a CPython module without having a corresponding HiPy shim. |
| `get_attr(obj, name)` | Dynamic-attribute access dispatched at generation time — `name` must be a constant string `CValue`. Resolves via the usual `_context.get_attr` path (not a runtime pyobj attr). Used by virtual types that forward attribute access to an inner value (e.g. `_MaybeNone.__hipy_getattr__`). |
| `track_nested(nested, container)` | Explicit `NestedValueAlias` event. Use when you build a composite value whose components should share the container's escape fate. |
| `reinterpret(value, return_type)` | **Hard cast** — take the value's IR SSA as-is and re-wrap it as `return_type`. Requires the IR types to be compatible; use sparingly (ABI casts, cross-lib interop). |
| `undef(type)` | Emit `ir.Undef` of the given type and wrap. Useful for placeholder results in conditionals where one branch doesn't produce a meaningful value. |
| `as_abstract(value)` | Force materialization: calls `value.as_abstract(_context)`. |
| `as_constant(value)` | Wrap the (already const) value in a `MaterializedConstantValue` — pins it into a closure record as a generation-time constant. |
| `create_record(names, values)` | Build a `RawValue` holding an `ir.MakeRecord`. Used when a library needs a raw IR record value (e.g. for `CallBuiltin` args). |
| `call_indirect(fn, args, res_type)` | Emit `ir.CallIndirect` against a `function_ref`-typed value. Used when dispatching through a first-class function. |

## 5. `bind(fn, arg_types)` — materialize callables to IR functions

This is the single most important intrinsic for higher-order builtins
(sort, map, filter, apply, …). It takes a `LambdaValue` /
`HLCFunctionValue` / Python callable (`lib.builtins.object`) and produces
a `MaterializedFunction` wrapping an `ir.FunctionRef` that the IR can
pass into a `CallBuiltin` as a callback.

Implementation sketch:

- For `LambdaValue`: uses `bind_staged_fn` from the 3-variant expansion. A
  nested `bind_fn(closure_dict, fn)` builds a closure record (via
  `binding.create_closure`), creates a new `ir.Function` with signature
  `[arg_types..., closure_type?]`, threads args through the closure into
  the body, emits the call, and wraps the result in `MaterializedFunction`
  + the inferred `res_type`.
- For `HLCFunctionValue`: ordinary IR function, no closure needed.
- For `lib.builtins.object` (an already-pyobj callable): emits a wrapper
  IR function that takes a trailing `pyobj` arg and calls through it.

The result, `MaterializedFunction`, stores `res_type` as a `TypeValue` so
the caller can inspect it (see `try_or_default` below).

`bound_ctr` is a module-global counter used to name each synthesized
function (`bound_fn0`, `bound_fn1`, …). Every `bind()` produces a fresh
function, so repeated `bind(sameLambda, …)` inside a hot loop will emit
multiple IR functions unless the caller caches.

## 6. `try_or_default(fn, default)`

Example of a useful composition built on top of the other intrinsics —
binds `fn` with no args, checks that its return type matches `typeof(default)`,
and emits a `CallBuiltin("try_or_default", T, [bound, default])` that the
back-end runs as an IR-level try/catch.

## 7. How to write a library shim using intrinsics

The common pattern for a pandas/numpy/sklearn-style wrapper:

```python
from hipy import intrinsics

@hipy.compiled_function
def my_op(x, y):
    intrinsics.only_implemented_if(intrinsics.isa(x, MyType),
                                   intrinsics.isa(y, MyType))
    return intrinsics.call_builtin("mytype.op",
                                   intrinsics.typeof(x),
                                   [x, y],
                                   side_effects=False)
```

This keeps the source plain Python (so CPython can still import + exercise
it at test time), but the `_context`-wrapped path emits IR via the helpers.

## 8. Gotchas

- **`call_builtin` return type must be `None`, `HLCClassValue`, or
  `TypeValue`.** Passing a raw `Type` instance doesn't work — wrap it in
  `TypeValue(t)` first or use `intrinsics.typeof`.
- **`constify` returns `None` when it can't fold.** If your intrinsic
  requires a constant argument, match on `case None: raise …` after
  calling it, otherwise you'll silently get a null arg.
- **`bind` allocates a new IR function every call.** Cache or memoize if
  you call it in a hot loop. The paper mentions this — `skllearn/pipeline`
  uses `bind` heavily and is careful about it.
- **`reinterpret` bypasses `__merge__` / `__topython__`.** Don't use it
  across unrelated types; it's for changing the Python-level wrapping of
  an IR value whose runtime representation hasn't changed.
- **`error(msg)` fires at generation time.** That's the point — use it as
  a "dependent typechecker". Don't confuse with `RuntimeError` in staged
  code, which fires in the host CPython if you call the function directly.
