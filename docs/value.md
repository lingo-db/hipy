# Virtual values and types — the generator-time object model

**File:** `hipy/value.py`

Every value that flows through a `Context` during IR generation is a
`ValueHolder` wrapping a `Value` subclass. Every `Value` subclass exposes a
`Type` (via `__hipy_get_type__`) which knows how to materialize it into a
single `ir.SSAValue` at some point later. Together, `Value` + `Type` make
up the **virtual object** abstraction of the paper (§4.2).

This file defines the abstract bases plus the handful of universally-used
concrete values (closures, modules, class handles, function/method handles,
lambdas). The large catalog of *concrete Python builtins* (int, list, dict,
etc.) lives in `hipy/lib/builtins.py` and `hipy/internal_values.py` — see
`internal-values.md` / `standard-library.md`.

## 1. `Type` — the abstract base

```python
class Type(ABC):
    @abstractmethod
    def ir_type(self): ...            # -> an ir.* type
    @abstractmethod
    def construct(self, value, context): ...   # ir.SSAValue → Value
    @abstractmethod
    def __eq__(self, other): ...
    def get_cls(self): return None    # optional: the virtual class bound to this type
```

- `ir_type()` is the lowered IR type (e.g. `ir.int`, `ir.string`,
  `ir.RecordType([...])`).
- `construct(ir_value, context)` builds the corresponding virtual `Value`
  from an already-produced `ir.SSAValue`. Used when unpacking records,
  constructing loop iteration variables, consuming `CallBuiltin` results,
  building arguments at function entry, etc.
- `__eq__` is what `context.merge` / fix-point iteration uses to tell
  whether two values have the same "shape". Must be symmetric.
- `get_cls()` optionally returns the virtual class (useful for
  `TypeValue.__call__` — see below).

### Concrete `Type` subclasses

| Class | Use |
|---|---|
| `AnyType` | Catch-all: constructs `lib.builtins.object` over `pyobj`. |
| `SimpleType(cls, ir_type)` | One-shot binding of a virtual class to an IR type. Used everywhere: `SimpleType(int, ir.int)`, `SimpleType(str, ir.string)`, etc. |
| `_NotRelevantType` | Returned by lambdas — they never flow through merges or args. |
| `ValueHolder.AbstractViaPython` | Sentinel returned by `__abstract__` to say "I can only be materialized by going through pyobj." |

Most concrete types are **nested inside** their `Value` subclass
(e.g. `VoidValue.VoidType`, `HLCFunctionValue.FunctionValueType`) — this
keeps the type definition next to the value class it mates with.

## 2. `Value` — the abstract virtual object

```python
class Value(ABC):
    __HIPY_MUTABLE__ = True
    __HIPY_NESTED_OBJECTS__ = True
    __HIPY_MATERIALIZED__ = True
    def __init__(self, value: ir.SSAValue): ...
    def __ir_value__(self): return self.__value__
    def __abstract__(self, context): return self
    @abstractmethod
    def __topython__(self): ...
    @staticmethod
    def __merge__(self, other, self_fn, other_fn, context):  # returns (self', other', create_merged)
    @staticmethod
    @abstractmethod
    def __hipy_create_type__(*args) -> Type: ...  # class-level: build Type from Type-args
    @abstractmethod
    def __hipy_get_type__(self) -> Type: ...      # instance-level: my current Type
```

Three important class-level booleans:

| Flag | Meaning |
|---|---|
| `__HIPY_MUTABLE__` | Whether in-place mutation is observable — drives the `VirtualObjectHolder` swap behaviour, and whether `to_python` mutates the holder or returns a fresh one. |
| `__HIPY_NESTED_OBJECTS__` | True if this type can contain *other* virtual values whose escape is not trivially observable (lists, dicts, records of records). Drives escape-analysis conservatism in `internal_values.create_closure` and friends. |
| `__HIPY_MATERIALIZED__` | False means the value currently holds Python-side state (e.g. `_const_int`, `_concrete_list`) and must be materialized into IR before anything generic can consume it. Controls `MaterializedConstantValue` wrapping when values leak out of closures. |

Conventions:

- `__value__` is the lone `ir.SSAValue` — stored as `__value__` (double
  underscore) to stay out of the way of subclass attributes. `None` means
  "not yet materialized".
- `__abstract__(context)` — "turn me into a single-SSA-value version of
  myself and materialize it in the current block." The default is a no-op
  (already materialized). Unmaterialized subclasses override this.
- `__topython__` — emit IR to produce a `pyobj` equivalent. Every virtual
  mutable type must implement this (see "Gotchas" in the paper summary).
- `__merge__` — for control-flow merging. Signature documented inline:
  receives two values and two *branch-local injection* callables
  (`self_fn`, `other_fn`); returns the adjusted values plus a
  `create_merged` that takes the merge-point IR SSA and returns the merged
  `Value`. Default raises `NotImplementedError` → `context.merge` falls
  back to pyobj.

## 3. `ValueHolder` — the stable identity

```python
class ValueHolder:
    context: Context
    t_location: tuple[...]   # stable identifier (see context.md)
    value: Value
```

Every time a generator code path produces a virtual object, it is wrapped in
a `ValueHolder` via `context.wrap(value)`. The holder:

1. Records its `t_location` = `tuple(context.location)` at creation time.
2. In debug mode, asserts no two holders share a `t_location`.
3. If its `t_location` appears in `context.decisions` (from a prior
   generation run's escape-analysis verdict), the value is **immediately
   converted to pyobj** via `context.to_python(self)` so mutation-heavy code
   sees the pyobj form consistently.
4. Intercepts `self.value = X` via `__setattr__` to call
   `context.log_set_value(self)` — this is the hook that transactions use to
   build their undo log.

`holder.as_abstract(context)`:
- If the current `value` is mutable, *mutates the holder in place* to the
  abstract form — the paper's `VirtualObjectHolder` swap (§4.5).
- If immutable, wraps the abstract form in a fresh holder.

`holder.get_ir_value(context)`:
- If already materialized, returns the `ir.SSAValue`.
- Otherwise calls `as_abstract` first.

The `VirtualObjectHolder` is *this class*. The reason for the indirection:
mutable values can transition between Unmaterialized → Materialized →
pyobject forms while still being referenced from elsewhere. Because every
access goes through the holder (the rewriter made sure of this), swapping
`holder.value` propagates transparently.

## 4. Universal virtual values defined here

| Class | Purpose |
|---|---|
| `VoidValue` | Python `None`. `ir.void` IR type. Constant folding just compares `VoidValue` instances. |
| `RawValue` | Bare `SSAValue` passthrough. Used when the IR needs to feed raw refs (`FunctionRef`, record pointers) through a `ValueHolder` to mix them into a `CallBuiltin` arg list. Never converted to Python; never merged. |
| `MaterializedConstantValue` | Wraps a constant-at-generation-time value that has been materialized (i.e. its container forced materialization). IR type is `void`; the actual value is preserved as `_value`. Used by `internal_values.create_closure.keep_const`. |
| `ConstIterValue` | A generation-time iterator over a Python list of `ValueHolder`s. Used to "unroll" constant iteration — `for x in (a, b, c):` becomes three inlined body calls rather than an IR loop. |
| `PythonModule` | Wrapper for a Python module (the host one, or a HiPy-mocked replacement). `__hipy_getattr__` falls back to CPython `get_attr` for unknown attrs. The `hipy.mocked_modules` registry (see `hipy/__init__.py`) swaps in HiPy's replacements for `numpy`, `pandas`, etc. |
| `RawModule` | Like `PythonModule` but skips the mocked-modules rewrite. Used via `hipy.value.raw_module(m)` when code *needs* the real module. |
| `HLCClassValue` | Reference to a `@hipy.classdef` class. Calling it goes through `cls.__create__`; attribute access is direct (via `__getattr__`). `__topython__` finds the original class in the host module via `__HIPY_MODULE__` metadata. |
| `HLCFunctionValue` | Reference to an `HLCFunction`. `__topython__` registers the source into `module.py_functions` (if in base module) and imports the host-module version. Strips hipy decorators from the source before registering. |
| `HLCMethodValue` | Reference to a bound method; `__topython__` reads the method off the pyobj form of `self`. |
| `HLCGeneratorFunctionValue` | Reference to a `@hipy.raw_generator` function. Has a nested `_iterator` class (attached as `str._iterator` in the current code — see Gotchas) that implements `__itertype__` / `__iterate__` so the generator's output can be iterated over. |
| `GeneratorExpressionValue` | Compile-time representation of a Python generator expression. Holds (`iter_fn`, `iterable`, `type_infer_fn`, `packed_values`). `__iterate__(loopfn, x, iter_vals)` delegates to `iter_fn(packed_values, iterable, loopfn, x, iter_vals)`; `packed_values` is a tuple of the free variables read by the `elt` expression so the generator body can reconstruct its closure. `__itertype__` runs `type_infer_fn` inside an aborted `context.transaction()` to infer the element type without emitting IR. Produced by the rewriter for every `(... for ... in ...)` expression — see `compiler.md §3 GeneratorExp`. |
| `LambdaValue` | The 3-variant lambda from the rewriter. Holds `staged` (callable-in-generator), `bind_python` (callback producing pyobj lambda from `(closure_dict, src, name)`), `bind_staged` (producing an IR `Function` from `(closure_dict, staged_ref)`). `__topython__` emits `_hipy_bind` into `module.py_functions`. |
| `TypeValue` | Holds a `Type` at generation time. `__eq__` supports comparing with `HLCClassValue` via `cls.__hipy_create_type__()`. `__call__` dispatches to `cls.__create__` if the bound class has one — this is how `int(x)`, `str(x)`, etc. work when they come through as types. |
| `MaterializedFunction` | Materialized `function_ref` with known result type — used by `intrinsics.bind`. |
| `CValue` | Marker/mixin for "I carry a Python-side cval (constant)". Implements all binary/compare/contains/len/neg via `_cvalue_binary_op` that folds when both operands have `cval`s; falls through to `_super_<dunder>` otherwise. `_const_int`, `_const_str`, etc. inherit from this. |

## 5. `static_object[member_names]`

A factory that returns a virtual class whose instances are records with a
fixed set of named fields. Used to build named-tuple-like types on the fly
(see `hipy/internal_values._named_tuple`). Call as
`static_object["field1", "field2"]`; the returned class takes a *constructor
callback* + the fields and provides `__getattr__` access plus a Record-backed
materialization in `static_object_value_type`.

## 6. `CValue` — constant folding hook

Any class that mixes in `CValue` by inheritance gets:
- per-arith/compare dunder that folds when both sides have a `cval`,
- `__len__`, `__neg__` folding,
- `_super_<dunder>` fallthrough (set up by `@hipy.classdef`) to the
  non-folded implementation on the parent class.

That's why `_const_int + _const_int` evaluates at generation time; but
`_const_int + int` transparently delegates to `int`'s IR-emitting path.

## 7. Invariants and extension points

- **Always return a `ValueHolder` from any `@hipy.raw` method that is meant
  to be usable.** `@hipy.classdef` wraps this — inside raw methods, `self`
  is already a `ValueHolder` (NOT the raw value). Unpack with
  `match self: case ValueHolder(value=<YourClass>(…)): …`.
- **Every `Value` subclass must define `__hipy_get_type__` AND
  `__hipy_create_type__`.** `__hipy_get_type__` reads the instance; the
  static `__hipy_create_type__` is called when the class is referenced as a
  type (e.g. as an entry-function arg annotation).
- **If your type is mutable, implement `__topython__` + `__merge__`.**
  Missing `__merge__` silently falls back to pyobj on type divergence.
  Missing `__topython__` asserts at the first fallback attempt.
- **`__HIPY_MUTABLE__ = False` on any `Value` subclass that shouldn't
  participate in the holder-swap dance.** All `_const_*` values should set
  this. Mutable containers (lists, dicts, numpy arrays, pandas Series)
  should keep the default `True`.
- **The `str._iterator` class nested inside `HLCGeneratorFunctionValue` is
  a hack** — the inner code comment notes so. When you need to support
  iteration over a generator function, that class is where the merge
  between the callback type collection and the real iteration happens.
- **`remove_decorators` in `HLCFunctionValue.__topython__`** evals and
  imports every decorator AST node to decide whether it's a hipy decorator.
  If you introduce a new hipy decorator outside the `hipy.*` package,
  update the heuristic (it checks `inspect.getmodule(result).__name__.startswith("hipy")`).
- **`raw_module`** is the backdoor when you deliberately want the real
  (non-mocked) Python module, e.g. for bootstrapping pickling.
