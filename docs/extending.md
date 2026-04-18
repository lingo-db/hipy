# Extending HiPy

This guide covers the three ways to grow HiPy:

1. **Add a library shim** for an existing Python package (numpy,
   pandas-style replacement) — most common.
2. **Add a new virtual type** (record / column / specialized container) —
   when your domain needs first-class treatment.
3. **Add an IR builtin and backend handler** — when a high-level op needs
   to lower to something backends can recognize & optimize.

Plus smaller extensions:
- New optimization pass.
- New backend target.
- New language construct (needs cogen work).

Read the component docs first (`compiler.md`, `context.md`, `value.md`,
`ir.md`, `intrinsics.md`, `standard-library.md`, `cpp-backend.md`) —
this file assumes you know what each piece does.

## 1. Add a library shim

**Use case:** You want `import foo` to route through HiPy's compiler
instead of the real `foo` module.

### 1.1 The minimum template

Create `hipy/lib/foo.py`:

```python
import sys
import hipy
from hipy import intrinsics

__HIPY_MODULE__ = "foo"                     # name of the real module to replace

@hipy.compiled_function
def bar(x, y):
    # ordinary Python body — will be cogen-rewritten
    return x + y

@hipy.compiled_function
def baz(s):
    # emit a custom IR builtin
    return intrinsics.call_builtin("foo.baz", str, [s], side_effects=False)

hipy.register(sys.modules[__name__])         # MUST be called or the shim is invisible
```

Checklist:
- [ ] `__HIPY_MODULE__ = "<real_module>"` at the top.
- [ ] `hipy.register(sys.modules[__name__])` at the bottom.
- [ ] Test: `import hipy.lib.foo` somewhere in the test file before
      `check_prints`.

### 1.2 Picking `@compiled_function` vs `@raw`

Default: `@hipy.compiled_function`. Write ordinary Python; the cogen
rewrites every expression into `_context.*` calls. Your body can freely
use other `@compiled_function` helpers, arithmetic, lists, dicts, etc.

Drop to `@hipy.raw` only when you need one of:
- Pattern-match on `ValueHolder` contents (e.g. only handle
  `_const_int`, fallback otherwise).
- Access `_context` for things `intrinsics.*` doesn't expose.
- Constant-fold entire calls (compute the result in Python).

### 1.3 Emitting a direct IR builtin

```python
@hipy.compiled_function
def upper(self):
    return intrinsics.call_builtin("string.upper", str, [self], side_effects=False)
```

- First arg: the builtin name (a string). Convention: `namespace.op` —
  `scalar.<type>.<op>`, `list.<op>`, `dict.<op>`, `table.<op>`, etc.
- Second arg: result type. Either a Python class (`str`, `int`), a
  `TypeValue(...)`, or `None`/void.
- Third arg: a list of `ValueHolder`s.
- `side_effects=False` lets DCE drop the op when its result is unused.
  Use `True` for I/O, RNG, prints.
- The backend must implement this builtin — see §4.

### 1.4 Guard clauses

```python
@hipy.compiled_function
def my_op(x, y):
    intrinsics.only_implemented_if(
        intrinsics.isa(x, MyType),
        intrinsics.isa(y, MyType)
    )
    return intrinsics.call_builtin("mytype.op", intrinsics.typeof(x), [x, y])
```

`only_implemented_if(*conds)` raises `NotImplementedError` if any
condition fails, which triggers the context's dispatch machinery to fall
back (to another overload or `pyobj`). `isa(v, cls)` is a constant-folded
type check.

### 1.5 Higher-order operations — `intrinsics.bind`

When a builtin needs a callback function (sort, filter, apply,
row-wise map):

```python
@hipy.compiled_function
def apply_rows(table, fn):
    return intrinsics.call_builtin(
        "table.apply_row_wise_scalar",
        some_col_type,
        [table, intrinsics.bind(fn, [row_type])],   # materializes fn to an IR FunctionRef
    )
```

`bind(fn, [arg_types])` turns a `LambdaValue` / `HLCFunctionValue` /
callable pyobj into a `MaterializedFunction` wrapping an `ir.FunctionRef`.
The wrapping records the inferred result type (accessible as
`.res_type`), which the caller can use to fill the builtin's return type.

Cache a `bind` result when you call it in a loop — it allocates a fresh
IR function every invocation.

### 1.6 Registering pickled objects

For library functions that return trained models / constant data:

```python
class MyModel(static_object["param1", "param2"]):
    @staticmethod
    def __from_constant__(py_obj, context):
        # py_obj is the fitted Python model (passed through pickle.loads at compile time)
        return cls(context.constant(py_obj.param1), ...)

    def __topython__(self):
        # reconstruct a real MyModel object from the HiPy fields (for fallback)
        ...
```

See `hipy/lib/sklearn/*` for examples. The `pickle.loads` shim calls
`__from_constant__` at generation time, so the fitted model's weights
are lifted into IR constants with no runtime pickle dependency.

## 2. Add a new virtual type

**Use case:** Introduce a new first-class container or specialized
representation — e.g., a bitmap, a sparse matrix, a custom datetime type.

### 2.1 The boilerplate

```python
import hipy
import hipy.ir as ir
from hipy.value import Value, Type, ValueHolder

@hipy.classdef                              # mandatory on every Value subclass
class my_scalar(Value):
    __HIPY_MUTABLE__ = False                # set True only if in-place mutation is observable
    __HIPY_NESTED_OBJECTS__ = False         # set True only if it contains other virtual values
    # __HIPY_MATERIALIZED__ defaults to True — set False for _const_* variants

    class MyScalarType(Type):
        def ir_type(self):    return ir.int                  # or your own ir type
        def construct(self, value, context):
            return context.wrap(my_scalar(value))
        def __eq__(self, other):
            return isinstance(other, my_scalar.MyScalarType)
        def __hash__(self):
            return hash(type(self))

    def __init__(self, value: ir.SSAValue):
        super().__init__(value)

    @staticmethod
    def __hipy_create_type__():
        return my_scalar.MyScalarType()

    def __hipy_get_type__(self):
        return my_scalar.MyScalarType()

    def __topython__(self, _context):
        # emit IR to produce a pyobj equivalent
        return intrinsics.call_builtin("myscalar.to_python", object, [self])
```

Checklist:
- [ ] `@hipy.classdef` — wires up `_super_<method>` attributes and class
      registration.
- [ ] `__HIPY_MUTABLE__`, `__HIPY_NESTED_OBJECTS__` set correctly.
- [ ] Nested `Type` subclass with `ir_type`, `construct`, `__eq__`,
      `__hash__`.
- [ ] `__hipy_create_type__` (static) and `__hipy_get_type__`
      (instance).
- [ ] `__topython__` for fallback.
- [ ] `__merge__` if mutable or if control-flow merges will happen
      (otherwise falls back silently to `pyobj`).

### 2.2 Adding a `_const_*` variant for compile-time folding

```python
@hipy.classdef
class _const_my_scalar(my_scalar, CValue):   # inherit CValue for auto-folding dunders
    __HIPY_MUTABLE__ = False
    __HIPY_MATERIALIZED__ = False

    def __init__(self, cval):
        self.cval = cval

    def __abstract__(self, context):
        ssa = ir.Constant(context.block, self.cval, ir.int).result
        return my_scalar(ssa)

    def __topython__(self, _context):
        # can fold: emit a constant pyobj directly
        ...
```

Inheriting `CValue` makes every `_cvalue_binary_op` dunder
compile-time-fold when both operands have `cval`; otherwise delegates to
`_super_<op>` which calls the non-folding parent class (emitting an IR
builtin). This is how `_const_int(5) + _const_int(3)` becomes `8` at
generation time.

### 2.3 Mutable containers — `__merge__` and `__topython__`

For anything mutable that might flow through `if`/`while`/`for`,
implement `__merge__`:

```python
@staticmethod
def __merge__(self, other, self_fn, other_fn, context):
    # called when control-flow branches produce two different my_container Values
    # self_fn / other_fn inject IR ops into the branch's local block
    ...
    def create_merged(ssa_at_merge_point):
        return my_container(ssa_at_merge_point, ...)
    return (adjusted_self, adjusted_other, create_merged)
```

If you don't implement it, `ctx.merge` falls back to converting both
sides to `pyobj` — which works but kills optimization.

### 2.4 Registering in a shim

If the new type is the principal export of a library, put it in the
shim module and expose it as the `__HIPY_MODULE__`'s class. See
`hipy/lib/numpy/__init__.py`'s `ndarray`, or `hipy/lib/pandas/__init__.py`'s
`DataFrame`.

## 3. Add an IR builtin and backend handler

Every `intrinsics.call_builtin("fn_name", ...)` introduces an IR
`CallBuiltin` op. Backends must handle every name the generator could
emit.

### 3.1 Pick a name

Convention: `namespace.op`, where `namespace` groups related ops.
Existing namespaces:

- `scalar.<int|float|bool|string>` — primitives
- `list`, `dict`, `set` — general containers
- `table`, `column`, `array` — tabular / numeric
- `python.operator`, `python.create_*` — pyobj operations
- `dbg` — printing / debugging

### 3.2 Add a C++ backend handler

In `hipy/cppbackend/__init__.py`, `CPPBackend.generate_builtin`:

```python
case "foo.bar", args:
    # emit a C++ expression as a string, using self.mapping[arg.result] etc.
    self.emit(f"auto val_{self.next_id()} = builtin::foo::bar({arg_names});")
    self.mapping[op.result] = f"val_{...}"
```

Then add a matching implementation in the runtime headers (`cppbackend/*.h`).
Keep a namespace consistent: `scalar.string.*` maps into `builtin::string::*`.

Always match on `(name, arg_types)` when the handler cares about
specialization — the MLIR backend does this too.

### 3.3 (Optional) add an MLIR backend handler

In `hipy/mlirbackend/__init__.py`, add a case to `to_mlir_stmt`'s
`CallBuiltin` match. Note the MLIR backend covers only a small subset;
don't be alarmed if you skip this.

### 3.4 (Optional) add an optimization pattern

If your builtin benefits from fusion or constant folding, write a
`RewritePattern` under `hipy/opt/` and register it with a
`PatternRewriter`. See `optimizations.md` for the API.

## 4. Add an optimization pass

In `hipy/opt/`:

```python
from hipy.opt.pattern_rewriter import PatternRewriter, RewritePattern
import hipy.ir as ir

class MyPattern(RewritePattern):
    def rewrite(self, op, rewriter):
        match op:
            case ir.CallBuiltin(name="foo.bar", args=[x]):
                # transform
                rewriter.replace_with(op, ir.CallBuiltin, name="foo.bar_fast",
                                      args=[x], ret_type=op.result.type)
                return True
        return False

def run(module: ir.Module):
    PatternRewriter([MyPattern()], module).rewrite()
    return module
```

Always use `rewriter.replace_with` / `replace_with_value` / `create` —
they maintain the `rewriter.uses` index. See `optimizations.md §1` for
the driver details.

Wire it in by invoking `my_pass.run(module)` from wherever the
optimization pipeline is assembled (search for `canonicalize(` or
`inline.run(` in the repo for the current call sites).

## 5. Add a new backend target

Mirrors `hipy/mlirbackend/__init__.py`'s shape:

1. `hipy/mybackend/__init__.py` with:
   - `to_my_type(ir_type)` — IR type → target type.
   - `to_my_stmt(stmt, mapping)` — IR op → target op emission.
   - `to_my_func(fn)` — IR function → target function.
   - `to_my_module(module)` — walks `module.funcs()`.
   - `compile(module)` — entry point.

2. Handle every IR op class you expect to see after optimization:
   `Constant`, `CallBuiltin`, `Call`, `CallIndirect`, `IfElse`,
   `Yield`, `Return`, `FunctionRef`, `MakeRecord`, `RecordGet`,
   `PyImport`, `PyGetAttr`, `PySetAttr`, `PythonCall`, `Free`.

3. For `CallBuiltin`, build a match table over `(name, arg_types)`. The
   MLIR backend's table is a good minimum set; the C++ backend's is a
   complete one.

4. If your target has structured control flow ops (loops, joins),
   consider extending the backend with backend-specific IR op classes
   (see the historical `cppir.py` scaffolding) plus a lowering pass
   that rewrites generic ops into them before codegen.

5. Optional: a runtime header library (analog to `cppbackend/builtin*.h`)
   providing the target-language implementations of every builtin.

## 6. Add a new language construct

Rare — most Python gets handled by cogen. But if you need (say) `match`
/ `with` / `async` / etc.:

1. Add a case in `hipy/compiler.py::stage_stmt` (for statements) or
   `stage_expr` (for expressions) that emits the appropriate
   `_context.*` call with a fresh `_action_id`.
2. Add the handler method on `Context` in `hipy/context.py`. Wrap the
   body in `with self.handle_action(_action_id): …` — this is mandatory
   for escape analysis to get stable t_locations.
3. If the construct speculates (evaluates code that might not make it
   into the final IR), use `ctx.transaction()`.
4. If it introduces control flow, use `ctx.merge(...)` to type-converge
   branch results.

Study `context.py::_if`, `_while`, `_for`, `_try` for patterns. Any new
construct that allocates `ValueHolder`s must push a location label
(`"if"`, `"body"`, etc.) so t_locations stay stable across reruns.

## 7. Testing your extension

Always end-to-end: write a `test/test_<feature>.py` that:

- Imports `hipy.lib.<module>` if you added a shim.
- Defines a `@hipy.compiled_function` using the new feature.
- Uses `not_constant(...)` on inputs to defeat constant folding.
- Calls `check_prints(fn, expected)`.

For optimization passes, compile a function then run your pass on the
resulting `module`, then assert something about `str(module)`. See
`test/opt/test_pattern_rewriter.py` for the pattern.

## 8. Invariants to preserve

- **`ValueHolder` is the only stable identity.** Never cache a `Value`
  directly — it can be swapped (mutable holder swap, `as_abstract`).
- **Every public `Context` method takes `_action_id=None` and wraps its
  body in `handle_action(_action_id)`.** Required for escape analysis.
- **Every `@hipy.compiled_function` must be callable as plain Python.**
  The cogen uses `inspect.getsource`; keep decorators wrapper-only, no
  AST mutation.
- **Every shim module must call `hipy.register(sys.modules[__name__])`**
  at the bottom. Every virtual class must use `@hipy.classdef`.
- **Every new IR builtin must have a handler in every backend you
  support.** Missing names fail at codegen time.
- **`__HIPY_MATERIALIZED__ = False`** on `_const_*` / `_concrete_*`
  variants — otherwise they'll be eagerly materialized and you lose
  constant folding.
- **`eager_free` runs after the last use.** Anything you emit *after*
  `eager_free` runs must not reference freed SSAs. Run new passes before
  `eager_free`.
- **Fix-point iteration is capped at 3.** In `_while`/`_for`, if your new
  type's `__merge__` doesn't converge in 3 iterations, you'll hit a "too
  many iterations" error. Ensure the merge strictly reduces differences.

## 9. Common pitfalls

| Symptom | Likely cause |
|---|---|
| Shim silently ignored; real library used | Missing `hipy.register(sys.modules[__name__])` |
| `AttributeError: 'NoneType' has no attribute 'whatever'` from `constify` | Your intrinsic needs a constant arg but got `None` back — match on `case None` and raise a proper error |
| `NotImplementedError: early return in only one branch` | `return` inside one branch of an `if` with the other branch still flowing — lift the return out or restructure |
| Generated C++ fails with `no handler for "foo.bar"` | You added the builtin but didn't register it in `cppbackend/__init__.py::generate_builtin` |
| Tests pass but hit a runtime assert in `datastructures.h` | Type mismatch between what your builtin claims (return type) and what the backend emits — double-check `ret_type` on `CallBuiltin` |
| `assert False` crash in `mlirbackend` | The MLIR backend doesn't support the type / op you emitted — fine for tests running through the C++ backend, but guard explicitly if MLIR is a target |
| `counter > 2` raise in `_while`/`_for` | Fix-point type inference didn't converge; likely a `__merge__` that doesn't stabilize |
| `ValueHolder.t_location` mismatch across runs | Forgot `with self.handle_action(...)` in a new Context method, or forgot to push a location label (`"body"`, `"if"`, etc.) around an internal transaction |
| Pyobj fallback you didn't expect | Some IR builtin wasn't recognized and escape analysis converted the value; grep `events` in `context.py` and add a proper handler |
| Builtin that should be pure is not DCE'd | `side_effects=True` by default — pass `side_effects=False` on `intrinsics.call_builtin` for pure ops |

## 10. Reading path for contributors

If you're new to the codebase, read in this order:

1. `hipy_paper.md` (in repo root) — the OOPSLA'24 summary; the *why*.
2. `docs/ir.md` — the data model you'll be producing and consuming.
3. `docs/compiler.md` — how Python becomes a program generator.
4. `docs/context.md` — how that generator actually runs.
5. `docs/value.md` — the virtual-object world.
6. `docs/intrinsics.md` — the API you'll use most.
7. `docs/runtime-glue.md` — decorators + function wrapping + closures.
8. `docs/standard-library.md` — concrete examples of every pattern.
9. `docs/optimizations.md` — how IR is transformed before codegen.
10. `docs/cpp-backend.md` — where IR becomes a runnable program.
11. `docs/mlir-backend.md` — reference for alternative targets.
12. `docs/tests.md` — how to exercise anything you add.
