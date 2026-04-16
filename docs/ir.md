# IR — The HiPy Intermediate Representation

**File:** `hipy/ir.py`

The HiPy IR is a statically-typed, high-level intermediate representation. It is
the *output* of `cogen`-generated "program generators" and the *input* of every
back-end (C++, MLIR stub). It deliberately stays small and high-level — rich
enough to express tables, arrays, columns as first-class types, but simple
enough to lower to plain C++ or MLIR dialects.

The whole module is plain Python data classes plus a `serialize()` method on
every node that emits a JSON-friendly dict (used e.g. by `hipy/binding.py` to
talk to the standalone interpreter binary).

## 1. Type system

The paper's Table "Types" (§4) is realized as a set of small Python classes.
Every type class has:

- `__str__` — pretty printer
- `serialize()` — JSON form (`{"kind":"type","name":…, …}`)
- `mangle()` — stable string used in symbol/name mangling and dict keys
- `__eq__` / `__hash__` by mangled name
- most also have `get_generic()` — drops parameters so the same family hashes
  together (used by the optimizer/pattern rewriter)

| Class | Python construct | Notes |
|---|---|---|
| `VoidType` (`ir.void`) | returned by side-effecting ops; no value | |
| `PyObjType` (`ir.pyobj`) | CPython-owned ref-counted object | anchor for fallback |
| `BoolType` (`ir.bool`) | Python `bool` | |
| `IntegerType(width)` / `ir.i8, i16, i32, i64` | fixed-width integer | typed for back-ends |
| `IntType()` (`ir.int`) | Python arbitrary-width `int` | `serialize.name == "pyint"` |
| `FloatType(width)` / `ir.f32, f64` | `float` | |
| `StringType` (`ir.string`) | `str` / `bytes` unified | |
| `RecordType([(name,type), …])` | records / named tuples / closures | has `member_type(col)` |
| `ListType(element_type)` | `list[T]` | `element_type=None` ⇒ generic |
| `DictType(key,val)` | `dict[K,V]` | |
| `ArrayType(element_type, shape)` | `numpy.ndarray` | shape may be `None` |
| `ColumnType(element_type)` | one column of a table | |
| `TableType([(name,type), …])` | Apache-Arrow-shaped table | first-class |
| `FunctionRefType(function, closure_type)` | reference to a `Function` with a captured closure | used for callbacks (e.g. `list.sort`, `table.map`) |
| `DateType` (`ir.date`) | calendar date | used by `hipy.lib.datetime.date` |
| `IntervalType` (`ir.interval`) | a time interval / `timedelta` | result of `date - date` |
| `NullableType(type)` | wraps any type with a possible SQL-style NULL | produced by `hipy/lib/sql.py`; MLIR backend lowers to LingoDB `db.nullable` |

Singletons for the parameterless types are constructed at module load
(`void, bool, i8, i16, i32, i64, int, f32, f64, string, pyobj, date, interval`).

## 2. SSA values, blocks, modules

- `SSAValue(type, producer)` — globally numbered (`ssa_value_ctr`), hashable,
  ordered by id. Every op emits exactly one (sometimes multiple) result(s).
- `Block()` — ordered list of `ops`; has a numeric id. Blocks are nested inside
  ops for structured control flow (`IfElse.ifBody`, `IfElse.elseBody`).
- `Function(module, name, arg_types, res_type)` — SSA function; args are
  pre-constructed `SSAValue`s owned by the function, body is a `Block` that
  must terminate in a `Return`.
- `Module()` — top-level container. Holds:
  - `block` — a `Block` whose ops are `Function`s (top-level definitions)
  - `imports` — dict of module-name → CPython-side module alias; emitted as
    `import` statements in the C++ stub
  - `py_functions` — dict of name → inlined Python source, used by the fallback
    path (CPython `exec`s these at C++-runtime startup)
- `Module.merge(other)` — merges all three of the above. Used when a generator
  produces incremental pieces of a module.

No phi-nodes. Control-flow merges only happen via `IfElse` results.

## 3. Operations

Every operation derives from `Operation` and implements:
- `get_used_values()` / `get_produced_values()` — used by DCE, pattern
  rewriter, etc.
- `has_side_effects()` — pure ops can be eliminated / moved freely
- `replace_uses(old, new)` — cheap SSA rewriter support
- `get_nested_blocks()` — non-empty only for structured control flow
- `clone(block, mapping)` — deep-clone into a new block, remapping SSA values
- `serialize()` — JSON form for external consumers

Helper utilities: `replace_usage_in_list`, `replace_usage_in_dict_values`, `flatten`.

### 3.1 Core ops

| Op | Purpose |
|---|---|
| `Constant(block, v, t)` | immediate value of type `t` |
| `Undef(block, t)` | poison / placeholder (used by transactions rolling back) |
| `Return(block, values)` | sole function terminator; guarded against duplicates |
| `Yield(block, values)` | terminator for nested `IfElse` / callback blocks |
| `Free(block, v)` | explicit free, emitted by `opt/eager_free.py` |
| `IfElse(block, cond, return_types)` | sole structured branch — contains two
  nested `Block`s (`ifBody`, `elseBody`), each ending in a `Yield` whose values
  become the op's `results`. This is the only way the IR joins control flow. |

### 3.2 Record ops

- `MakeRecord(block, res_type, {name: value, …})` — construct a record.
- `RecordGet(block, res_type, record, member)` — project one field.

Records are how HiPy represents **tuples, named tuples, closures**, and generic
structured data. The closure mechanism in `internal_values.closure_record`
generates these two ops.

### 3.3 Function / call ops

- `FunctionRef(block, func, closure=None)` — produces a `function_ref(…)->t`
  value; optional `closure` SSA value is forwarded through the reference and
  passed as an extra argument by the back-end.
- `Call(block, name, args, ret_type)` — direct call by IR symbol name.
- `CallIndirect(block, fnref, args, ret_type)` — call through a `function_ref`
  (possibly carrying a closure). Back-ends are responsible for expanding the
  closure into the trailing argument slot.

### 3.4 The catch-all: `CallBuiltin`

```
result = CallBuiltin(block, name, args, ret_type, side_effects=True, attributes=None)
```

`name` is a string key like `"int.add"`, `"string.substr"`, `"list.append"`,
`"array.compute"`, `"table.map"`, `"table.filter"`, `"table.apply_row_wise"`,
`"range.iter"`. Every high-level primitive in the IR ends up as a `CallBuiltin`;
this keeps the IR tiny and lets the back-end implement whatever subset it
wants — anything unsupported can be lowered by the optimizer or caught by the
fallback path and re-emitted against `pyobj`.

`side_effects=False` is a strong hint (enables DCE / hoisting). The optional
`attributes` dict carries per-op metadata (e.g. sort order, column names).

### 3.5 CPython-interop ops

These are the anchors for HiPy's fine-grained fallback. All of them return
`pyobj` unless noted:

- `PyImport(name)` — `import name` at runtime; result is the imported module
  object.
- `PyGetAttr(on, name)` — `on.name`.
- `PySetAttr(on, name, value)` — returns `void`.
- `PythonCall(callable, args, kw_args)` — invoke a `pyobj` callable with
  positional + keyword arguments; emitted whenever HiPy has to defer to CPython.

When the C++ back-end emits a binary, it wires these through the embedded
CPython via pybind11 (see `cppbackend/builtin.h` and `hipy/binding.py`).

## 4. Serialization

Every node has a `serialize()` method that produces a JSON-friendly dict. The
module-level `serialize()` yields:

```
{"kind":"module",
 "imports":{…},
 "py_functions":{name: python_source, …},
 "functions":[{"kind":"function", …}, …]}
```

`hipy/binding.py` uses this to hand a compiled IR off to an external
interpreter. The standalone C++ path emits code directly; the MLIR stub
consumes either the Python data model or the serialized form.

## 5. Invariants and gotchas

- **Exactly one `Return` per block** — the constructor asserts this; raise
  early rather than silently drop ops.
- **No phis, no unstructured CFG** — merges happen only at `IfElse` results.
  Loop-like constructs are `CallBuiltin`s that take `function_ref` callbacks
  (`range.iter`, `list.sort`, `table.apply_row_wise`). Their bodies are regular
  `Function`s in the module, referenced via `FunctionRef`.
- **`clone()` uses an SSA mapping** — when cloning an op that references an
  outer value not yet in the mapping, the op either ignores it
  (`Constant`, `Undef`) or reads the original value (explicit `if v in mapping else v`).
  If you add a new op, decide consciously which values it captures from outside
  its own block.
- **`Return.clone` asserts False** — returns should never be cloned; they mark
  function boundaries.
- **Types are compared by `mangle()`** — when adding a new type, implement
  `mangle`, `__hash__`, `__eq__`, `serialize`, `__str__`. Optionally
  `get_generic()` so the pattern rewriter can match families.

## 6. Extending the IR

See `extending.md` for the full story. The short version:

1. **Prefer a new `CallBuiltin` name** over a new op. Back-ends can handle it
   explicitly; anything unsupported falls back via the generator context.
2. **Add a new op only if it changes structure** (new kind of control flow,
   new closure convention, new binding-level construct). If you do, implement
   all `Operation` abstracts, add a `serialize()` kind, update every back-end,
   and consider whether it needs an entry in `opt/pattern_rewriter.py`.
3. **Add a new type only if the back-end needs to distinguish it at the
   runtime level.** Higher-level distinctions belong in `VirtualType`
   subclasses in `hipy/value.py`.
