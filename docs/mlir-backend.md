# MLIR backend — `hipy/mlirbackend/`

**File:** `hipy/mlirbackend/__init__.py` (~640 LOC)

A **prototype** backend that lowers `hipy.ir` to MLIR via the external
[LingoDB](https://github.com/lingo-db/lingo-db) project's MLIR Python
bindings (`lingodbbridge.mlir`). Compared to the snapshot from the paper,
this file has grown well past a stub — it now covers most scalar and
container IR types and enough builtins to compile real programs. It does
**not** currently produce an executable on its own — `compile(module)`
returns the MLIR module object; downstream execution is the caller's
responsibility (LingoDB's own passes / JIT).

Most production HiPy programs still use the C++ backend
(`cpp-backend.md`). Think of the MLIR backend as the path toward
database-aware codegen — the relational ops (`relalg`, `subop`) are where
this backend is headed.

## 1. Dependencies

All MLIR access goes through `lingodbbridge`, which bundles MLIR Python
bindings + custom LingoDB dialects:

- `mlir.dialects.func` — function / call / return
- `mlir.dialects.arith` — scalar arithmetic (constants, add/sub/mul/div,
  comparisons — both signed-integer and floating-point, bool via And/Or/XOr)
- `mlir.dialects.scf` — structured control flow (`IfOp`, `ForOp`,
  `WhileOp`, `YieldOp`, `ConditionOp`)
- `mlir.dialects.util` — `UndefOp`, `PackOp` / `UnPackOp`, `GetTupleOp`
  (for records / closures as MLIR tuples)
- `mlir.dialects.db` — LingoDB's database dialect
  (`StringType`, `ListType`, `DictType`, `DateType`, `IntervalType`,
  `NullableType`, `ConstantOp`, `CmpOp`, `CastOp`, `RuntimeCall`,
  list/dict ops, `TryExcept`, `Hash`, `IsNullOp`, `NullableGetVal`, …)
- `mlir.dialects.py_interp` — Python-object interop (`PyObject` type,
  `CastToPyObject`, `CastFromPyObject`, `GetAttr`, `Call`, `ImportOp`) —
  used for the pyobj fallback path
- `mlir.dialects.relalg` — `SQLQueryOp` (used by `sql.execute`)
- `mlir.dialects.subop`, `builtin`, `tuples` — imported, reserved

Two module-level globals hold the active MLIR context (`curr_context`)
and top-level module (`curr_module`) for the duration of a compile.
Marked `todo: remove this hack` in-source — see Gotchas.

## 2. Type lowering — `to_mlir_type(t)`

| `hipy.ir` type | MLIR type |
|---|---|
| `BoolType()` | `i1` |
| `IntegerType(w)` | `i{w}` (signless) |
| `IntType()` | `i64` |
| `FloatType(32|64)` | `f32` / `f64` |
| `StringType()` | `db.StringType` |
| `RecordType(members)` | `tuple<t1, t2, …>` (MLIR tuple of member types, names dropped — index-addressed via `util.GetTupleOp` / `PackOp`) |
| `FunctionRefType(args, res, closure=None)` | `(args...) -> res` MLIR function type |
| `FunctionRefType(args, res, closure=T)` | `tuple<(args...) -> res, T>` — closure packed alongside the raw fn |
| `ListType(elem)` | `db.ListType(elem)` |
| `DictType(key, val)` | `db.DictType(key, val)` |
| `VoidType()` | `i1` (dummy — MLIR has no void; see Gotchas) |
| `PyObjType()` | `py_interp.PyObject` |
| `DateType()` | `db.DateType(day)` |
| `IntervalType()` | `db.IntervalType(daytime)` |
| `NullableType(inner)` | `db.NullableType(inner)` |
| anything else | `assert False` |

Still **not** lowered: `ColumnType`, `TableType`, `ArrayType` — whatever
uses those will fall through the fallback assertion.

## 3. Statement lowering — `to_mlir_stmt(stmt, mapping)`

Pattern-matches each IR op and emits the corresponding MLIR ops into the
current `InsertionPoint`. Keeps a `mapping: dict[ir.SSAValue,
mlir.Value]` threaded through so consumers see the lowered producer's
result.

Supported ops:

- **`ir.Return`** — `func.ReturnOp([v])`, or `func.ReturnOp([])` if the
  return type is void.
- **`ir.Constant`** — dispatches on result type: bool → `arith.ConstantOp(i1, 0|1)`;
  integer/float → `arith.ConstantOp`; string → `db.ConstantOp(db.StringType, StringAttr)`;
  void → `util.UndefOp`; list → `assert False`.
- **`ir.CallBuiltin`** — dispatches on `(name, arg_types)`; see §4.
- **`ir.Call`** — `func.CallOp`. Void-returning `ir.Call` currently
  asserts `False` (known gap).
- **`ir.CallIndirect`** — routed through the `call(callee, args, mapping)`
  helper, which handles three cases: (a) `function_ref` with no closure
  and a known `FunctionRef` producer → `func.CallOp`, (b) same without a
  known producer → `func.CallIndirectOp`, (c) `function_ref` with a
  closure — unpacks the closure from the producer's `FunctionRef(closure=…)`
  and passes it as the trailing argument.
- **`ir.IfElse`** — `scf.IfOp` with then/else blocks populated by
  recursing into nested blocks; `Yield` → `scf.YieldOp`.
- **`ir.FunctionRef`** — `func.ConstantOp(FlatSymbolRefAttr)` if no
  closure; otherwise `util.PackOp([raw_fn, closure])` to tuple up the
  closure.
- **`ir.MakeRecord`** — empty record → `util.UndefOp`; otherwise
  `util.PackOp` in declared-member order.
- **`ir.RecordGet`** — linear lookup of the member index, then
  `util.GetTupleOp`.
- **`ir.PyGetAttr`** — `py_interp.GetAttr`.
- **`ir.PythonCall`** — `py_interp.Call`; keyword args are currently
  `assert False` (only positional supported).
- **`ir.PyImport`** — `py_interp.ImportOp`.

Still unsupported: `ir.PySetAttr`, `ir.Undef` as a top-level op, `ir.Free`.

## 4. Supported CallBuiltin names

The backend's single big `match` covers (organized by category):

**Int arithmetic (`IntegerType` + `IntType`):** `scalar.int.add/sub/mul/div/mod/lshift`
→ `arith.{Add,Sub,Mul,DivS,RemS,ShL}IOp`. Compares
`scalar.int.compare.{eq,neq,lt,lte,gt,gte}` → `arith.CmpIOp(eq|ne|slt|sle|sgt|sge)`.

**Float arithmetic:** `scalar.float.{add,sub,mul,div,mod,neg}` →
`arith.{Add,Sub,Mul,Div,Rem,Neg}FOp`. Compares → `arith.CmpFOp(OEQ|ONE|OLT|OLE|OGT|OGE)`
(ordered predicates).

**Float transcendentals** (via `db.RuntimeCall`): `scalar.float.{exp,log,sqrt,sin,cos,acos,pow,ceil,round}`
→ `db.RuntimeCall("Exp"|"Log"|…|"Pow"|"Ceil"|"RoundFloat")`; `scalar.float.atan2`
→ `ATan2`.

**Bool:** `scalar.bool.and/or` → `arith.{And,Or}IOp`; `scalar.bool.not` →
`arith.XOrIOp(v, 1)`.

**Int↔float / ↔string / ↔pyobj:**
- `scalar.float.from_int(IntegerType|IntType)` → `arith.SIToFPOp`
- `scalar.float.to_int` → `arith.FPToSIOp`
- `scalar.int.pyint_to_int64(IntType)` → identity (IntType already lowers to i64)
- `scalar.int.from_string` / `scalar.float.from_string` → `db.CastOp`
- `scalar.int.to_string` / `scalar.float.to_string` → `db.CastOp`
- `scalar.{string,int,float}.to_python` → `py_interp.CastToPyObject(…, "builtins.{str|int|float}")`
- `scalar.string.from_python` → `py_interp.CastFromPyObject(…, "builtins.str")`

**String ops (`db.CmpOp` / `db.RuntimeCall`):** compare.{eq,lt,lte}; `lower`/`upper`
→ `ToLower`/`ToUpper`; `contains`; `length` → `StringLength`; `find`/`rfind`
→ `PyStringFind`/`PyStringRFind`; `substr` (off+1 adjustment); `replace`; `concatenate`;
`strip` → `StringStrip`; `ord` → `Ord`; `at` → `Substring(s, i+1, 1)`;
`split` → `StringSplit`. Formatting via `scalar.string.format_single` →
`FmtDouble` / `FmtInt` depending on the value type.

**Iteration builtins** (the interesting ones — these compile a whole
loop with a callback):
- `scalar.string.iter` → `scf.ForOp` stepping 1..len+1, passing
  `Substring(s, i, 1)` into the per-iteration callback.
- `list.iter` → `scf.ForOp` 0..len stepping 1, `db.ListGetOp` per
  element.
- `range.iter` — two paths. If the step is a **positive compile-time
  constant** (`is_positive_constant` matches an `ir.Constant` with `v>0`),
  emits a plain `scf.ForOp`. Otherwise emits a `scf.WhileOp` with a
  "negative step?" select to pick between `>` and `<` exit conditions at
  runtime.
- `dict.iter_items` — `scf.WhileOp` with `db.DictGetIter` / `DictIterValid`
  / `DictIterGetKey` / `DictIterGetValue`, yielding `tuple<key, val>`
  into the callback.
- `while.iter(cond_fn, body_fn, …)` → `scf.WhileOp` where the `before`
  block calls `cond_fn` and the `after` block calls `body_fn`.

All callbacks are dispatched through the `call(callee, args, mapping)`
helper (§3), so closures go through automatic unpack.

**List / dict ops:** `list.create`, `list.at` (i64 → index cast), `list.append`,
`list.length` (returns i64 via `IndexCastOp`), `list.set`, `list.sort`
(requires a `FunctionRef` without closure — sees note in Gotchas);
`dict.create` (requires a plain `FunctionRef` for the key-eq function —
closure case asserts with the message *"Dict creation with closure type
not supported"*), `dict.contains`, `dict.get`, `dict.set` (each builds a
hash via `db.Hash`).

**Date / interval:** `date.diff(date, date) → interval` → `DateDiffInterval`;
`interval.days` → `IntervalGetDay`.

**Nullable / SQL:** `nullable.is_null` → `db.IsNullOp`; `nullable.get_value`
→ `db.NullableGetVal`; `sql.execute(query_const_str, params...)` →
`relalg.SQLQueryOp` — the first arg must be an `ir.Constant` string
(extracted via `str_from_const`).

**Control / misc:** `undef` → `util.UndefOp`; `dbg.print` → `db.RuntimeCall("DumpValue")`
(no result); `regex.search` → `RegexSearch`; `error(msg)` →
`db.RuntimeCall("RaiseRuntimeError", [msg])`; `try_except(try_fn, except_fn)`
→ `db.TryExcept`, with the closure tuples auto-unpacked if present.

Unrecognized builtins print `"Can not translate op <name> for types
<args>"` to stderr then **`assert False`** — the older "silently emit
UndefOp" behavior is no longer in the live path (an unreachable
`UndefOp` follows the assertion but is never executed).

## 5. Top-level — `to_mlir_func` and `to_mlir_module`

- `to_mlir_func(fn)` — creates `func.FuncOp` with the lowered signature,
  opens an entry block, seeds `mapping` with the block's arguments,
  lowers every op. If no explicit `Return` was emitted, emits
  `func.ReturnOp([])` at the end.
- `to_mlir_module(module)` — creates an MLIR `Context` via
  `mlir_init.init_context(context)` (LingoDB-specific — registers the
  dialects), wraps an `mlir.Module.create()`, and lowers every
  `module.funcs()`.
- `compile(module)` — the public entry. Returns the MLIR module object.
  **Does not compile or execute** — the caller decides what to do next.

## 6. What's still missing

In rough order of "what you'd need to make this a production backend":

- **Composite types.** `ColumnType`, `TableType`, `ArrayType` — not
  lowered. This is the biggest gap for database-style code; the `relalg`
  dialect is where these would land but the bridge is only wired for
  `SQLQueryOp` so far.
- **`ir.Call` with void return.** Explicitly asserts `False`.
- **`ir.PySetAttr`, `ir.Undef`, `ir.Free`** — not handled in the
  top-level dispatch.
- **Keyword args to `ir.PythonCall`** — asserted out.
- **Runtime / execution driver.** `compile()` returns an MLIR module
  that the caller must run through LingoDB's own passes / JIT — there is
  no equivalent of `cppbackend.write_compile_run_cpp`.

## 7. When to use this

Today: research target, and the path toward database-integrated codegen
via LingoDB (`relalg`, `subop`). Run real programs through the C++
backend; use this one when you want to inspect the MLIR form, test a
`sql.execute` path, or prototype a new op before writing the C++ side.

## 8. Gotchas

- **`curr_context` / `curr_module` are module-level globals.** Don't run
  two `compile(module)` calls concurrently in the same process; swap the
  globals for a context manager if you ever need to. The source marks
  this `todo: remove this hack`.
- **`VoidType` lowers to `i1`.** MLIR has no void type; returns of void
  become zero-arg `func.ReturnOp()` but any `Constant` of void type gets
  a dummy `i1` UndefOp value. Be careful when chaining through void
  producers.
- **`db` dialect is LingoDB-specific.** Running this outside a
  `lingodbbridge`-built MLIR will fail at import time. Don't add it as
  a default dependency.
- **String ops and many float ops go through `db.RuntimeCall` with a
  string-attribute name** (e.g. `"ToLower"`, `"Substring"`, `"Pow"`,
  `"RegexSearch"`). These names must match what LingoDB's runtime
  actually exposes — grep LingoDB's runtime for the matching symbol
  before adding a case here.
- **Substring offset is shifted by +1** for the LingoDB convention
  (`scalar.string.substr` and `scalar.string.at`).
- **`list.sort` produces no real result.** The C++ backend allocates a
  new list and returns it. Here `db.ListSortOp` is emitted for its
  side-effect and the SSA result is filled with `util.UndefOp` — any
  caller that reads the result will see undef. If `list.sort` ever
  switches to returning a new list, revisit this case.
- **`dict.create` requires a non-closure `FunctionRef`** for the key
  equality/hash function. A closured function asserts with the message
  *"Dict creation with closure type not supported"* (copy-pasted from
  the list.sort case — note the dict-vs-list mismatch).
- **`sql.execute` requires a constant query string** as its first arg —
  `str_from_const` matches an `ir.Constant(v=str)` and otherwise
  `assert False`.
- **Unhandled CallBuiltins now raise `assert False`** (older revisions
  silently emitted `UndefOp`). Expect loud failures rather than broken
  modules — a win, but any script that relied on partial compilation
  will break.
