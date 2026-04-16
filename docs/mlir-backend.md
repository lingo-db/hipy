# MLIR backend — `hipy/mlirbackend/`

**File:** `hipy/mlirbackend/__init__.py` (~280 LOC)

A **prototype** / **stub** backend that lowers a subset of `hipy.ir` to
MLIR via the external [LingoDB](https://github.com/lingo-db/lingo-db)
project's MLIR Python bindings (`lingodbbridge.mlir`). Does **not**
currently compile to an executable — `compile(module)` returns the MLIR
module as a Python object; downstream execution is out of scope for this
file. Most HiPy programs use the C++ backend (see `cpp-backend.md`).

Think of this as a research vehicle for "what would happen if we pointed
HiPy at a database-aware MLIR stack instead of emitting C++?" — the paper
mentions LingoDB as the inspiration for integrating DB-style codegen.

## 1. Dependencies

All MLIR access goes through `lingodbbridge`, which bundles MLIR Python
bindings + custom LingoDB dialects:

- `mlir.dialects.func` — function / call / return
- `mlir.dialects.arith` — scalar arithmetic (constants, add/sub/mul/div,
  comparisons — both signed-integer and floating-point)
- `mlir.dialects.scf` — structured control flow (`IfOp`, `YieldOp`)
- `mlir.dialects.util` — `UndefOp` for placeholders
- `mlir.dialects.db` — LingoDB's database dialect (`StringType`,
  `ConstantOp`, `CmpOp`, `CastOp`, `RuntimeCall` for `ToLower`,
  `Contains`, `StringLength`, `PyStringFind`, `Substring`, `Replace`,
  `Concatenate`, `DumpValue`, …)
- `mlir.dialects.tuples`, `relalg`, `subop`, `builtin` — imported but not
  currently used by the lowering (reserved for future relational ops).

A `curr_context` / `curr_module` pair of module-level globals holds the
active MLIR context for the duration of a compile. Marked `todo: remove
this hack` in-source.

## 2. Type lowering — `to_mlir_type(t)`

| `hipy.ir` type | MLIR type |
|---|---|
| `BoolType()` | `i1` (`IntegerType.get_signless(1)`) |
| `IntegerType(w)` | `i{w}` (signless) |
| `IntType()` | `i64` |
| `FloatType(32)` | `f32` |
| `FloatType(64)` | `f64` |
| `StringType()` | `db.StringType` (LingoDB) |
| anything else | `assert False` |

Composite types (list, dict, record, tuple, table, column, array,
function-ref, pyobj) are **not** supported. The `_` fallback asserts.

## 3. Statement lowering — `to_mlir_stmt(stmt, mapping)`

Pattern-matches each IR op and emits the corresponding MLIR ops into the
current `InsertionPoint`. Keeps a `mapping: dict[ir.SSAValue,
mlir.Value]` threaded through so consumers see the lowered producer's
result.

Supported ops:

- **`ir.Return([v])`** — `func.ReturnOp([v])`, or `func.ReturnOp([])` if
  `v.type` is void.
- **`ir.Constant`** — dispatches on result type:
  - bool → `arith.ConstantOp(i1, 0|1)`
  - integer/float → `arith.ConstantOp`
  - string → `db.ConstantOp(db.StringType, StringAttr)`
  - list → `assert False` (not implemented)
  - other → `arith.ConstantOp` if integer-y; otherwise assert.
- **`ir.CallBuiltin`** — dispatches on `(name, arg_types)`. The full
  lowering table (see §4 below) covers the arith & string subset.
- **`ir.Call`** — `func.CallOp`. Void returns are asserted against (the
  `if isinstance(r.type, ir.VoidType): assert False` branch is a known
  gap).
- **`ir.IfElse`** — `scf.IfOp` with then/else blocks populated by
  recursing into nested blocks. Yields through `scf.YieldOp`.
- **`ir.Yield`** — `scf.YieldOp`.
- **`ir.FunctionRef`** — `func.ConstantOp` carrying a
  `FlatSymbolRefAttr`.

Everything else raises `assert False` — `FunctionRef` with a closure,
`CallIndirect`, `MakeRecord`, `RecordGet`, `PyImport`, `PyGetAttr`,
`PySetAttr`, `PythonCall`, `Undef`, `Free`, and every `CallBuiltin` not
in the table below are unsupported.

## 4. Supported CallBuiltin names

The `match (name, arg_types)` arm enumerates:

**Integer (`IntegerType` and `IntType`):** `scalar.int.add`, `.sub`,
`.mul`, `.div`, `.mod`, `.lshift`, `.compare.eq`, `.neq`, `.lt`, `.lte`,
`.gt`, `.gte` → `arith.AddIOp` / `SubIOp` / `MulIOp` / `DivSIOp` /
`RemSIOp` / `ShLIOp` / `CmpIOp(<predicate>)`.

**Float:** `scalar.float.add`, `.sub`, `.mul`, `.div`, `.compare.{eq,neq,lt,lte,gt,gte}`
→ `arith.AddFOp` / … / `CmpFOp(OEQ|ONE|OLT|OLE|OGT|OGE)` (ordered
predicates).

**Bool:** `scalar.bool.and` → `arith.AndIOp`; `.or` → `arith.OrIOp`;
`.not` → `arith.XOrIOp(v, const_1)`.

**Int↔float:** `scalar.float.from_int` → `arith.SIToFPOp`.

**String ops:** via `db.CmpOp` (eq/lt/lte) and `db.RuntimeCall` for
`ToLower`, `Contains`, `StringLength`, `PyStringFind`, `PyStringRFind`,
`Substring` (with off+1 adjustment), `Replace`, `Concatenate`.

**Conversions:** `scalar.int.from_string` → `db.CastOp`.

**Misc:** `undef` → `util.UndefOp`; `dbg.print(str)` →
`db.RuntimeCall("DumpValue", [s])` with no result type.

Unrecognized builtins print `"Can not translate op <name> for types
<args>"` to stderr and return a `util.UndefOp` placeholder. This lets
partially-unsupported functions still produce a complete MLIR module for
inspection (without failing the whole compile).

## 5. Top-level — `to_mlir_func` and `to_mlir_module`

- `to_mlir_func(fn)` — creates `func.FuncOp` with the lowered signature,
  opens an entry block, seeds `mapping` with the block's arguments, lowers
  every op. If no explicit `Return` was emitted, emits
  `func.ReturnOp([])` at the end.
- `to_mlir_module(module)` — creates an MLIR `Context` via
  `mlir_init.init_context(context)` (LingoDB-specific initialization,
  registers the dialects), wraps an `mlir.Module.create()`, and lowers
  every `module.funcs()`.
- `compile(module)` — the public entry. Returns the MLIR module object.
  **Does not currently compile or execute** — the caller decides what to
  do next (print, run LingoDB's passes, etc.).

## 6. What's missing

In rough order of "what you'd need to make this a production backend":

- **Composite types.** Record, list, dict, tuple, column, table, array,
  function-ref-with-closure, pyobj — none are lowered. The DB dialect has
  plenty of composite types (tuples, lists, records) that could back
  these, but the bridge code isn't written.
- **Function refs with closures.** Currently only simple function
  references — closures would require encoding the closure record in a
  dialect-appropriate way (and likely a `CallIndirect` lowering).
- **`ir.Call` with void return.** The code explicitly asserts false in
  this branch.
- **Relational ops.** `relalg` / `subop` dialects are imported but never
  used — the intention is presumably to lower `table.*` ops through
  LingoDB's relational dialect (which is where most of the value of this
  backend would come from).
- **Runtime / execution driver.** There is no equivalent of
  `cppbackend.write_compile_run_cpp` — `compile()` returns an MLIR module
  that the caller must run through LingoDB's own passes / JIT.

## 7. When to use this

In the current state: as a target for IR-level tests that want to check
"is this subset of the IR representable as MLIR?", or as a scaffold for
future DB-integrated lowering.

The real backend for running programs is `hipy/cppbackend/`. This one is
kept in the repo as a reference / research artifact for a future port.
See the OOPSLA'24 paper §7 (§"Integrations") for context on the LingoDB
angle.

## 8. Gotchas

- **`curr_context` / `curr_module` are module-level globals.** Don't run
  two `compile(module)` calls concurrently in the same process; swap the
  globals for a context manager if you ever need to.
- **`db` dialect is LingoDB-specific.** Running this outside a
  `lingodbbridge`-built MLIR will fail at import time. Don't add it as
  a default dependency.
- **String operations go through `db.RuntimeCall` with a string attribute**
  (e.g. `"ToLower"`, `"Substring"`). These names must match what
  LingoDB's runtime actually exposes — if you add new string ops in
  `hipy/lib/builtins.py`, grep LingoDB's runtime for the matching symbol
  before adding a case here.
- **Substring offset is shifted by +1** for the LingoDB convention. Keep
  that in mind when copying a builtin from `cpp-backend`.
- **Unhandled ops produce `UndefOp` silently (for CallBuiltin only).**
  Other unsupported ops raise `assert False`, so failures surface loudly;
  unsupported *builtin names* are the ones to watch out for — check
  stderr after compile.
