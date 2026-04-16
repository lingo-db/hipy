# GeneratorContext — The Runtime of the Program Generator

**File:** `hipy/context.py` (≈1250 LOC)

After `hipy/compiler.py` rewrites a Python function into a **program
generator**, that generator takes a `_context` keyword argument and executes
against a `Context` object. Every `_context.*` method in the rewritten AST is
dispatched here. `Context` owns:

- the **IR module** under construction,
- the **current block** (mutable, swapped on block enter/exit),
- **transaction / event streams** for escape analysis,
- **location stacks** that give every action a stable identifier,
- the **fallback machinery** for dispatching unsupported ops to CPython.

This is the single most load-bearing file in HiPy. Everything paraphrased in
paper §4 ("Implementation") concretely lives here.

## 1. Event types

The post-mortem escape analysis in `compiler.compile_function` builds a graph
from the `context.events` list. Each event is one of:

| Class | Meaning |
|---|---|
| `ValueAlias(val1, val2)` | Two `ValueHolder`s logically refer to the same object (e.g. after a merge). |
| `NestedValueAlias(nested, container)` | `nested` is contained inside `container` — propagates "conversion taints" through containers. |
| `ConvertedToPython(val)` | `val` was lowered to a `pyobj` representation. In the Union-Find this unions the value's node with the special `invalid` node. |
| `ValUsage(val)` | `val` was used (read). If it can reach `invalid` in the graph, the value should have been converted *before* this point — its location is recorded for the next rerun. |

`Context.track_using`, `track_same`, `track_nested`, `track_converted_to_python`
emit these events. Ops on `lib.builtins.object` (i.e. already `pyobj`) are
skipped — they can't be converted further.

## 2. Stable identifiers and the location stack

Every `_context.<method>(…, _action_id=N)` call is tagged by the rewriter with
a unique integer ID per function. On entry, `handle_action(action_id)` pushes
that ID into `context.location[-1]`. The tuple `(location, nested_location, …)`
becomes the "t_location" — a stable key for an allocation site across
regeneration runs.

Scopes pushed onto the stack:
- `NewScope` (returned by `handle_action`) — one per context method call.
- `transaction:<id>` — pushed by `with ctxt.transaction() as t:`; scopes
  every value created inside a transaction.
- Fixed labels like `"if"`, `"else"`, `"try"`, `"except"`, `"cond"`, `"body"`,
  `"const_for{i}"`, `"args"` — pushed by `_if`, `_try`, `_for`, `_while`, and
  `compile_function` for argument construction. These labels keep tuples
  stable even across reruns.
- `DynAction` generates `dyn:<counter>` IDs for unannotated actions
  (e.g. transient values inside transactions).

The `decisions` list passed to the `Context` constructor is the set of
t_locations that the *previous* generation run flagged as problematic; they
correspond to places where values should be force-converted to `pyobj`
*before* they are used. `Context.unwrap` / `wrap` / `to_python` consult this
list to pre-empt a late conversion.

## 3. Fallback control

- `ctx.fallback()` returns True iff the stack of `no_fallback()` contexts is
  empty. Inside `with ctx.no_fallback(): …`, methods re-raise
  `NotImplementedError` instead of falling back to CPython.
- `perform_call`, `perform_binop`, `get_attr`, `get_item`, `neg_`, `invert_`,
  `in_`, etc. all catch `(NotImplementedError, TypeError, AttributeError)` and:
  1. If fallback is on and we are not already on a `pyobj`, retry the op on
     the `pyobj` form of the receiver (via `to_python`).
  2. Otherwise, re-raise.
- `perform_binop` implements the paper's ternary semantics:
  try `left.__op__(right)`, then `right.__rop__(left)`, then convert left to
  `pyobject` and retry.

`no_fallback` is used internally whenever the context itself wants to detect
"op not implemented" (e.g. trying `__constiter__` to enter a const-for) and
handle it itself, without the ambient fallback reacting.

## 4. Transactions

`ctx.transaction()` returns a `Transaction` context manager. Entering it:
- swaps out `module`, `block`, `events`, `seen` for fresh ones,
- pushes a `transaction:<id>` location label + a new DynAction.

Inside, every `_context.wrap(value)` creates `ValueHolder`s whose
`t_location` is scoped to the transaction. The transaction records every
mutation of `ValueHolder.value` in an `undo_log`.

On exit:
- `commit()` — merges the transaction's module/block/events into the parent,
  and propagates the undo_log upward (nested transactions).
- `abort()` — discards everything, calls `undo_value_changes()` to restore
  all `ValueHolder.value` bindings to their pre-transaction state.
- `undo_value_changes()` — returns the *map of changes that were made*, and
  restores the original values. Used by `_if`/`_while`/`_for` to inspect what
  each branch/iteration did before deciding how to merge.

Transactions are how the generator can *speculatively* evaluate branches that
may not end up in the final IR (both arms of an `if`, loop bodies during
fix-point, try/except bodies) without polluting the outer state.

## 5. Control flow — `_if`, `_while`, `_for`, `_try`

All four follow the same general pattern:

1. **Try to statically resolve.** `_if` with a constant condition just calls
   the chosen branch. `_for` tries `__constiter__` on the iterable
   (e.g. `_concrete_list` exposes a constant iterator) and inlines the
   iteration at generation time.
2. **Speculate in a transaction.** Run each branch (or one iteration of the
   loop body) inside `ctx.transaction()`, emitting IR into a fresh block.
   Capture the produced values and side-effects.
3. **Type-converge via fix-point.** Loops: if the iteration variables' types
   differ after the body ran (vs before), merge them, feed the merged types
   back, and re-run. If the types stabilize after ≤3 iterations, emit the
   final IR; otherwise raise "too many iterations".
4. **Merge branch results.** Call `ctx.merge(val1, val2, fn_val1, fn_val2)`
   on each result pair. `__merge__` on the virtual type returns
   `(merged_val1, merged_val2, create_merged)` with branch-local conversion
   functions (`fn_val1`, `fn_val2`) injecting any extra ops into the branch's
   block. If no merge is defined, fall back to `pyobject`.
5. **Emit structured op.** `_if` emits one `ir.IfElse` with two nested blocks
   each terminating in `ir.Yield`. `_while`/`_for` emit IR `Function`s for
   the body (and for `_while`, the condition), package inputs/iter-vals as
   records, and call `"while.iter"` or `iter_val.__iterate__(...)` as a
   `CallBuiltin`.
6. **Early return handling.** The rewriter turned nested `return` into
   `raise _context.EarlyReturn(val)`. `_if` / `_try` / `_for` / `_while`
   catch it inside their branch transaction. If *both* branches raised,
   the early-return is re-raised at the outer level. If only *one* branch
   did, `NotImplementedError("early return in only one branch")` — left as a
   limitation.

`_try` additionally materializes the try/except bodies as two IR functions
with closure record arguments (via `binding.create_closure`), wires them into
a single `try_except` `CallBuiltin`, and unpacks the result record.

## 6. Expression-level dispatch methods

These are the targets of the AST rewrite in `compiler.py`. Each one is one
case in the rewriter's table. Non-exhaustive list:

| Context method | Role |
|---|---|
| `constant(val)` | Wrap a Python constant in a `_const_<type>` virtual value. |
| `create_tuple / create_list / create_dict / create_dict_simple / create_slice` | Build virtual aggregate values. `create_list` logs `NestedValueAlias` events for every element. |
| `create_lambda(staged, bind_python, bind_staged)` | Wraps the 3-variant lambda into a `LambdaValue`. |
| `generator_expr(iter_fn, iterable, type_infer_fn, packed_vals)` | Packages the helper functions the rewriter emitted for a `(... for ... in ...)` expression into a `GeneratorExpressionValue`. `packed_vals` is a tuple of the free variables read by the element expression (so the generator body can reconstruct its closure). The iter fn is always wrapped as an `HLCFunction`. |
| `infer_return_type(fn, arg_types_input)` | Runs `fn(*fake_args)` inside an **aborted** `transaction()` to read the return type without emitting IR. Inputs are a `_concrete_list` of `TypeValue`/`HLCClassValue`. Used by `GeneratorExpressionValue.__itertype__` and similar "what would this call return?" probes. |
| `get_by_name / get_recursive_raw / get_attr / set_attr / get_item / set_item` | Name and attribute resolution. `get_by_name` is also where Python modules and HiPy-mocked libs are wired in (see `mocked_modules` map), and where functions in the **base module** have their source registered into `module.py_functions` for the fallback. |
| `perform_call(fn, args, kwargs)` | Core dispatch. Handles `HLCFunction`, `HLCMethod`, `HLCClassValue`, `LambdaValue`, `Value(__call__)`, and falls back to `pyobj` on failure. |
| `perform_binop / perform_unaryop` | Binary/unary operators — retries with reversed method, then converts to pyobj on failure. |
| `bool_and / bool_or / bool_not / _to_bool` | Short-circuit-preserving boolean handling with constant folding and `scalar.bool.*` builtins. |
| `is_ / in_ / neg_ / invert_` | `is`, `in`, unary -, `~` — with fallback builtins (`python.operator.*`) for the `pyobj` path. |
| `unpack / const_unpack` | Tuple unpacking: `unpack` iterates via `__getitem__`; `const_unpack` uses `__constiter__` to stay at generation time. |
| `call_builtin(fn, res, args, side_effects, attributes)` | Emits an `ir.CallBuiltin` op with given name and typed result. This is the bridge every virtual type uses to produce IR. |
| `import_pymodule(name)` | Emits `ir.PyImport`. For the base module, also walks the host module namespace and registers every non-hipy import into `module.imports` so the C++ backend emits the right `import` preamble. |
| `to_python(val)` | Materializes a virtual value to `pyobj` via `__topython__`. For mutable values, mutates the `ValueHolder` in place so later ops see the pyobj form. Logs `ConvertedToPython`. |
| `merge(val1, val2, fn_val1, fn_val2)` | Symmetric: tries `type(val1).__merge__` first, then `type(val2).__merge__` with swapped args. Last resort: convert both sides to `pyobj`. |
| `pyobj(val)` | Wrap a CPython SSA value in a `lib.builtins.object`. |
| `is_mutable(val)` | Checks `__HIPY_MUTABLE__` on the value's class. Lambdas are explicitly not mutable. |
| `get_abstract_type(val)` | Maps a Python primitive to its HiPy type (for args construction). |

### 6.1 `handle_action`, `NewScope`, `DynAction`

Every public context method wraps its body in `with self.handle_action(_action_id): …`.
That pushes a `NewScope` onto the location stack so any `ValueHolder.t_location`
created inside is stable across reruns. If `_action_id` is `None`, a dynamic
ID is generated from the innermost `DynAction`.

## 7. Helper objects

- `NewScope` — context manager pushing/popping a slot on
  `location` and `dyn_actions`. Every `handle_action` returns one.
- `DynAction` — counter on the dyn_actions stack; generates `"dyn:N"` IDs
  for actions that don't have a static `_action_id`.
- `Context.EarlyReturn(Exception)` — thrown by `early_return(val)`; caught by
  `_if`/`_try`/`_for`/`_while` and by the rewritten top-level `try` wrappers.

## 8. Invariants for maintainers

- **Every public method takes `_action_id=None`** and wraps its body in
  `with self.handle_action(_action_id): …`. Forgetting this breaks escape
  analysis — the value's t_location becomes non-reproducible.
- **`ctx.wrap` returns a fresh `ValueHolder`**; never wrap a `ValueHolder`
  again (the `match ValueHolder(): assert False` case).
- **Transactions can nest.** `commit()` propagates undo logs upward; if the
  outer transaction then aborts, the inner transaction's effects also roll
  back. Don't short-circuit a nested commit.
- **`to_python` of a *mutable* value mutates the holder in place.** Subsequent
  reads see the pyobj form — this is critical for soundness (multiple
  representations of one logical value must stay in sync).
- **`_if`/`_while`/`_for` all use fix-point type inference.** If you add a
  new virtual type that participates in control-flow merges, implement
  `__merge__` or expect silent `pyobject` fallback on divergence.
- **`counter > 2`** in `_while`/`_for` bounds the fix-point iteration count —
  if you need more iterations for legitimate reasons, increase it carefully;
  runaway fix-points were an early source of hangs.
- **`__deepcopy__` returns `self`.** The `Context` must be identity-preserved
  across `copy.deepcopy` (used by transaction snapshotting of `seen`).
- **Recursion is handled specially.** `get_recursive_raw` uses
  `sys.modules[module]` to resolve the currently-being-compiled function by
  going through the Python module; ensure the function is reachable by
  attribute access there.
