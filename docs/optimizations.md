# Optimization passes — `hipy/opt/`

**Files:** `pattern_rewriter.py`, `canonicalization.py`,
`eliminate_dead_code.py`, `eliminate_dead_symbols.py`, `inline.py`,
`eager_free.py`, `array_patterns.py`, `tabular_patterns.py`.

After the generator has emitted IR via `hipy/compiler.py` + `hipy/context.py`,
the module passes through a sequence of optimization passes before the
backend lowers it further. Passes are thin: most are a handful of pattern
match arms around a `PatternRewriter`.

> **Historical note:** the OOPSLA'24 prototype also shipped a
> data-centric codegen pass (`dccg.py`) and a pre-backend lowering pass
> (`rewrite_cpp.py`). Both were removed after it turned out they were no
> longer wired into any entry point, along with the backend-only
> `cppir` op module they emitted into.

## 1. `pattern_rewriter.py` — the rewrite driver

```python
class PatternRewriter:
    def __init__(self, patterns: list[RewritePattern], module: ir.Module): ...
    def rewrite(self): ...
```

- Walks every op in every function body, top-down (outer ops first). Each op
  is tried against every pattern in order; the first pattern to return
  `True` wins. Non-matching ops then recurse into their nested blocks
  (for `IfElse`, loops, etc.).
- Maintains a `uses: dict[ir.SSAValue, set[ir.Operation]]` index. Every
  helper keeps this index consistent so DCE-on-demand works after each
  rewrite.
- Key methods for patterns:
  - `replace_with(op, cls, *args, **kwargs)` — constructs a new op via
    `cls(before_current(), …)`, forwards uses of the old result SSA to the
    new one, requeues the new op for further matching. Asserts the
    produced types match (so patterns can't silently change types).
  - `replace_with_value(op, value)` — replace a single-result op with an
    already-produced SSA.
  - `remove(op)`, `maybe_remove(op)` — `maybe_remove` only deletes if no
    live uses and no side effects.
  - `create(cls, *args)` — insert a new op just before the op currently
    being rewritten (via `before_current()`), update the uses index.
  - `before(beforeop)` — insert before any specific op (not just the
    current one).

Every pattern subclasses `RewritePattern` and implements
`rewrite(op, rewriter) -> bool`. Return `True` if the op was rewritten,
`False` to let other patterns (or the recursion) have a go.

## 2. Small, general passes

### `canonicalization.py`

One pattern: `RecordGetPattern` folds `RecordGet(MakeRecord(…), member)` →
the value stored for `member`. Cleans up the record plumbing that the
generator emits heavily for closures / tuples / row records.

`canonicalize(module)` is the public entry.

### `eliminate_dead_code.py`

Classic backward liveness per function body. Walks `block.ops` in reverse:
- `Return` / `Yield` mark their operands live.
- A side-effect-free op whose results are unused is dead.
- Otherwise, its operands are marked live.

Recurses into nested blocks first (so liveness is established bottom-up
through `IfElse` bodies etc.). Removes dead ops in-place.

`run(module)` is the entry.

### `eliminate_dead_symbols.py`

Reachability-based function pruning. Starting from a `root` function name,
walks every `ir.Call(name=...)` and `ir.FunctionRef(name=...)` it can reach
through nested blocks, builds the set of live function symbols, and drops
everything else from `module.block.ops`. Run this at the end of the
pipeline, after inlining and function fusion have potentially orphaned
helpers.

`run(module, root)` is the entry.

### `inline.py`

One pattern: `InlineFunc` inlines every `ir.Call(name, args)` by cloning
the callee's body op-by-op into the caller (building a `{param: arg}`
remapping), and replacing the call result with the cloned `Return`'s
operand. Assumes every function has a single `Return` (the generator's
invariant).

Run *aggressively* after cogen to enable subsequent pattern matching —
e.g., `FuseColumnApply` relies on seeing the raw `FunctionRef`-to-closure
chains, not abstract calls.

### `eager_free.py`

Post-pass (not pattern-based): walks each function body in reverse,
remembering the *last* use of each SSA value. After that use, inserts an
`ir.Free` op. This is how the C++ backend knows when to release owned
resources (allocated strings, columns, tables). Runs once, late in the
pipeline.

## 3. Array & tabular pattern fusion

### `array_patterns.py`

- `canonicalize_array_ops` normalizes numpy builtins: both
  `array.apply_scalar` and `array.binary_op` are rewritten to a common
  `array.compute(array…, func)` form — a single op the backend can fuse.
- `fuse_array_ops` merges chained `array.compute`s. When the outer
  `array.compute(f, x1, x2…)` has an argument whose producer is another
  `array.compute(g, y1, y2…)`, the pass:
  1. Collects every upstream `FunctionRef` + closure into a single
     closure record `RecordType([("f0", c0.type), ("f1", c1.type), …])`,
  2. Synthesizes a new IR function `fused_N` with arg types
     `[all_input_element_types…, closure_record_type]` that unpacks the
     closure members, calls each original function with its sub-args, then
     calls the outer function with all the sub-results, and returns that.
  3. Emits a single `array.compute(all_arrays…, fused_func_ref)` op.

The resulting array op runs one pass over the input arrays and does all
the arithmetic in-register — equivalent to numexpr-style fusion.

### `tabular_patterns.py`

Larger pattern set; this is the workhorse for pandas pipelines. Shared
helpers:

- `FuncManager` accumulates a collection of `FunctionRef`s (with their
  closures) and provides:
  - `closure_type()` — `RecordType([("f<id>", closure.type), …])`,
  - `closure_val(rewriter)` — a `MakeRecord` producing the combined
    closure,
  - `call(block, func, args, new_func)` — emits the inner `ir.Call`,
    threading the right slice of the combined closure out of the outer
    function's last argument.

Patterns registered by `rewrite_set_column(module)`:

| Pattern | Trigger | Output |
|---|---|---|
| `RewriteFromDict` | `table.from_dict(MakeRecord(col -> (table.get_column \| column.sequential)))` where every column comes from *one* common table | `table.select(common_table, columns=[…])` + `table.add_index_column` per sequential column |
| `RewriteAddIndexColumn` | `table.set_column(t, column.sequential(table.length(t)), name=n)` | `table.add_index_column(t, name=n)` |
| `RewriteSetComputedColumn` | `table.set_column(t, column.apply_scalar(table.get_column(t, c), f))` or the `column.binary_op` variant | `table.compute(t, f, input=[…], output=[new_col])` |
| `RewriteSetColumnRowApply` | `table.set_column(t, table.apply_row_wise_scalar(t, f))` | builds a wrapper function that materializes a row record from the required columns and calls `f` once per row; emits `table.compute(t, wrapper_ref, input=[cols], output=[col])` |
| `RewriteFilter` | `table.filter(t, <column-computation rooted at t>)` | walks the filter column's producer chain (handles `column.apply_scalar` / `column.binary_op` / `table.get_column`) to build a single `fused_filter_N` function, then emits `table.filter_by_func(t, fused_ref, columns=[…])` |
| `FuseColumnApply` | `column.apply_scalar(column.apply_scalar(c, g), f)` | fuses into a single `column.apply_scalar(c, compose(f,g))` via `FuncManager` |

Every fused function captures the pass-specific `fused_cntr` to get a
unique name. The resulting ops (`table.compute`, `table.filter_by_func`,
`table.add_index_column`) are the normalized relational primitives that
the backend consumes directly.

## 4. Pipeline ordering (as invoked by `hipy.compiler.compile` + backend)

Roughly, the order from raw generator output to the backend:

1. `canonicalize` — collapse `RecordGet(MakeRecord)`.
2. `inline.run` — inline regular `ir.Call`s to expose closure/FunctionRef
   patterns.
3. `eliminate_dead_code.run` — prune after inlining.
4. `canonicalize_array_ops` + `fuse_array_ops` — numpy fusion.
5. `rewrite_set_column` — normalize table ops to `table.compute` /
   `table.filter_by_func` / `table.add_index_column`.
6. `eager_free.run` — insert `ir.Free` after last uses.
7. `eliminate_dead_symbols.run(module, root)` — drop unreachable functions.
8. Backend codegen (`hipy.cppbackend.*`).

Whether all passes are wired into a given entry point is best checked by
grepping the backend driver — `hipy.cppbackend.__init__` and
`compile.py` are the places to look.

## 5. Gotchas for adding a pass

- **Every new pattern must keep `rewriter.uses` consistent.** Use
  `replace_with` / `replace_with_value` / `remove` helpers instead of
  poking `block.ops` directly. Forgetting an update causes `replace_with`
  to later miss rewrites on downstream ops.
- **`replace_with` asserts type equality on produced values.** If you're
  changing op semantics in a way that changes the result type, insert a
  separate `cast` op and use `replace_with_value` against it instead.
- **Patterns are tried top-down.** A pattern that fires on a parent op
  hides everything below it. If you add a general canonicalization
  pattern, make sure it runs in a separate rewriter from the fusion
  patterns, or you'll lose fusion opportunities.
- **`FuncManager` is stateful per fusion.** Don't share one between two
  independent rewrites — the closure-record slots get tangled.
- **`fused_cntr` is module-global inside each file.** If you add another
  fusion pass, use a new counter variable; reusing `fused_cntr` causes
  name collisions across passes.
