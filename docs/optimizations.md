# Optimization passes — `hipy/opt/`

**Files:** `pattern_rewriter.py`, `canonicalization.py`,
`eliminate_dead_code.py`, `eliminate_dead_symbols.py`, `inline.py`,
`eager_free.py`, `array_patterns.py`, `tabular_patterns.py`,
`rewrite_cpp.py`, `dccg.py`.

After the generator has emitted IR via `hipy/compiler.py` + `hipy/context.py`,
the module passes through a sequence of optimization passes before the
backend lowers it further. Passes are thin: most are a handful of pattern
match arms around a `PatternRewriter`. The heavy lifting lives in
`dccg.py` (data-centric codegen), which is where pandas/DataFrame workloads
get their JIT-equivalent speed via operator-fusion.

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
e.g., `FuseColumnApply` and the dccg rewriter rely on seeing the raw
`FunctionRef`-to-closure chains, not abstract calls.

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
`dccg.py` then consumes.

## 4. `dccg.py` — data-centric code generation

**This is the "query compiler" pass.** Name comes from the Neumann-style
"data-centric compilation" idea (Produce/Consume operator model). Handles
pandas/table pipelines in one go: takes a DAG of `table.compute` /
`table.filter_by_func` / `table.select` / `table.add_index_column` /
`table.join_inner` / `table.join_left` / `table.aggregate` ops rooted at
some "materialization" boundary (a non-table op that consumes a table)
and produces a *single* C++-backend pipeline that:

1. Allocates any hash tables / aggregation hash tables / builders,
2. Scans base tables once,
3. Pipes tuples through joins, filters, maps without intermediate table
   materialization,
4. Emits exactly one final `TableBuilderFinish` at the end.

### 4.1 The operator tree

Every relational op becomes a Python class implementing
`produce(required_cols, block, module, parent)` and
`consume(cols, block, module)` — the classic Neumann "produce/consume"
interface.

| Class | Role | Emits |
|---|---|---|
| `TableScan(table)` | Leaf: iterate a base table | `cppir.IterateTable` |
| `InnerJoin(left, right, left_key, right_key, left_keep, right_keep, …)` | Hash join; right side builds, left probes | `cppir.CreateJoinHt`, `JoinHtInsert`, `JoinHtBuild`, `JoinHtLookup` |
| `LeftJoin(…)` | Same as InnerJoin + a `cppir.CreateFlag`/`SetFlag`/`CheckFlag` for null padding | Adds `ir.IfElse` with NaN constants on the right cols |
| `Aggregation(group_by, input, output, child, init_fn, agg_fn, finalize_fn, …)` | Hash aggregation with init/accumulate/finalize | `cppir.CreateAggregationHt`, `Aggregate`, `IterateAggregationHt` |
| `Project(cols, child)` | Pass-through (erases uncols in `required_cols` propagation) | — |
| `AddIndexColumn(col, child)` | Appends a running counter column | `cppir.CreateCounter`, `IncrementCounter` |
| `Filter(required_cols, child, filter_fn)` | Predicate | `ir.IfElse` wrapping the consume |
| `Map(required_cols, output_cols, child, map_fn)` | Scalar computation | Calls `map_fn`, unpacks record if multi-output |
| `Materialize(required_cols, child)` | Sink: accumulates results into a table builder | `cppir.CreateTableBuilder`, `TableBuilderAppend`, `TableBuilderFinish` |

Pattern: the root of the tree is always a `Materialize`, and leaves are
always `TableScan`s. `produce()` walks from root to leaves (to allocate
state in order); `consume()` walks back up through the `parent` pointer
(to wire tuples through).

### 4.2 The tree builder — `RelRewriter`

`RelRewriter.construct_tree(v, rewriter, first=False)` recursively turns a
`CallBuiltin` DAG rooted at the SSA value `v` into the operator tree
above. Highlights:

- The `first` flag is set at the root — the root becomes a `Materialize`,
  ensuring a table builder is emitted at the top.
- `sort_irelevant` forwards sort semantics through operators that don't
  care about row order (e.g. `Aggregation` consumes its input in any
  order).
- **Multi-use guard.** If a value has more than one use in the surrounding
  module, the rewriter builds a sub-tree for it, *materializes it
  separately* (emits a `Materialize` for just that sub-tree), and then
  treats it as a `TableScan` of the new materialized table. This avoids
  the combinatorial blow-up of inlining a shared subplan twice.
- Unknown producers bottom out as `TableScan(v)` — a leaf that just
  iterates the already-materialized table.

### 4.3 The tree optimizer — `optimize(tree)`

Tiny; currently limited to:

- Pulling `Map`s above `InnerJoin`s when the mapped cols aren't join
  keys, so map evaluation happens after the join reduces the data.
- Pulling `Filter`s above `InnerJoin`s unconditionally (filters always
  commute with inner join over the non-join side).

Both rewrites recurse into the lifted operator's `.child`, so the
optimizer cascades.

### 4.4 The trigger — `RelRewriter.rewrite(op)`

Only fires on ops that *consume* a table but are not themselves
relational producers (i.e., the "boundary" between relational and
non-relational). Explicit bail-outs for each table-producing CallBuiltin
prevent the pattern from matching mid-tree.

For each table-typed arg of the triggering op, it builds the tree,
optimizes it, emits `produce([], …)` into a fresh insertion point before
the arg's producer, and replaces the arg's producer result with the tree's
materialized output. The `IfElse` bail-out (`case ir.IfElse: return False`)
keeps the pass from mangling structured control flow.

`rewrite(module)` is the public entry.

## 5. `rewrite_cpp.py` — final IR → cppir lowering

After all optimization passes have run, `rewrite_cpp.rewrite(module)`
lowers the two remaining control-flow builtins to the C++ backend's
equivalents:

- `range.iter(func, read_only, iter_vals, start, end, step)` →
  `cppir.IterRange` with an inner block that unpacks `iter_vals`,
  calls `func` per iteration, and yields the updated iter_vals record.
- `while.iter(cond_func, body_func, read_only, iter_vals)` →
  `cppir.WhileIter` with a `cond_block` (calls `cond_func`, yields bool) and
  `iter_block` (calls `body_func`, yields new iter_vals).

There's also a `FusePyMethodCall` pattern (defined but not in the default
pattern list of `rewrite`) that folds
`PythonCall(PyGetAttr(on, method), args)` into a single
`cppir.PyMethodCall(on, method_str, args)` when the `PyGetAttr` has only
that one use. Enable by adding it to the `PatternRewriter` list if
measured useful.

## 6. Pipeline ordering (as invoked by `hipy.compiler.compile` + backend)

Roughly, the order from raw generator output to `cppir`:

1. `canonicalize` — collapse `RecordGet(MakeRecord)`.
2. `inline.run` — inline regular `ir.Call`s to expose closure/FunctionRef
   patterns.
3. `eliminate_dead_code.run` — prune after inlining.
4. `canonicalize_array_ops` + `fuse_array_ops` — numpy fusion.
5. `rewrite_set_column` — normalize table ops to `table.compute` /
   `table.filter_by_func` / `table.add_index_column`.
6. `dccg.rewrite` — consume the relational DAG, emit a fused pipeline.
7. `rewrite_cpp.rewrite` — lower `range.iter` / `while.iter` to `cppir`.
8. `eager_free.run` — insert `ir.Free` after last uses.
9. `eliminate_dead_symbols.run(module, root)` — drop unreachable functions.
10. Backend codegen (`hipy.cppbackend.*`).

Whether all passes are wired into a given entry point is best checked by
grepping the backend driver — `hipy.cppbackend.__init__` and
`compile.py` are the places to look.

## 7. Gotchas for adding a pass

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
- **DCCG assumes a single materialization boundary per tree.** If the
  relational DAG has multiple consumers of the same intermediate table,
  the multi-use guard builds a sub-materialization; but chained complex
  DAGs can still miss fusion opportunities. Watch for
  `table.get_column`/`table.length` at the consumer boundary if a
  rewrite seems to stop short.
- **`FuncManager` is stateful per fusion.** Don't share one between two
  independent rewrites — the closure-record slots get tangled.
- **`fused_cntr` is module-global inside each file.** If you add another
  fusion pass, use a new counter variable; reusing `fused_cntr` causes
  name collisions across passes.
