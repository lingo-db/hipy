# Compiler — AST rewrite ("cogen") + whole-pipeline driver

**File:** `hipy/compiler.py` (≈1300 LOC)

`compiler.py` implements the second half of the paper's 4-stage pipeline:

1. `inspect.getsource` + `ast.parse` — get the source AST.
2. **Desugar** (`rewrite_func`) — turn break/continue/comprehensions/if-return
   into forms the rewriter can handle.
3. **Rewrite** (`stage_function`) — turn every expression and statement into
   calls on a runtime `_context` object (paper Table, §5). This is the
   `cogen` transformation.
4. **Compile + exec** (`stage_and_compile`) — `builtins.compile` the rewritten
   AST and execute it with a patched `__builtins__` (→ `hipy.lib.builtins`).
   The result is a Python function that takes an extra keyword arg `_context`
   and, when called, emits IR into that context.

A second top-level entry (`compile_function`, `compile`) then drives the
actual IR-generation run, including the post-mortem escape analysis
fix-point loop.

## 1. Key entry points

| Name | Purpose |
|---|---|
| `stage_and_compile(func)` | Rewrite + exec. Called lazily by `HLCFunction.get_compiled_fn` in `hipy/function.py`. |
| `stage_function(fn_ast, globals, nested, outer_context)` | Core AST rewrite. Handles default args, var-args, kw-args. Adds `_context` as last kw-only arg. |
| `stage_block(block, ctx)` / `stage_stmt(stmt, ctx)` / `stage_expr(expr, ctx)` | Recursive rewriters. |
| `compile(fn, arg_types, fallback, debug)` | Full pipeline: build `ir.Module`, create `Context`, run the generator, run escape analysis, rerun until fixpoint. |
| `compile_function(hlc_fn, arg_types, kw_types, module, fallback, debug)` | Inner loop of `compile` (see §4). |

## 2. `StageContext` (rewrite-time bookkeeping)

Populated fresh per function body being rewritten.

- `block` — growing Python-AST list the rewriter appends to (the rewritten body)
- `globals` — host module globals (resolved later at exec time)
- `available_variables` — tracks names in scope, used by free-variable binding
- `nested` — `True` if inside a nested body (if/else/for/while/try);
  controls how `return` is emitted — at top level it becomes a `return`,
  otherwise it becomes `_context.early_return(...)` so an outer
  `_context.EarlyReturn` catch can turn it into the final return.
- `action_id` — monotonic counter, attached to every `_context.<method>` call as
  a `_action_id=<int>` kwarg. These IDs are the stable identifier the
  post-mortem escape analysis uses to decide which allocation sites to
  force-convert on the next run.
- `current_functions` — set of names that *are the function currently being
  rewritten* (or outer ones), so `Name(id=…)` that resolves to a self-call
  is emitted as `_context.get_recursive_raw(module=__name__, func=…)` rather
  than an ordinary `get_by_name` lookup. This is what makes recursion work.

## 3. The rewrite — expression → `_context.*` call

`stage_context_call(name, kwargs, …)` produces `_context.<name>(**kwargs,
_action_id=N)`. Every Python construct maps to one. Highlights:

| Python construct | Rewritten call |
|---|---|
| `Constant(v)` | `_context.constant(val=v)` |
| `Name(id)` | `_context.get_by_name(val=<name>, name="id")` — or `_context.get_recursive_raw(...)` for self-recursion |
| `Attribute(e, a)` | `_context.get_attr(val=⟦e⟧, attr="a")` |
| `Call(f, args, kwargs)` | `_context.perform_call(fn=⟦f⟧, args=[…], kwargs={…})` — starred args go through `_context.const_unpack` first |
| `BinOp(l, op, r)` | `_context.perform_binop(left=⟦l⟧, right=⟦r⟧, left_method="__add__", right_method="__radd__")` (see `get_binop_methods`) |
| `Compare`, chained | Expanded into `And` of simple `Compare`s; `is`, `is not`, `in`, `not in` have dedicated hooks |
| `BoolOp(And/Or)` | `_context.bool_and/bool_or` (short-circuit-preserving; note these are left-associated) |
| `UnaryOp(Not/USub/Invert)` | `_context.bool_not / neg_ / invert_` |
| `IfExp` | Lowered to `_context._if(cond, bodyfn, elsefn, inputs=[…])` writing to a tmp var |
| `Lambda` / `FunctionDef` | Produces the 3-variant expansion (see §3.1) |
| `List/Tuple/Dict/Slice/Subscript/DictComp/ListComp` | Comprehensions are desugared to explicit `for` + `append`; the rest map to dedicated `_context` methods |
| `Assign` / `AugAssign` / `AnnAssign` | Via `Target` classes: `NameTarget`, `AttributeTarget`, `SubscriptTarget`, `TupleTarget` |
| `If` | `_context._if(cond, bodyfn, elsefn, inputs=[…])`, wrapped in `try: … except _context.EarlyReturn as e: …` |
| `While` / `For` | `_context._while(cond, body, read_only_inputs, iter_vals)` / `_context._for(over, target="tmpname", body, read_only_inputs, iter_vals)`. The iter target name is staged (so it's available inside the body) |
| `Try(… except …)` | Split into two nested functions (`_try`/`except` closures, staged) joined via `_context._try(tryfn, exceptfn, try_closure, except_closure)`; also wrapped in `except _context.EarlyReturn` |
| `Pass` | passed through |

`rewrite_func` runs ahead of the rewrite:
- `rewrite_if_return` — hoists the tail after an `if …: return` into the
  `else:` branch so returns stay at the block-tail invariant.
- `rewrite_loop_break` / `rewrite_loop_if_continue` — desugars `break` and
  `continue` using a synthesized boolean guard. This avoids modeling
  non-local control inside loops.

### 3.1 Lambdas and nested functions — three variants

Every `lambda` and every `FunctionDef` is expanded into **three** Python
functions emitted at the current scope:

1. `__hipy_lambda_fnN` — the staged (rewritten) version that is called when
   the lambda is used from other cogen code.
2. `__hipy_lambda_fn_explicit_stagedN` — like (1) but with free variables
   rewritten to `__closure__["name"]`. This is what gets materialized to a
   real `ir.Function` when the lambda is handed to a builtin that needs a
   callback.
3. `__hipy_lambda_fn_explicitN` — an **unrewritten** copy of (2), whose
   `ast.unparse`d source is stashed with the lambda. When the lambda must be
   handed to CPython (fallback), the generator installs this Python source
   into the IR module via `py_functions` so CPython can re-parse it at
   runtime.

The expression evaluates to
`_context.create_lambda(staged=fn1, bind_python=λ f: f(closure_obj, src, name),
bind_staged=λ f: f(closure_obj, ref_to_fn2))`.

Free variables are found by `BindFreeVars` (a NodeTransformer tracking names
that are in `available_variables` but not shadowed).

## 4. The IR-generation loop — `compile_function`

```
problematic = []; invalid_action_id = -1
while True:
    module.block.ops.clear()
    fn = ir.Function(module, name, [t.ir_type() for t in arg_types], ir.void)
    ctxt = Context(module, fn.body,
                   decisions=problematic,          # forced-convert sites
                   invalid_action_id=invalid_action_id,
                   debug=debug,
                   base_module=base_module)
    with ctxt.handle_action("args"):
        args = [ctxt.wrap(t.construct(v, ctxt)) for t, v in zip(arg_types, fn.args)]
    if not fallback: ctxt.no_fallback() wrapper, else direct:
        res = compiled_fn(*args, _context=ctxt)          # actually emits IR!
    ir.Return(fn.body, [...]); fn.res_type = res.type

    # Post-mortem escape analysis
    build DiGraph from ctxt.events of kinds:
        ValueAlias, NestedValueAlias, ConvertedToPython, ValUsage
    For every Usage: if a path exists to the `bad_node`, record its
        t_location as a new forced-early-conversion site.
    If no new sites: done. Else: loop.
```

The decisions list (`problematic`) persists across iterations; it's fed back
into the next `Context` as `decisions=`. The `invalid_action_id` also
decreases monotonically so the solver can distinguish rerun generations.

`Context`, `ValueAlias`, `NestedValueAlias`, `ConvertedToPython`, `ValUsage` all
live in `hipy/context.py` — see `context.md`.

## 5. Miscellaneous helpers

- `tmp_name_counter`, `get_tmp_name` — globally-unique tmp var names for
  rewritten assignment targets; essential for avoiding name clashes when
  multiple rewrites introduce new variables.
- `lambda_fn_counter`, `tmp_function_counter` — unique names for lambda and
  per-block helpers.
- `stage_tmp_function` / `plain_function` — emit a helper function alongside
  the rewritten body; used by `if`, `while`, `for`, `try` to package up the
  branch/body/condition into a callable the context can drive.
- `VariableAnalyzer` — `NodeVisitor` collecting read/written names from a
  subtree; used to compute `read_only_inputs` and `iter_vals` for loops and
  the `changed_variables` set for `if`/`try`.
- `LineNumberAdapter` — bumps the `lineno`/`col_offset` of the parsed
  source back to its *original* position in the host file, so Python's
  debugger (and stack traces) land on the correct source line after `exec`.

## 6. Gotchas for maintainers

- **Every new expression/statement rewrite must set `lineno` / `col_offset`
  on every AST node it constructs** — otherwise the exec-side `compile` will
  die with `Missing lineno`.
- **`_action_id` must be strictly monotonic per function.** If you add a
  `stage_context_call` variant that bypasses it, escape analysis breaks.
- **`current_functions` is populated per `stage_function` call** with the
  outer context's set plus the current name. If you add another recursion
  mechanism (e.g. method-inside-class), extend this set.
- **`nested=True` for inner bodies (if/for/while/try).** It changes the
  semantics of `return` (early_return vs top-level return) and of default
  args (only staged once, at top level).
- **`__builtins__` is overridden to `hipy.lib.builtins`** at exec time, which
  is how `int`, `str`, etc. resolve to HiPy virtual classes inside the
  generator. Changing this requires replacing the class lookup for every
  builtin the rewriter emits.
- **Unhandled AST kinds raise `NotImplementedError`.** See paper §8
  "Limitations" — coroutines, class defs in user code, `with`, pattern
  matching, generators, scoping keywords, string templates, decorators are
  all deliberately out of scope.
