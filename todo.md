# TODO — open hacks and things that still need care

Collected while walking every commit between `main` and `claude-init`
(32 commits) to refresh the docs. Each item is something that smells
like a temporary shortcut, a hard-coded assumption, or an
under-specified contract — worth returning to before calling the
corresponding feature "done".

## Module-level globals / mutable state

- **`hipy/config.py` → `function_suffix`.** Global mutable string used
  by `compile_function` to disambiguate generated symbol names
  (`compile.py` passes it in per invocation). Makes concurrent compiles
  in the same process interfere. A context manager or a field on
  `Module` would be cleaner.
- **`hipy/mlirbackend/__init__.py` → `curr_context` / `curr_module`.**
  Source is already tagged `todo: remove this hack`. Hold the MLIR
  context for the duration of a compile; prevent concurrent compiles
  from coexisting.

## MLIR backend

- **`VoidType` lowers to `i1`.** MLIR has no void, so we use a dummy.
  Ripples through `ir.Constant` (void) → UndefOp and
  `ir.Call` (void return) asserts-false.
- **`list.sort` returns `util.UndefOp`.** The op is emitted for side
  effect only — the SSA result is garbage. If `list.sort` ever becomes
  non-mutating, revisit `to_mlir_stmt` for `list.sort`.
- **`dict.create` closure error message says "Dict creation with closure
  type not supported"** — the `list.sort` case copy-pasted the same
  message verbatim (see `hipy/mlirbackend/__init__.py`). Minor but
  misleading in a debugger.
- **`ir.Call` with void return → assert False.** Known gap.
- **`ir.PythonCall` keyword args → assert False.** Positional only.
- **`ir.PySetAttr`, `ir.Undef`, `ir.Free`** — not handled in the
  top-level dispatch.
- **Unhandled `CallBuiltin` now raises `assert False`** (older
  behaviour was "emit UndefOp and keep going"). Fine for correctness,
  but any tooling that relied on partial compilation will break.
- **Composite types not lowered:** `ColumnType`, `TableType`,
  `ArrayType`. This is the biggest gap for database workloads — the
  whole point of the MLIR backend.
- **`sql.execute` requires a literal string query.** `str_from_const`
  asserts if the first arg isn't an `ir.Constant(str)`. Works for
  hand-written queries but blocks any compile-time string building.

## C++ backend

- **`dict.create` receives an `eq_fn` arg that the C++ backend ignores.**
  See `hipy/cppbackend/__init__.py` around the `dict.create` handler —
  it falls back to `std::make_shared<std::unordered_map<>>()` and
  silently drops the custom equality function. MLIR wires the fn
  through; C++ does not. This will matter if any user type ever relies
  on custom `__eq__` / `__hash__` semantics in a dict.
- **`sql.execute` is not implemented** in the C++ backend. Only MLIR
  has a handler. Currently C++ compilation of code that uses
  `hipy.lib.sql.execute` will fail at the backend step.

## Standard library

- **`hipy/lib/re.py` regex pattern whitelist.** `_is_simple_regex` only
  accepts literal chars, `\d \w \s`, `.`, quantifiers `? * + {n,m}`,
  anchors `^ $`, `(...)` groups, and escaped punctuation. Any other
  metacharacter (character classes `[abc]`, alternation `|`, lookahead,
  named groups, …) → `intrinsics.not_implemented()`. No runtime
  fallback — generator-time refusal. The whitelist is intentionally
  conservative; expand it as the C++ / LingoDB regex runtimes gain
  features.
- **`_MaybeNone` lives in `hipy/lib/builtins.py`.** It's the Python-like
  "either T or None" virtual used by regex `search` to return a
  `Match | None`. Logically it belongs in an optional / maybe module —
  consider moving it once the shape of "optional result" values
  stabilises.
- **`hipy/lib/sql.py` is a stub.** `Nullable.__topython__` raises; the
  C++ backend doesn't handle `sql.execute`; the whole module exists
  only to produce MLIR + LingoDB relational ops. Works end-to-end only
  via the MLIR backend.
- **`hipy/lib/datetime.py` is tiny.** Only `date.__sub__(date)` → `timedelta`
  and `timedelta.days` → `int`. No constructors (`date(y, m, d)`), no
  `date.today()`, no week/month/year helpers, no `__topython__`. Enough
  for SQL-style date arithmetic but not much else.
- **`math.ceil` types the builtin as `int`** but the IR name is
  `scalar.float.ceil`. The C++ / MLIR side is responsible for rounding
  the `float` result to an integer consistently — implicit convention.
- **`str.__iadd__` path** was added for coverage but still routes
  through regular string concatenation (no real in-place mutation at
  the IR level, since `str` is immutable). Make sure this stays in
  sync with any future Python-side expectation of `str += x` being
  cheap.

## In-source `todo:` comments

- `hipy/cppbackend/__init__.py` — a couple of `todo:` comments mark
  "cast to str" assumptions in `str.join` that don't verify the
  element type is actually string-like.
- `hipy/mlirbackend/__init__.py` substring correctness — the `+1`
  offset shift relies on LingoDB's 1-indexed Substring runtime.
  Documented in the doc's Gotchas, but worth auditing if any new
  string op lands with different indexing.
- `hipy/mlirbackend/__init__.py` `curr_context` / `curr_module` —
  already mentioned above; source-tagged `todo: remove this hack`.
- `hipy/mlirbackend/__init__.py` `to_mlir_func` — `todo: translate
  function signature` comment above the sig construction; unclear what
  aspect is incomplete.

## Testing / coverage holes

- **No tests for `sql.execute` via the C++ backend.** (It can't work;
  that's the point, but there's no explicit skip either.)
- **`hipy/lib/re.py` tests cover only the whitelist.** Anything that
  lands outside `_is_simple_regex` will fall through
  `not_implemented()` at compile time — no end-to-end test guards
  against accidentally broadening the whitelist.
- **`_MaybeNone` semantics** are only exercised indirectly via regex.
  If you use it for anything else, write dedicated tests first.
