# Bugs discovered while adding coverage tests

Tests that exposed these bugs are marked with `@pytest.mark.xfail` or
commented out so they don't block CI, but the inputs stay in the repo
as reproducers.

---

## 1. `str.format()` with empty spec `"{}"` produces the empty string — **FIXED**

**Was:** `hipy/lib/builtins.py` — `_const_str.translate_python_spec_to_cpp`.

When the Python spec was empty (e.g. `"{}"`), the translator returned
`""` so the generated C++ call was `std::vformat("", arg)` — producing
an empty string instead of formatting `arg`.

**Fix:** return `"{}"` for the empty spec so `std::vformat` invokes the
default formatter for the argument.

**Regression test:** `test/test_string_format.py::test_format_basic`.

---

## 2. `str.__mod__` does not unpack a tuple RHS — **FIXED**

**Was:** `hipy/lib/builtins.py` — `_const_str.__mod__`.

`"%d %d" % (1, 2)` passed the tuple `(1, 2)` as the single positional
`__mod__` arg, so `args = ((1, 2),)` and `args[1]` raised `IndexError`.
Python's real `%` unpacks a tuple RHS into positional args; HiPy did
not.

**Fix:** `_const_str.__mod__` now detects a single-arg tuple RHS and
unpacks it before indexing.

**Regression test:** `test/test_string_format.py::test_percent_tuple_rhs`.

---

## 3. `df.groupby(by)[col].nunique()` trips `MultiIndex.__new__() got an unexpected keyword argument 'dtype'` — **FIXED**

**Was:** `hipy/lib/pandas/__init__.py` — `MultiIndex.__topython__` called
`pd.MultiIndex(self._cols, name=self.names, dtype=self.dtype)`. pandas
3.x's `MultiIndex()` constructor no longer accepts `dtype=`, so the
pybind materialization of any Series backed by a MultiIndex (including
the result of `df.groupby(by)[col].nunique()`) aborted with:

    TypeError: MultiIndex.__new__() got an unexpected keyword argument 'dtype'

**Fix:** `MultiIndex.__topython__` now builds the index with
`pd.MultiIndex.from_arrays(self._cols, names=self.names)`, the supported
constructor for "list of arrays + names".

**Regression test:** `test/pandas/test_groupby_merge.py::test_groupby_series_nunique`.

---

## 4. `str.format(...)` fallback breaks on pyobj-method lookup — **FIXED**

**Was:** `hipy/lib/builtins.py` — `_const_str.format` is a
`@hipy.compiled_function` whose body calls the hipy-internal raw method
`self._const_str__get_format_parts()`. When the inner parser raised
`NotImplementedError` (e.g. for `"{:n}"` or `"{:=8d}"`), the method-call
fallback kicked in *at the helper call site* and tried to resolve
`__get_format_parts` on a pyobj str — which doesn't exist — via a
typeshed lookup that raised `KeyError: '__get_format_parts'`.

**Fix:** `@hipy.raw` / `@hipy.compiled_function` now accept
`helper=True`. Functions marked helper skip the automatic pyobj
fallback entirely — their exceptions propagate to the caller, whose own
fallback is the one that should handle them. `__get_format_parts` and
`__get_percentage_format_parts` are marked helper; the exception now
bubbles up past `format()` and the outer fallback re-runs the call
against the real pyobj `str.format`.

**Regression tests:** `test/test_fallback_coverage.py::test_format_locale_n_falls_back`
and `::test_format_sign_aware_align_falls_back`.

---

## Optional dependency: `typeshed_client`

Several fallback paths (e.g. `str.__mod__` with an unsupported format
specifier) use `hipy/lib/builtins.py:103` to infer pyobj method return
types from typeshed stubs. Without `typeshed_client` installed, those
paths raise `ModuleNotFoundError` at generation time. The test harness
now installs `typeshed_client` in CI (`.github/workflows/test.yml`) so
these paths are exercised.
