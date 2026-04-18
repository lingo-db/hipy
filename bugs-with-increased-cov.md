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

## 2. `str.__mod__` does not unpack a tuple RHS

**Location:** `hipy/lib/builtins.py:1382-1390` — `_const_str.__mod__`.

`"%d %d" % (1, 2)` passes the tuple `(1, 2)` as the single positional
`__mod__` arg. `__mod__(self, *args)` receives `args = ((1, 2),)`
and indexes `args[1]`, raising `IndexError`.

Python's real `%` unpacks a tuple RHS automatically; HiPy's
implementation does not.

**Reproducer:** `test/test_string_format.py::test_percent_escape_and_multiple` (xfail).

    "100%% of %d is %d" % (50, 50)   # IndexError: tuple index out of range

**Fix sketch:** in `__mod__`, detect a tuple RHS (single arg that is a
tuple) and treat it as positional args.

---

## 3. `df.groupby(by)[col].nunique()` trips `MultiIndex.__new__() got an unexpected keyword argument 'dtype'`

**Location:** `hipy/lib/pandas/__init__.py` — `DataFrameGroupBySeriesGroupBy.nunique`
(around line 301) combined with the eventual `__topython__` /
`print` conversion path. The standalone binary aborts with:

    terminate called after throwing an instance of 'pybind11::error_already_set'
      what():  TypeError: MultiIndex.__new__() got an unexpected keyword argument 'dtype'

Inner aggregate steps succeed; the crash happens when the resulting
Series is materialized via pybind back to a pandas object — something
in the Series→pandas path is passing `dtype=` to a MultiIndex ctor
that no longer accepts it under pandas 3.x.

**Reproducer:** `test/pandas/test_groupby_merge.py::test_groupby_series_nunique` (xfail).

    df.groupby(["k"])["v"].nunique()    # aborts with the above

**Fix sketch:** in the Series-from-groupby topython path, stop passing
`dtype=` to `MultiIndex.__new__` (pandas 3.x removed that kwarg).

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
