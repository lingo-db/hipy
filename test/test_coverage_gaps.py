"""Targeted tests for previously-uncovered library paths.

Each test names the file:line it was written to cover.
"""
import hipy
from hipy import intrinsics
import hipy.lib.datetime
import hipy.lib.re
import hipy.lib.urllib.parse
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
import re
import datetime
from urllib.parse import urlsplit, urlparse


# hipy/lib/re.py:92, 94 — `[` and `]` branches in _count_groups that
# flip in_char_class so `(` inside a character class is not counted as
# a capturing group.
@hipy.compiled_function
def fn_regex_char_class_group_counting():
    m = re.search(r"([abc])(\d+)", not_constant("x b42 y"))
    if m is not None:
        print(m.group(0))
        print(m.group(1))
        print(m.group(2))


def test_regex_char_class_group_counting():
    check_prints(fn_regex_char_class_group_counting, """
b42
b
42
""")


# hipy/lib/re.py:60 — _is_simple_regex returns False when a `\x` escape
# is not in the supported set (\\d, \\w, \\s, \\D, \\W, \\S). That
# forces the whole search through the Python fallback.
@hipy.compiled_function
def fn_regex_unsupported_escape_falls_back():
    # \b is a word-boundary anchor — supported in real re, not in the
    # simple-regex path.
    m = re.search(r"\bword\b", not_constant("a word here"))
    if m is not None:
        print(m.group(0))


def test_regex_unsupported_escape_falls_back():
    check_prints(fn_regex_unsupported_escape_falls_back, """
word
""", fallback=True)


# hipy/lib/datetime.py:55 — date.__hipy__repr__. `__hipy__repr__` is
# used by `repr(...)`.
@hipy.compiled_function
def fn_date_repr():
    d = datetime.date(2024, 5, 16)
    print(repr(d))


def test_date_repr():
    check_prints(fn_date_repr, """
datetime.date(2024, 5, 16)
""")


# hipy/lib/datetime.py:203 — timedelta.__hipy__repr__.
@hipy.compiled_function
def fn_timedelta_repr():
    t = datetime.timedelta(days=3, seconds=45)
    print(repr(t))


def test_timedelta_repr():
    check_prints(fn_timedelta_repr, """
datetime.timedelta(days=3, seconds=45)
""")


# hipy/lib/datetime.py:274-275 — timedelta.__bool__.
@hipy.compiled_function
def fn_timedelta_bool():
    a = datetime.timedelta(days=0, seconds=0)
    b = datetime.timedelta(days=1)
    if a:
        print("nonzero a")
    else:
        print("zero a")
    if b:
        print("nonzero b")
    else:
        print("zero b")


def test_timedelta_bool():
    check_prints(fn_timedelta_bool, """
zero a
nonzero b
""")


# hipy/lib/datetime.py:222-223 — timedelta.__neg__.
@hipy.compiled_function
def fn_timedelta_neg():
    t = datetime.timedelta(days=2, seconds=30)
    print(-t)


def test_timedelta_neg():
    check_prints(fn_timedelta_neg, """
-3 days, 23:59:30
""")


# hipy/lib/urllib/parse.py:145 — urlsplit not_implemented branch for
# non-str url (e.g. bytes). Falls back to CPython's urlsplit which
# handles bytes natively.
@hipy.compiled_function
def fn_urlsplit_bytes_falls_back():
    b = intrinsics.to_python(not_constant(b"https://host/path"))
    r = urlsplit(b)
    print(r.scheme)
    print(r.path)


def test_urlsplit_bytes_falls_back():
    check_prints(fn_urlsplit_bytes_falls_back, """
b'https'
b'/path'
""", fallback=True)


# hipy/lib/urllib/parse.py:173 — urlparse not_implemented branch for
# non-str url.
@hipy.compiled_function
def fn_urlparse_bytes_falls_back():
    b = intrinsics.to_python(not_constant(b"http://host/path;params?q=1"))
    r = urlparse(b)
    print(r.scheme)
    print(r.params)


def test_urlparse_bytes_falls_back():
    check_prints(fn_urlparse_bytes_falls_back, """
b'http'
b'params'
""", fallback=True)


# hipy/lib/builtins.py:2277-2280 — slice(start, stop, step) 3-arg
# form. Only the 1- and 2-arg forms were tested.
@hipy.compiled_function
def fn_slice_three_args():
    s = slice(1, 8, 2)
    print(s.start)
    print(s.stop)
    print(s.step)
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(xs[s])


def test_slice_three_args():
    check_prints(fn_slice_three_args, """
1
8
2
[1, 3, 5, 7]
""")
