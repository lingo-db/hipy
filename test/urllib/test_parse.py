import hipy
from hipy import intrinsics
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
from urllib.parse import urlparse, urlsplit
import hipy.lib.urllib.parse

@hipy.compiled_function
def fn_urlparse():
    print(urlparse(not_constant('//www.cwi.nl:80/%7Eguido/Python.html')))
    print(urlparse(not_constant('www.cwi.nl/%7Eguido/Python.html')))
    print(urlparse(not_constant('help/Python.html')))
    print(urlparse(not_constant("http://docs.python.org:80/3/library/urllib.parse.html?"
             "highlight=params#url-parsing")))

def test_urlparse():
    check_prints(fn_urlparse, """
ParseResult(scheme='', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html', params='', query='', fragment='')
ParseResult(scheme='', netloc='', path='www.cwi.nl/%7Eguido/Python.html', params='', query='', fragment='')
ParseResult(scheme='', netloc='', path='help/Python.html', params='', query='', fragment='')
ParseResult(scheme='http', netloc='docs.python.org:80', path='/3/library/urllib.parse.html', params='', query='highlight=params', fragment='url-parsing')
""", debug=False)


@hipy.compiled_function
def fn_urlparse_hostname_port():
    # NetlocResultMixin.__hipy_getattr__ dispatches to _hostinfo -> hostname/port.
    r = urlparse(not_constant("http://docs.python.org:80/3/library/urllib.parse.html"))
    print(r.hostname)
    print(r.port)
    r2 = urlparse(not_constant("https://example.com/path"))
    print(r2.hostname)


def test_urlparse_hostname_port():
    check_prints(fn_urlparse_hostname_port, """
docs.python.org
80
example.com
""")


@hipy.compiled_function
def fn_urlparse_userinfo():
    # NetlocResultMixin._userinfo -> username/password split via rpartition('@') + partition(':').
    r = urlparse(not_constant("https://user:pass@example.com/path"))
    print(r.username)
    print(r.password)
    r2 = urlparse(not_constant("https://user@example.com/path"))
    print(r2.username)
    print(r2.password)
    r3 = urlparse(not_constant("https://example.com/path"))
    print(r3.username)
    print(r3.password)


def test_urlparse_userinfo():
    check_prints(fn_urlparse_userinfo, """
user
pass
user
None
None
None
""")


@hipy.compiled_function
def fn_urlsplit_basic():
    # urlsplit returns a ParseResult (reuses its ctor) but without params.
    r = urlsplit(not_constant("http://example.com/path?q=1#frag"))
    print(r.scheme)
    print(r.netloc)
    print(r.path)
    print(r.query)
    print(r.fragment)


import pytest


def test_urlsplit_basic():
    check_prints(fn_urlsplit_basic, """
http
example.com
/path
q=1
frag
""")


@hipy.compiled_function
def fn_urlparse_topython():
    # ParseResult.__topython__ converts to a real urllib.parse.ParseResult via fallback.
    r = urlparse(not_constant("http://host/path"))
    p = intrinsics.to_python(r)
    print(p.scheme)
    print(p.path)


def test_urlparse_topython():
    check_prints(fn_urlparse_topython, """
http
/path
""", fallback=True)


@hipy.compiled_function
def fn_urlsplit_topython():
    # SplitResult.__topython__ (hipy/lib/urllib/parse.py:94) — mirrors the
    # ParseResult case but for the shorter 5-field SplitResult produced by
    # urlsplit directly.
    r = urlsplit(not_constant("https://host:8080/path?x=1#frag"))
    p = intrinsics.to_python(r)
    print(p.scheme)
    print(p.netloc)
    print(p.query)
    print(p.fragment)


def test_urlsplit_topython():
    check_prints(fn_urlsplit_topython, """
https
host:8080
x=1
frag
""", fallback=True)
