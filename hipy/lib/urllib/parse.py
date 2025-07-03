__HIPY_MODULE__ = "urllib.parse"

import numpy as np
import hipy.lib.numpy
import sys

from hipy.internal_values import _named_tuple
from hipy.value import static_object, HLCFunctionValue
import hipy
from hipy import intrinsics, ir
from hipy.value import SimpleType, Value, Type, raw_module

import urllib.parse as _parse

hipy.register(sys.modules[__name__])
original = raw_module(_parse)


@hipy.classdef
class NetlocResultMixin:

    @hipy.compiled_function
    def _userinfo(self):
        netloc = self.netloc
        userinfo, have_info, hostinfo = netloc.rpartition('@')
        if have_info:
            username, have_password, password = userinfo.partition(':')
            if not have_password:
                password = None
        else:
            username = None
            password = None
        return username, password

    @hipy.compiled_function
    def _hostinfo(self):
        netloc = self.netloc
        _, _, hostinfo = netloc.rpartition('@')
        _, have_open_br, bracketed = hostinfo.partition('[')
        if have_open_br:
            hostname, _, port = bracketed.partition(']')
            _, _, port = port.partition(':')
        else:
            hostname, _, port = hostinfo.partition(':')

        if not port:
            port = None
        return hostname, port


    @hipy.compiled_function
    def __hipy_getattr__(self, name):
        if name == "username":
            return self._userinfo()[0]
        elif name == "password":
            return self._userinfo()[1]
        elif name == "hostname":
            return self._hostinfo()[0]
        elif name == "port":
            return self._hostinfo()[1]
        else:
            return self._super___hipy_getattr__(name)


@hipy.classdef
class ParseResult(_named_tuple, NetlocResultMixin):
    def __init__(self, elts, value=None):
        super().__init__(elts, "ParseResult", ["scheme", "netloc", "path", "params", "query", "fragment"], value=value)
        NetlocResultMixin.__init__(self)

    @staticmethod
    @hipy.raw
    def __create__(scheme, netloc, path, params, query, fragment, _context):
        return _context.wrap(ParseResult([scheme, netloc, path, params, query, fragment]))

    @hipy.compiled_function
    def __topython__(self):
        return _parse.ParseResult(self.scheme, self.netloc, self.path, self.params, self.query, self.fragment)


@hipy.classdef
class SplitResult(_named_tuple):
    def __init__(self, elts, value=None):
        super().__init__(elts, "ParseResult", ["scheme", "netloc", "path", "query", "fragment"], value=value)
        NetlocResultMixin.__init__(self)

    @staticmethod
    @hipy.raw
    def __create__(scheme, netloc, path, query, fragment, _context):
        return _context.wrap(ParseResult([scheme, netloc, path, query, fragment]))

    @hipy.compiled_function
    def __topython__(self):
        return _parse.SplitResult(self.scheme, self.netloc, self.path, self.query, self.fragment)

@hipy.compiled_function
def _splitnetloc(url, start=0):
    delim = len(url)   # position of end of domain part of url, default is end
    for c in '/?#':    # look for delimiters; the order is NOT important
        wdelim = url.find(c, start)        # find first of this delim
        if wdelim >= 0:                    # if found
            delim = min(delim, wdelim)     # use earliest delim position
    return url[start:delim], url[delim:]   # return (domain, rest)

@hipy.compiled_function
def urlsplit(url, scheme='', allow_fragments=True):
    if intrinsics.isa(url, str) and intrinsics.isa(scheme, str):
        scheme_chars = ('abcdefghijklmnopqrstuvwxyz'
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        '0123456789'
                        '+-.')
        _UNSAFE_URL_BYTES_TO_REMOVE = ['\t', '\r', '\n']

        #url = url.lstrip(_WHATWG_C0_CONTROL_OR_SPACE)
        #scheme = scheme.strip(_WHATWG_C0_CONTROL_OR_SPACE)

        for b in _UNSAFE_URL_BYTES_TO_REMOVE:
            url = url.replace(b, "")
            scheme = scheme.replace(b, "")

        allow_fragments = bool(allow_fragments)
        netloc = ''
        query = ''
        fragment = ''
        i = url.find(':')
        if i > 0 and url[0].isascii() and (('A'<= url[0]<='Z') or ('a'<=url[0]<='z')):# and url[0].isalpha():
            if len(url[:i])>0:
                scheme, url = url[:i].lower(), url[i+1:]
        if url[:2] == '//':
            netloc, url = _splitnetloc(url, 2)
            #if (('[' in netloc and ']' not in netloc) or
            #        (']' in netloc and '[' not in netloc)):
            #    raise ValueError("Invalid IPv6 URL")
            #if '[' in netloc and ']' in netloc:
            #    bracketed_host = netloc.partition('[')[2].partition(']')[0]
            #    _check_bracketed_host(bracketed_host)
        if allow_fragments and '#' in url:
            url, fragment = url.split('#', 1)
        if '?' in url:
            url, query = url.split('?', 1)
        #_checknetloc(netloc)
        v = SplitResult(scheme, netloc, url, query, fragment)
        return v
    else:
        intrinsics.not_implemented()


@hipy.compiled_function
def _splitparams(url):
    if '/'  in url:
        i = url.find(';', url.rfind('/'))
        if i < 0:
            return url, ''
        else:
            return url[:i], url[i+1:]
    else:
        i = url.find(';')
        return url[:i], url[i+1:]
@hipy.compiled_function
def urlparse(url, scheme='', allow_fragments=True):
    uses_params = ['', 'ftp', 'hdl', 'prospero', 'http', 'imap',
                   'https', 'shttp', 'rtsp', 'rtsps', 'rtspu', 'sip',
                   'sips', 'mms', 'sftp', 'tel']
    if intrinsics.isa(url, str) and intrinsics.isa(scheme, str):
        splitresult = urlsplit(url, scheme, allow_fragments)
        scheme, netloc, url, query, fragment = splitresult
        if scheme in intrinsics.as_abstract(uses_params) and ';' in url:
            url, params = _splitparams(url)
        else:
            params = ''
        return ParseResult(scheme, netloc, url, params, query, fragment)
    else:
        intrinsics.not_implemented()