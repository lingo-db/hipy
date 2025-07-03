import hipy
from hipy.interpreter import check_prints
from hipy.test_utils import not_constant
from urllib.parse import urlparse
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
