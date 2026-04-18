import os
import sys

import hipy.compiler as compiler
from hipy import cppbackend

if "DBPY_BACKEND" in os.environ and os.environ["DBPY_BACKEND"].lower() == "lingodb":
    from dbpy import lingodbbackend


def check_prints(fn, str,fallback=False, debug=None):
    if debug is None:
        if "HIPY_DEBUG" in os.environ:
            debug = bool(os.environ["HIPY_DEBUG"])
        else:
            debug = True
    module = compiler.compile(fn,fallback=fallback,debug=debug)#, forward_error=True,fallback___topython__=fallback)
    cout, cerr, rc = cppbackend.run(fn.get_name(), module)
    if rc != 0:
        print(f"Error running cpp standalone (exited with {rc}):", file=sys.stderr)
        print(cerr, file=sys.stderr)
        print(cout, file=sys.stderr)
        assert False
    if len(cerr) > 0:
        print(cerr, file=sys.stderr)
        assert False
    if cout.strip() != str.strip():
        print(f"Expected:\n{str}\nGot:\n{cout}", file=sys.stderr)
        assert False
