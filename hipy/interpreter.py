import subprocess
import os
import hipy.ir as ir
import hipy.compiler as compiler
#import dbpy.passes
#import dbpy.translator
import pathlib
import json
import sys

from hipy import cppbackend

if "DBPY_BACKEND" in os.environ and os.environ["DBPY_BACKEND"].lower() == "lingodb":
    from dbpy import lingodbbackend


def run_interpreter(function_name, module: ir.Module):
    if module is None:
        return "", "module was none", -100
    as_json = json.dumps(module.serialize())
    json_file = f"{pathlib.Path(__file__).parent.resolve()}/../program.dbpyir.json"
    with open(json_file, "w") as program_file:
        program_file.write(as_json)
    binary = f"{pathlib.Path(__file__).parent.resolve()}/../../interpreter/cmake-build-debug/interpreter"
    if "DBPYIR_INTERPRETER" in os.environ:
        binary = os.environ["DBPYIR_INTERPRETER"]
    command = [binary, function_name, json_file]

    try:
        print(':'.join(sys.path))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={"PYTHONPATH": ':'.join(sys.path)})
        stdout, stderr = process.communicate()
        return stdout.decode('utf-8'), stderr.decode('utf-8'), process.returncode
    except Exception as e:
        return None, str(e), -1


def show(result):
    stdout, stderr, returncode = result
    if returncode != 0:
        print(f"Error running interpreter (exited with {returncode}):", file=sys.stderr)
        print(stderr, file=sys.stderr)
        print(stdout)
        return
    print(stdout)
    print(stderr, file=sys.stderr)


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
